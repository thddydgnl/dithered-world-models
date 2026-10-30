"""E5 QAT (design 6.5, short budget): fine-tune the DINO-WM predictor with fake-quantized GELU outputs (the E1 site,
det rounding at k bits, straight-through gradient) on the task's training split with the DINO-WM training loss
(z_loss = latent MSE of the predicted next frames; frozen DINOv2 encoder, frozen action/proprio encoders, predictor
dropout 0.1 as in the original training). Ranges: the E1 per-tensor ranges ('main'). bf16 autocast, AdamW.
Resumable (checkpoint every --save_every steps). Output: MAIN/qat/<task>_<name>/{predictor.pt, state.pt, log.json}.
The evaluation runs through runner.py with job['ckpt'] = <name>.
usage: python qat.py <task> --k 4 --name qat_k4 [--steps 8000] [--bs 32] [--lr 5e-5] [--workers 12] [--max_minutes 120]
The budget is whichever of --steps and --max_minutes comes first; minutes count across resumed sessions. A finished
run (log.json final) is not trained again."""
import os, sys, json, time, argparse
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wmlib as L
import qlib as QL
from runner import load_ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task'); ap.add_argument('--k', type=int, default=4); ap.add_argument('--name', default=None)
    ap.add_argument('--steps', type=int, default=8000); ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max_minutes', type=float, default=120); ap.add_argument('--save_every', type=int, default=500)
    ap.add_argument('--log_every', type=int, default=50); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    name = a.name or f'qat_k{a.k}'
    out = f'{L.MAIN}/qat/{a.task}_{name}'
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    L.patch_attention(True)
    import hydra
    T = L.load_task(a.task)
    m, cfg = T['model'], T['cfg']
    datasets, _ = hydra.utils.call(cfg.env.dataset, num_hist=cfg.num_hist, num_pred=cfg.num_pred, frameskip=cfg.frameskip)
    train = datasets['train']
    g = torch.Generator().manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                     pin_memory=True, persistent_workers=a.workers > 0, generator=g)
    for p in m.predictor.parameters():
        p.requires_grad_(True)
    Q = QL.Quantizer(m, use_compile=True)
    Q.set_ranges(load_ranges(a.task, 'main'))
    Q.configure('det', a.k); Q.begin_plan([0], 'qat'); Q.ste = True
    opt = torch.optim.AdamW(m.predictor.parameters(), lr=a.lr)
    step, log = 0, dict(task=a.task, k=a.k, name=name, lr=a.lr, bs=a.bs, ranges='main', losses=[])
    if os.path.exists(f'{out}/state.pt'):                      # resume
        st = torch.load(f'{out}/state.pt', map_location='cuda')
        m.predictor.load_state_dict(st['predictor']); opt.load_state_dict(st['opt']); step = st['step']
        log = json.load(open(f'{out}/log.json'))
        print(f'resumed at step {step}', flush=True)
    m.predictor.train()
    t0 = time.time(); run = []
    prev_min = float(log.get('minutes', 0))       # training minutes of earlier sessions (the budget is cumulative)

    def minutes():
        return prev_min + (time.time() - t0) / 60

    def save(final=False):
        m.predictor.eval()
        torch.save(m.predictor.state_dict(), f'{out}/predictor.pt.tmp'); os.replace(f'{out}/predictor.pt.tmp', f'{out}/predictor.pt')
        torch.save(dict(predictor=m.predictor.state_dict(), opt=opt.state_dict(), step=step), f'{out}/state.pt.tmp')
        os.replace(f'{out}/state.pt.tmp', f'{out}/state.pt')
        log.update(step=step, final=final, minutes=minutes(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   gpu_name=torch.cuda.get_device_name(0), max_minutes=a.max_minutes, max_steps=a.steps)
        L.jdump(log, f'{out}/log.json')
        m.predictor.train()
    if log.get('final'):
        print(f'QAT_DONE {a.task} {name} already final at step {step} ({log.get("minutes", 0):.1f} min) -> {out}', flush=True)
        return
    done = False
    while not done:
        for obs, act, state in dl:
            obs = {k: v.cuda(non_blocking=True) for k, v in obs.items()}
            act = act.cuda(non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, _, _, loss, lc = m(obs, act)
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            opt.step()
            step += 1
            run.append(float(loss))
            if step % a.log_every == 0:
                rec = dict(step=step, loss=sum(run) / len(run), z_visual=float(lc['z_visual_loss']),
                           z_proprio=float(lc['z_proprio_loss']), min=minutes())
                log['losses'].append(rec); run = []
                print(json.dumps(rec), flush=True)
            if step % a.save_every == 0:
                save()
            if step >= a.steps or minutes() >= a.max_minutes:
                done = True
                break
    save(final=True)
    print(f'QAT_DONE {a.task} {name} steps {step} in {minutes():.1f} min -> {out}', flush=True)


if __name__ == '__main__':
    main()
