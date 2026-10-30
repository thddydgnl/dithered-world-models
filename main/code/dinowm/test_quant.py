"""Unit checks of the quantizer kernels and sharing structure (no model): bias and MSE in grid units, number of levels,
sharing of the dither across the candidates of a group, freshness across forward calls, compiled vs eager kernels.
usage: bash run.sh test_quant.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qlib as QL


class FakeMod(torch.nn.Module):
    def forward(self, x):
        return x


class FakeModel:
    """Minimal stand-in with predictor.transformer.layers[i][1].net[2] = identity module (the GELU hook site)."""

    def __init__(self, n=6):
        class Blk(list):
            pass
        layers = []
        for _ in range(n):
            ff = torch.nn.Module(); ff.net = [None, None, FakeMod()]
            layers.append([None, ff])
        self.predictor = type('P', (), {})(); self.predictor.transformer = type('T', (), {})()
        self.predictor.transformer.layers = layers


def main():
    torch.manual_seed(0)
    lo, hi, k = -0.17, 2.0, 3
    nl = (1 << k) - 1; d = (hi - lo) / nl
    G, N, Lt, C = 2, 64, 49, 256
    x = (lo + (hi - lo) * torch.rand(G * N, Lt, C, device='cuda')).to(torch.bfloat16)
    for use_compile in (True, False):
        m = FakeModel()
        Q = QL.Quantizer(m, use_compile=use_compile)
        Q.set_ranges([(lo, hi)] * 6)
        mod = m.predictor.transformer.layers[0][1].net[2]
        print(f'--- compile={use_compile}')
        for var in ('det', 'ind', 'shiter', 'shstep', 'shep', 'detshift', 'shfixed'):
            Q.configure(var, k); Q.begin_plan([11, 12], 'test')
            errs, outs = [], []
            for it in range(40):
                Q.it, Q.h = it, 0
                y = mod(x)
                outs.append(y)
                errs.append(((y.float() - x.float()) / d))
            E = torch.stack(errs)                    # (draws, B, Lt, C) in grid units
            xs = ((x.float() - lo) / d)
            levels = torch.unique(((outs[0].float() - lo) / d).reshape(-1)).numel()
            # per-entry bias across the 40 calls (fresh variants -> ~0), overall MSE
            bias_entry = E.mean(0)
            g0 = outs[0].reshape(G, N, Lt, C)
            same_input_rows = None
            # sharing check: rows of one group with the same input value get the same output?
            xx = x.reshape(G, N, Lt, C)
            xr = xx[:, :1].expand_as(xx).reshape(G * N, Lt, C)        # every row = row 0 of its group
            Q.it = 999
            yr = mod(xr).reshape(G, N, Lt, C)
            shared = bool((yr == yr[:, :1]).all())
            fresh = not torch.equal(outs[0], outs[1])
            print(f'{var:8s} mse {float((E ** 2).mean()):.4f} (det 1/12=0.0833, SR 1/6=0.1667) mean err {float(E.mean()):+.4f} '
                  f'|entry bias| {float(bias_entry.abs().mean()):.4f} levels(call0) {levels} '
                  f'same-input rows share output {shared} fresh across calls {fresh} max|err| {float(E.abs().max()):.3f}')
        Q.remove()


if __name__ == '__main__':
    main()
