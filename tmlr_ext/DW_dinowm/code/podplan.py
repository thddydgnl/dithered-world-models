"""Pod plan of the DINO-WM group (dw) of paper A's TMLR extension: cells, rules and job ids (DESIGN_tmlr.md v1.0
sections 1.2, 2.2, 3.2, 6, 8.1; operational choices in DW_dinowm/PLAN.md). Pure python, no torch.

Job = one runner.run_job unit: (task, split, planner, ranges, variant, k, episode chunk). Ids:
  <split>_none_<ptag>_e<e0>-<e1>                    none (k-free; shared by every experiment of the pod, R5)
  <split>_<ranges>_<var>_k<k>_<ptag>_e<e0>-<e1>     quantized
  ptag = 'cem' (DINO-WM CEM of E1) or 'mppi<c>' (E16 MPPI-like update with lam = c x first-iteration elite cost std)
Ranges ids: 'main' (PT = E0 ranges_main), 'pc' (per channel), 'rc_k<k>' (rollout-chosen at k). Bias id '<ranges>_k<k>'.
Variant code names: sh-plan = shstep (as in E1), det+BC = detbc, det+shift-fixed = detshiftfix, sh-iter+b = shiterb,
det+noise = detnoise, det+shift = detshift.
"""
TAU, EPS = 0.10, 1e-9
NBOOT = 10000
E1_NONE = {'pusht': 0.83, 'wall': 0.61, 'pointmaze': 0.85}       # E1 none success (results/E1/e1_dinowm_cells.csv)
CHECK3_TOL = 0.10                                                # DESIGN 0.7-3: within 10 points of E1's none
K1 = {'pusht': 4, 'pointmaze': 3, 'wall': 3}                     # E11 k (DESIGN 1.2)
BATCH_E = {'pusht': 1, 'wall': 2, 'pointmaze': 1}                # E1 batch sizes (fixed per task across variants)
CHUNK = {'pusht': 10, 'wall': 20, 'pointmaze': 10}               # episodes per job (E1)
N_EVAL, N_HELD = 100, 20
E11_VARS = ('det', 'detbc', 'shstep', 'detshiftfix', 'shiterb', 'shiter', 'detshift', 'detnoise')
E12_ROUND = {'main': ('det', 'detbc', 'shiter'), 'pc': ('det', 'detbc', 'shiter'), 'rc': ('det', 'shiter')}
E12_GRID = {'pusht': {'main': (2, 3, 4), 'pc': (2, 3, 4), 'rc': (3, 4)},
            'pointmaze': {'main': (2, 3), 'pc': (2, 3)}, 'wall': {'main': (2, 3), 'pc': (2, 3)}}
TWO_STAGE = ('pusht',)                                           # E12: 50 episodes first, the rest if the CI holds tau
STAGE1 = 50
K_MIN, K_MAX = 1, 8
E16_VARS = ('det', 'shiter', 'shstep', 'detshift')
E16_K = 3
E16_C = (0.25, 0.5, 1.0, 2.0)

# Pods (DESIGN 8.1 with the operational details of PLAN.md). Calibration artifacts (PC ranges, percentile grid, bias
# vectors of the initial grid, RC ranges of the initial grid) are computed once per task by its owner pod and copied to
# the other pods of the task; extension-k artifacts are computed by the pod that runs the extension.
#   e12: grid k per ranges kind run on this pod (the E11 cells are reused for PT at K1, rule R1)
#   ext: extension directions handled here ('up' from the grid's highest k, 'down' from its lowest)
#   e13_part: None = every E13 configuration of the task here; 'main_ext' = only PT configurations above K1 (the
#             PT up-extension cells of this pod); 'rest' = all the others (2026-11-21T20:0xZ split of dw.1's queue onto
#             dw.5 at the coordinator's request; each pod uses its own none states)
PODS = {
    'dw.1': dict(tasks=('pusht',), owner=('pusht',), e11=('pusht',),
                 e12={'pusht': {'main': (4,)}}, ext={'pusht': ('up',)},
                 e16=(), e13=('pusht',), e13_part='main_ext'),
    'dw.5': dict(tasks=('pusht',), owner=(), e11=(),
                 e12={'pusht': {'pc': (4,), 'rc': (4,)}}, ext={'pusht': ('up',)},
                 e16=(), e13=('pusht',), e13_part='rest'),
    # [post hoc] (coordinator, 2026-11-22T01:36Z, user approved): the E11 cell PushT 3-bit (PT, eval8, none + 8 variants,
    # E11 diagnostics), its own none; no E12, extension, E16 or E13 cells. e11_k overrides K1 for this pod only.
    'dw.6': dict(tasks=('pusht',), owner=(), e11=('pusht',), e11_k={'pusht': 3},
                 e12={}, ext={}, e16=(), e13=(),
                 # X17 (D5, DESIGN_tmlr 11.1, coordinator 2026-11-22T02:08Z): PushT 4-bit on E1's eval6 split, after D4
                 x17=dict(task='pusht', split='eval6', k=4, ranges='main', vars=('det', 'shiter'), eps=(0, 100))),
    'dw.2': dict(tasks=('pusht',), owner=(), e11=(),
                 e12={'pusht': {'main': (2, 3), 'pc': (2, 3), 'rc': (3,)}}, ext={'pusht': ('down',)},
                 e16=(), e13=()),
    'dw.3': dict(tasks=('pointmaze',), owner=(), e11=('pointmaze',),
                 e12={'pointmaze': {'main': (3,), 'pc': (3,)}}, ext={'pointmaze': ('up',)},
                 e16=('pointmaze',), e13=('pointmaze',)),
    'dw.4': dict(tasks=('wall', 'pointmaze'), owner=('wall', 'pointmaze'), e11=('wall',),
                 e12={'wall': {'main': (2, 3), 'pc': (2, 3)}, 'pointmaze': {'main': (2,), 'pc': (2,)}},
                 ext={'wall': ('up', 'down'), 'pointmaze': ('down',)}, e16=('wall',), e13=('wall',)),
}


def plan_for(tag):
    """dw.3b continues dw.3 (same GPU type, DESIGN 0.3): the base tag's plan."""
    base = tag.rstrip('bcdefgh') if tag[-1].isalpha() else tag
    return PODS[base]


def rid_of(kind, k):
    return f'rc_k{k}' if kind == 'rc' else kind


def ptag_of(planner):
    return 'cem' if not planner else f"mppi{planner['mppi_c']:g}"


def job(task, split, var, k, e0, e1, ranges='main', diag=False, planner=None, exp=()):
    pt = ptag_of(planner)
    if var == 'none':
        jid = f'{split}_none_{pt}_e{e0:03d}-{e1:03d}'
        k = 0
    else:
        jid = f'{split}_{ranges}_{var}_k{k}_{pt}_e{e0:03d}-{e1:03d}'
    j = dict(id=jid, task=task, split=split, var=var, k=int(k), ep0=int(e0), ep1=int(e1), E=BATCH_E[task],
             ranges=ranges, diag=bool(diag), exp=list(exp))
    if var in ('detbc', 'shiterb'):
        j['bias'] = f'{ranges}_k{k}'
    if planner:
        j['planner'] = dict(planner)
    return j


def chunks(task, e0, e1):
    c = CHUNK[task]
    return [(s, min(s + c, e1)) for s in range(e0, e1, c)]


def cell_id(task, kind, var, k, planner=None):
    return f'{task}|{rid_of(kind, k)}|{var}|k{k}|{ptag_of(planner)}'


def grid_cells(tag, task):
    """E12 cells of this pod (kind, var, k), without the E11 cells (R1)."""
    P = plan_for(tag)
    out = []
    for kind, ks in P['e12'].get(task, {}).items():
        for k in ks:
            for var in E12_ROUND[kind]:
                if kind == 'main' and k == K1[task] and task in P['e11'] and var in E11_VARS:
                    continue
                out.append((kind, var, k))
    return out


def owner_bias_needs(task):
    """(ranges id, k) bias vectors the owner computes: every (PT, PC) k of the task's initial E12 grid (all pods) and
    the E11 k. Extension k are computed by the pod that runs them."""
    ks = set(E12_GRID[task]['main']) | set(E12_GRID[task]['pc']) | {K1[task]}
    return [(rid, k) for rid in ('main', 'pc') for k in sorted(ks)]


def owner_rc_needs(task):
    return sorted(E12_GRID[task].get('rc', ()))
