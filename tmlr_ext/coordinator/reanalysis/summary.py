"""Assemble summary.txt (Korean, plain text) from r1.json ... r5.json. Run after r1-r5."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

H = C.HERE
R1 = json.load(open(os.path.join(H, "r1.json")))
R2 = json.load(open(os.path.join(H, "r2.json")))
R3 = json.load(open(os.path.join(H, "r3.json")))
R4 = json.load(open(os.path.join(H, "r4.json")))
R5 = json.load(open(os.path.join(H, "r5.json")))
L = []


def p(s=""):
    L.append(s)


def pr(x):
    return f"{x:.3f}"


def pts(x):
    return f"{100 * x:+.1f}"


def ci_pts(ci):
    return f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"


def dist(d):
    return " ".join(f"{k}:{100 * v:.1f}%" for k, v in sorted(d.items(), key=lambda kv: int(kv[0])))


p("A TMLR 재분석 R1~R5 요약 (조정자, 2026-11-22, pod 없음, 이 Mac의 결과 파일만 사용)")
p("폴더: A/tmlr_ext/coordinator/reanalysis/  (common.py, r1~r5 스크립트, summary.py; 출력 r*.json, r*.csv)")
p("실행: python3 r1_kstar_G_bootstrap.py; r2_tau_sensitivity.py; r3_G_per_method.py; r4_dw_pooled_e1_eval8.py; "
  "r5_persistence_cost.py; summary.py (각 2초 안팎)")
p()
p("== 0. 재현 (먼저 확인, 실패 시 멈추도록 r1이 검사) ==")
p(f"- TD-MPC2 E12 칸 {R1['meta']['reproduced_cells']}개의 n, D, k*가 td_e12_cells.csv와 모두 같다. G(12과제 x PT/PC), "
  "Fr1 7/8, Fr2 7/8 재현.")
p("- DINO-WM G 재현: PushT 2/2/2(PT/PC/RC), Wall 2/2, PointMaze 0/0 (dw.1~dw.5; dw.6은 [사후] D4/D5 pod라 제외).")
nbad = sum(1 for c in R5["checks"] if not c[4])
p(f"- R5 대비값 {len(R5['checks'])}개 대조: 불일치 {nbad}. TD 5M은 td_e11_contrasts.csv와 5e-7 안, 317M은 verify_mt_e14.txt"
  "의 소수 셋째 자리 반올림 안, DINO-WM은 analysis_dw.json과 정확히 같다. R4의 분할별 값도 E1 csv·analysis_dw와 같다.")
p("- 재표본 장치 점검: R1의 첫 과제(push) 가중 행렬이 발표된 칸별 CI를 그대로 재현한다.")
p()
p("== 1. R1: k*와 G의 bootstrap 분포 ==")
p("- 방법: 과제마다 count 가중 행렬 하나(10,000 x 짝)를 모든 범위·반올림·k가 함께 쓴다(같은 짝). 과제끼리는 독립.")
p("  RandomState(0), TD는 MW8+DMC4 순서, DW는 PushT, PointMaze, Wall 순서로 뽑았다.")
p("  두 단계 칸: DMC는 단계별 층화(1단계 45짝 = 각 체크포인트 episode 0~14, 2단계 45짝)로, 45짝 칸은 1단계 45개 draw를")
p("  그대로 쓰고 90짝 칸과 공유한다. PushT도 같은 방식(쌍 0~49 / 50~99). 점검: 층화하지 않은 bootstrap(반쪽 칸은 제 반에")
p("  떨어진 draw만 사용)과 PushT 덩어리(10개) bootstrap.")
p("  검열: 가장 낮은 잰 k가 이미 통과하면 그 k, 아무 k도 통과하지 못하면 '가장 높은 k + 1'(경계 규약). 열린 쪽을 1~9로 둔")
p("  구간으로 P(G>=1)의 하한·상한도 셌다(아래 [하한, 상한]).")
fr = R1["fr"]
for b, nm in (("best", "나은 dither(발표 기준)"), ("shiter", "sh-iter 단독"), ("detshift", "det+shift 단독")):
    a, c = fr[b]["Fr1"], fr[b]["Fr2"]
    p(f"- {nm}: Fr1 점 {a['count_point']}/8, P(성립) {pr(a['P_holds'])} [{pr(a['P_holds_lo'])}, {pr(a['P_holds_hi'])}], 과제 수 분포 {dist(a['count_dist'])}")
    p(f"  {' ' * len(nm)}  Fr2 점 {c['count_point']}/8, P(성립) {pr(c['P_holds'])} [{pr(c['P_holds_lo'])}, {pr(c['P_holds_hi'])}], 과제 수 분포 {dist(c['count_dist'])}")
p("- MW 과제별 G (점) P(G>=1) [하한, 상한], PT | PC:")
for t in C.MW8:
    a, c = R1["td"][t]["PT|best"], R1["td"][t]["PC|best"]
    p(f"  {C.SHORT[t]:13s} PT G {a['G_point']} P {pr(a['P_ge1'])} [{pr(a['P_ge1_lo'])},{pr(a['P_ge1_hi'])}] 분포 {dist(a['dist'])}")
    p(f"  {'':13s} PC G {c['G_point']} P {pr(c['P_ge1'])} [{pr(c['P_ge1_lo'])},{pr(c['P_ge1_hi'])}] 분포 {dist(c['dist'])}")
p("- DMC 과제별 (층화 / 비층화 점검):")
for t in C.DMC4:
    for rk in ("PT", "PC"):
        a = R1["td"][t][f"{rk}|best"]
        p(f"  {C.SHORT[t]:10s} {rk} G {a['G_point']} P(G>=1) {pr(a['P_ge1'])} / {pr(a['unstratified']['P_ge1'])} 분포 {dist(a['dist'])}")
p("- DINO-WM E12 (짝 bootstrap; PushT는 비층화·덩어리 점검 병기):")
for t in C.DW_TASKS:
    for rk in C.DW_RANGES[t]:
        a = R1["dw"][t][rk]
        extra = ""
        if t == "pusht":
            extra = (f" | 비층화 P(G>=1) {pr(a['unstratified']['P_ge1'])}, 덩어리 P(G>=1) {pr(a['cluster']['P_ge1'])} "
                     f"P(G>=2) {pr(a['cluster']['P_ge2'])}")
        p(f"  {C.SHORT[t]:9s} {rk} G {a['G_point']} P(G>=1) {pr(a['P_ge1'])} [{pr(a['P_ge1_lo'])},{pr(a['P_ge1_hi'])}] "
          f"P(G>=2) {pr(a['P_ge2'])} 분포 {dist(a['dist'])}{extra}")
shaky = [(C.SHORT[t], rk, R1["td"][t][f"{rk}|best"]["P_ge1"]) for t in C.MW8 for rk in ("PT", "PC")
         if 0.02 < R1["td"][t][f"{rk}|best"]["P_ge1"] < 0.98]
pu = R1["dw"]["pusht"]
p("- 읽기: Fr1·Fr2는 재표본에서도 거의 항상 성립한다. 어느 과제가 G<1인지는 바뀐다. P(G>=1)이 0.02~0.98인 MW 칸: "
  + ", ".join(f"{t} {rk} {v:.2f}" for t, rk, v in shaky) + ".")
p(f"  PushT의 G>=1은 세 범위 모두 튼튼하지만 G=2는 PT만 튼튼하다(P(G>=2) PT {pu['PT']['P_ge2']:.2f}, PC {pu['PC']['P_ge2']:.2f}, "
  f"RC {pu['RC']['P_ge2']:.2f}). RC의 하한 {pu['RC']['P_ge1_lo']:.2f}는 RC det k2를 재지 않아 생긴 검열이다")
p(f"  (재표본의 {100 * pu['kstar|RC|det']['P_cens_dn']:.0f}%에서 RC det k3가 문턱을 통과). Wall은 0~2가 맞다. PointMaze는 0.")
p()
p("== 2. R2: tau 민감도 (점 추정; 괄호는 검열 구간) ==")
p("  tau      | Fr1: 나은/sh-iter/det+shift | Fr2: 나은/sh-iter/det+shift | DW G: PushT PT,PC,RC; Wall PT,PC; PointMaze PT,PC")


def fc(x):
    return f"{x['count']}" + (f"[{x['count_lo']},{x['count_hi']}]" if x["count_lo"] != x["count_hi"] else "")


def gc(x):
    return f"{x['G']}" + (f"[{x['G_lo']},{x['G_hi']}]" if x["G_lo"] != x["G_hi"] else "")


for tk in R2["meta"]["taus"]:
    T = R2["tau"][tk]
    f1 = "/".join(fc(T["fr"][f"Fr1|{b}"]) for b in ("best", "shiter", "detshift"))
    f2 = "/".join(fc(T["fr"][f"Fr2|{b}"]) for b in ("best", "shiter", "detshift"))
    dw = (",".join(gc(T["dw"]["pusht"][rk]) for rk in ("PT", "PC", "RC")) + "; " +
          ",".join(gc(T["dw"]["wall"][rk]) for rk in ("PT", "PC")) + "; " +
          ",".join(gc(T["dw"]["pointmaze"][rk]) for rk in ("PT", "PC")))
    p(f"  {tk:11s}| {f1:28s}| {f2:28s}| {dw}")
p("  (Fr1 문턱 6/8, Fr2 문턱 4/8. 0.10_strict = 'D < 0.10'.)")
p("- 모든 tau에서 Fr1·Fr2가 세 기준 모두 성립한다. sh-iter 단독은 tau 0.05, 0.10, 0.125에서 Fr1이 문턱과 같은 6/8,")
p("  tau <= 0.075에서 Fr2가 문턱과 같은 4/8이다. 나은 dither의 개수는 모든 tau에서 det+shift 단독과 같다.")
near = R2["tau"]["0.1"]["near"]
p(f"- tau 0.10에서 |D - tau| <= 0.01인 칸 {len(near)}개(전체 목록은 r2_near_tau.csv):")
for r in near:
    p(f"  {r[0]} {C.SHORT.get(r[1], r[1])} {r[2]} {r[3]} k{r[4]} n{r[5]} D {r[6]:.4f}")
p("- 정확히 0.100인 MW 칸 4개 중 G를 바꾸는 것은 pick-place PC det+shift k2 하나다. 'D < 0.10'이면 pick-place PC G가")
p("  1 -> 0이 되어 Fr2가 6/8(여전히 성립)이고, door-open PC det k2 때문에 door-open PC G는 1 -> 2가 된다.")
p("- DINO-WM: PushT PT G=2는 모든 tau에서 같다. PushT PC는 tau >= 0.125에서 1, PushT RC는 tau 0.20에서 1,")
p("  Wall PT는 tau >= 0.125에서 0, Wall PC는 tau >= 0.15에서 0 이하, PointMaze는 0(tau 0.05·0.20의 PC만 1).")
p()
p("== 3. R3: 방법별 G (tau 0.10, 점; P는 R1과 같은 재표본의 P(G>=1)) ==")
for unit in ("MW", "DMC"):
    for rk in ("PT", "PC"):
        s = " | ".join(f"{nm} {R3['counts'][f'{unit}|{rk}|{b}']['n_ge1']}/{R3['counts'][f'{unit}|{rk}|{b}']['n']}"
                       for b, nm in (("best", "나은"), ("shiter", "sh-iter"), ("detshift", "det+shift")))
        p(f"- {unit} {rk}: {s}")
        for b, nm in (("shiter", "sh-iter"), ("detshift", "det+shift")):
            g = R3["counts"][f"{unit}|{rk}|{b}"]["G"]
            ts = C.MW8 if unit == "MW" else C.DMC4
            pv = ", ".join(f"{C.SHORT[t]} {g[C.SHORT[t]]}({R3['td'][t][f'{rk}|{b}']['P_ge1']:.2f})" for t in ts)
            p(f"    {nm:9s}: {pv}")
for b, nm in (("shiter", "sh-iter 단독"), ("detshift", "det+shift 단독")):
    a, c = R3["fr_by_basis"][b]["Fr1"], R3["fr_by_basis"][b]["Fr2"]
    p(f"- {nm}: P(Fr1 성립) {pr(a['P_holds'])}, P(Fr2 성립) {pr(c['P_holds'])}")
p("- DINO-WM은 sh-iter뿐이라 발표된 G가 곧 sh-iter 단독 값이다(PushT 2/2/2, Wall 2/2, PointMaze 0/0).")
p()
p("== 4. R4 [사후]: DINO-WM E1 + eval8 합산 (짝 아님: E1 RTX A5000, eval8 RTX 4090; 역분산 고정효과) ==")
for t, nm in (("pusht", "PushT 4-bit"), ("pointmaze", "PointMaze 3-bit"), ("wall", "Wall 3-bit(E1 탐색 칸)")):
    e = R4["cells"][t]
    p(f"- {nm}: none E1 {100 * e['none_E1']:.0f}%, eval8 {100 * e['none_eval8']:.0f}%")
    for st, lab in (("D_det", "D_det"), ("red_shiter", "red(sh-iter)")):
        x = e[st]
        f = (lambda v: f"{100 * v:+.1f}") if st == "D_det" else (lambda v: f"{v:.2f}")
        p(f"    {lab:12s} E1 {f(x['E1']['est'])} [{f(x['E1']['ci'][0])}, {f(x['E1']['ci'][1])}] (se {f(x['E1']['se']).strip('+')}) | "
          f"eval8 {f(x['eval8']['est'])} [{f(x['eval8']['ci'][0])}, {f(x['eval8']['ci'][1])}] (se {f(x['eval8']['se']).strip('+')}"
          f"{', D_det*<=0 재표본 ' + str(x['eval8']['n_dropped']) + '개 제외' if x['eval8']['n_dropped'] else ''}) | "
          f"합산 {f(x['pooled']['est'])} [{f(x['pooled']['ci'][0])}, {f(x['pooled']['ci'][1])}], "
          f"Q {x['pooled']['Q']:.2f} (p {x['pooled']['p_Q']:.3f}), I2 {100 * x['pooled']['I2']:.0f}%, 가중 E1 {x['pooled']['weights'][0]:.2f}")
    if "sensitivity_eval8_cluster" in e:
        s = e["sensitivity_eval8_cluster"]["D_det"]
        p(f"    점검(eval8 덩어리 분산): D_det 합산 {pts(s['pooled']['est'])} {ci_pts(s['pooled']['ci'])}, Q {s['pooled']['Q']:.2f} "
          f"(p {s['pooled']['p_Q']:.3f}), I2 {100 * s['pooled']['I2']:.0f}%")
    p(f"    점검(red = 합산 (D_det - D_sh) / 합산 D_det): {e['red_ratio_of_pooled']['red']:.2f}")
p("- 읽기: PushT 4-bit는 두 분할이 이질적이라(I2 86%) 합산값 하나로 쓰면 안 된다. PointMaze 3-bit는 이질성이 없고")
p("  합산 D_det의 CI가 0을 넘는다([사후], 짝 아닌 두 분할). Wall은 두 분할이 같은 크기다. red의 합산은 비율 추정의")
p("  꼬리가 길어 정규 근사가 거칠다(특히 PointMaze; PushT eval8은 D_det가 0 근처라 가중이 0.01).")
p()
p("== 5. R5: 지속 비용 표 (MW·DINO-WM은 성공률 점, DMC는 상대 return %) ==")
p("- TD-MPC2 5M E11 (90짝): D(sh-ep)-D(sh-iter) [95% CI] | Delta_pers [95% CI]")
for t in C.MW8 + C.DMC4:
    x = R5["td5m"][t]
    p(f"  {C.SHORT[t]:13s} k{x['k']}: {pts(x['shep_minus_shiter']['est'])} {ci_pts(x['shep_minus_shiter']['ci'])} | "
      f"{pts(x['delta_pers']['est'])} {ci_pts(x['delta_pers']['ci'])}")
p("- TD-MPC2 5M E16 CEM식(참고): " + ", ".join(
    f"{C.SHORT[t]} {pts(x['shep_minus_shiter']['est'])} {ci_pts(x['shep_minus_shiter']['ci'])}" for t, x in R5["td5m_cem"].items()))
p("- 317M E14 (90 seed): D(sh-ep)-D(sh-iter)")
for t, x in R5["mt"].items():
    p(f"  {C.SHORT[t]:13s} {pts(x['shep_minus_shiter']['est'])} {ci_pts(x['shep_minus_shiter']['ci'])}")
p("- DINO-WM (eval8, 100쌍): D(sh-plan)-D(sh-iter) | Delta_pers")
for key, x in R5["dw"].items():
    t, pl = key.split("|")
    s = f"  {C.SHORT[t]:9s} {'CEM(E11)' if pl == 'cem' else 'MPPI식(E16)':11s}: {pts(x['shplan_minus_shiter']['est'])} {ci_pts(x['shplan_minus_shiter']['ci'])}"
    if "delta_pers" in x:
        s += f" | {pts(x['delta_pers']['est'])} {ci_pts(x['delta_pers']['ci'])}"
    if "cluster_ci" in x:
        s += (f"  (D1 덩어리 CI: {ci_pts(x['cluster_ci']['shplan_minus_shiter'])} | {ci_pts(x['cluster_ci']['delta_pers'])})")
    p(s)
lb = R5["td_lower_bounds"]
p("- 나란히: TD-MPC2 CI 하한 (작은 순)")
p("  5M MW sh-ep-sh-iter: " + ", ".join(f"{n} {100 * v:+.1f}" for v, n in lb["mw5m_shep"]))
p("  5M MW Delta_pers:    " + ", ".join(f"{n} {100 * v:+.1f}" for v, n in lb["mw5m_pers"]))
p("  5M DMC sh-ep-sh-iter: " + ", ".join(f"{n} {100 * v:+.1f}%" for v, n in lb["dmc5m_shep"]) +
  " / Delta_pers: " + ", ".join(f"{n} {100 * v:+.1f}%" for v, n in lb["dmc5m_pers"]))
p("  317M sh-ep-sh-iter:  " + ", ".join(f"{n} {100 * v:+.1f}" for v, n in lb["mt317m_shep"]))
p("  DINO-WM CI 상한과 비교 (그 상한보다 하한이 큰 TD 칸 수):")
for s in R5["side_by_side"]:
    t, pl = s["dw_cell"].split("|")
    mt = "" if s["n_mt317m_lower_above"] is None else f", 317M {s['n_mt317m_lower_above']}/8"
    p(f"  {C.SHORT[t]:9s} {pl:6s} {s['contrast']:22s} 상한 {100 * s['ci_hi']:+.1f} -> 5M MW {s['n_mw5m_lower_above']}/8{mt}")
td_all = ([(f"5M {C.SHORT[t]} sh-ep", x["shep_minus_shiter"]["ci"][0]) for t, x in R5["td5m"].items()] +
          [(f"5M {C.SHORT[t]} pers", x["delta_pers"]["ci"][0]) for t, x in R5["td5m"].items()] +
          [(f"5M CEM {C.SHORT[t]} sh-ep", x["shep_minus_shiter"]["ci"][0]) for t, x in R5["td5m_cem"].items()] +
          [(f"317M {C.SHORT[t]} sh-ep", x["shep_minus_shiter"]["ci"][0]) for t, x in R5["mt"].items()])
td_not = [n for n, lo in td_all if not lo > C.TOL]
dw_all = ([(k, "sh-plan", x["shplan_minus_shiter"]["ci"]) for k, x in R5["dw"].items()] +
          [(k, "pers", x["delta_pers"]["ci"]) for k, x in R5["dw"].items() if "delta_pers" in x])
dw_pos = [f"{C.SHORT[k.split('|')[0]]} {'CEM' if k.endswith('cem') else 'MPPI식'} "
          f"{'Delta_pers' if c == 'pers' else 'D(sh-plan)-D(sh-iter)'}" for k, c, ci in dw_all if ci[0] > C.TOL]
his = [ci[1] for k, c, ci in dw_all if c == "sh-plan"]
p(f"- 읽기: TD-MPC2는 대비 {len(td_all)}개 중 {len(td_all) - len(td_not)}개가 CI 하한 > 0 (예외: {', '.join(td_not) or '없음'}).")
p(f"  DINO-WM은 대비 {len(dw_all)}개 중 하한 > 0이 {', '.join(dw_pos) or '없음'}뿐이다(짝 bootstrap; PushT 덩어리 CI는 위).")
p(f"  다만 DINO-WM D(sh-plan)-D(sh-iter)의 상한({100 * min(his):+.0f}~{100 * max(his):+.0f}점)이 TD-MPC2의 가장 작은 하한"
  f"(5M {lb['mw5m_shep'][0][1]} {100 * lb['mw5m_shep'][0][0]:+.1f}, {lb['mw5m_shep'][1][1]} {100 * lb['mw5m_shep'][1][0]:+.1f}; "
  f"317M {lb['mt317m_shep'][0][1]} {100 * lb['mt317m_shep'][0][0]:+.1f})보다 크다.")
p("  두 계열의 구간은 큰 TD 칸과만 갈리고, 작은 TD 칸과는 겹친다.")
p()
f1, f2 = R1["fr"]["best"]["Fr1"], R1["fr"]["best"]["Fr2"]
s1, s2 = R1["fr"]["shiter"]["Fr1"], R1["fr"]["shiter"]["Fr2"]
ph = R5["dw"]["pusht|cem"]
pm, pq = R4["cells"]["pointmaze"]["D_det"]["pooled"], R4["cells"]["pusht"]["D_det"]["pooled"]
cmax = max(x["shplan_minus_shiter"]["ci"][1] for k, x in R5["dw"].items() if not k.startswith("pusht"))
cmin = min(x["shplan_minus_shiter"]["ci"][1] for k, x in R5["dw"].items() if not k.startswith("pusht"))
p("== 6. CLAIMS v2와 비교 (판정 변경 없음; 문구 후보) ==")
p(f"- 사전 지정 판정(Fr1, Fr2, P2, B2 등)을 바꾸는 결과는 없다. P(Fr1) {f1['P_holds']:.3f}, P(Fr2) {f2['P_holds']:.3f}, "
  f"sh-iter 단독도 {s1['P_holds']:.3f}, {s2['P_holds']:.3f}.")
p(f"- C4 'PushT 2(PT·PC·RC 모두)': G>=1은 셋 모두 튼튼하지만 P(G>=2)는 PT {pu['PT']['P_ge2']:.3f}, PC {pu['PC']['P_ge2']:.3f}, "
  f"RC {pu['RC']['P_ge2']:.3f}이고,")
p("  PC는 tau >= 0.125에서 1이다. 'PushT PT 2 bit, PC·RC 1~2 bit' 또는 'PushT 세 범위 모두 1 bit 이상(PT 2)'이 정확하다.")
p("- C4 '7/8(나은 쪽 기준)': det+shift 단독도 PT·PC 7/8로 같다. 즉 7/8은 det+shift 단독 값과 같고(K1 불성립으로")
p(f"  공동 처방이 아님), sh-iter 단독 6/8·5/8도 Fr1·Fr2 문턱을 넘는다(P {s1['P_holds']:.3f}, {s2['P_holds']:.3f}). 함께 적을 수 있다.")
p("  Fr2의 pick-place PC G=1은 D = 0.100(문턱과 정확히 같음)인 det+shift k2 칸 하나에 걸려 있다.")
p(f"- C2/C3 DINO-WM 지속 비용: C3의 'CI 상한 +4~+10점'(PointMaze·Wall 4칸: {100 * cmin:+.0f}~{100 * cmax:+.0f})은 재현된다. 그러나")
p(f"  PushT 4-bit E11(CEM)의 D(sh-plan)-D(sh-iter)는 {pts(ph['shplan_minus_shiter']['est'])} {ci_pts(ph['shplan_minus_shiter']['ci'])}이고 "
  f"D1 덩어리 CI는 {ci_pts(ph['cluster_ci']['shplan_minus_shiter'])}로 0을 넘는다.")
p("  DINO-WM 전체를 두고 '지속 비용이 0과 구별되지 않는다'고 쓰려면 PushT 칸을 밝히거나 범위를 PointMaze·Wall로 한정해야")
p("  한다(이 칸은 eval8에서 det 손상이 없다).")
p("- '두 계열이 다르다'는 'TD-MPC2는 CI 하한 > 0, DINO-WM은 CI가 0 포함(Wall Delta_pers 제외)'로 쓰는 것이 맞다. DINO-WM")
p("  상한이 TD-MPC2의 작은 칸 하한보다 크므로 'DINO-WM의 지속 비용이 TD-MPC2보다 작다'를 칸마다 주장할 수는 없다.")
p(f"- C1 PointMaze 3-bit: [사후] 합산 {pts(pm['est'])} {ci_pts(pm['ci'])}, I2 {100 * pm['I2']:.0f}%. 원고에 [사후] 기술로 덧붙일 수 있다.")
p(f"  PushT 4-bit는 두 분할이 이질적(Q {pq['Q']:.1f}, p {pq['p_Q']:.3f}, I2 {100 * pq['I2']:.0f}%)이므로 합산하지 말고 나란히 싣는 지금")
p("  방침이 맞다.")
p("- Wall 'G 0~2', PointMaze 0, DMC 'G 대부분 0'은 R1·R2와 맞는다. sh-iter 단독이면 walker PT G = -1이다.")
p()
p("== 7. 재현하지 못한 것과 한계 ==")
p("- 재현 못 한 발표값은 없다.")
p("- E1 PushT(eval6)의 덩어리 구조 파일이 없어 R4의 E1 쪽은 짝 bootstrap만 썼다.")
p("- 검열: 확장 규칙이 tau 0.10의 점 추정으로 잰 k를 정했으므로, 재표본·다른 tau에서는 잰 범위 밖의 k*가 필요할 수 있다.")
p("  TD에서는 P(G>=1)의 하한·상한이 거의 같았고, PushT RC(하한 0.80)와 tau 0.05의 몇 칸만 크게 열린다.")
p("- 두 단계 칸의 중간 확인(1단계 CI가 tau를 포함하면 2단계 실행)은 bootstrap이 다시 흉내 내지 않는다(설계대로 미보정).")
p("- 덩어리 bootstrap은 덩어리 10개라 분산을 과소 추정할 수 있다(PushT sh-plan 대비에서 짝 CI보다 좁아짐).")
with open(os.path.join(H, "summary.txt"), "w") as f:
    f.write("\n".join(L) + "\n")
print("\n".join(L))
