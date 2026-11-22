"""Formats REPORT.md section 12 (X6, D7) from results/analysis_x6/x6_summary.json. Usage: python x6_section.py <json>"""
import json
import sys

S = json.load(open(sys.argv[1]))
SHORT = {"mw-button-press-topdown": "mw-button-press", "mw-peg-insert-side": "mw-peg-insert"}
RN = {"shiter": "sh-iter", "detshift": "det+shift", "det": "det", "detbc": "det+BC"}


def pp(x):
    return f"{100 * x:+.1f}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


L = ["## 12. [사후] X6: 정책 기준선(D7)", ""]
m = S["meta"]
L.append("- 설계: DESIGN_tmlr.md §11.1 D7(자료 전 고정, commit 24a59a00; 사용자 승인 \"돌려줘\"). 기술 보고이며 판정 문턱이 없다.")
L.append(f"- 실행: pod td.8(A40 SECURE). E11의 12과제 × 90짝(체크포인트 1~3 × seed 100000~100029) × {{none(기본 MPPI, FP32), "
         f"policy(TD-MPC2의 `mpc=False` 경로: 정책 π의 평균 행동, eval_mode)}}. episode {S['n_episodes']:,}개, job 결과 {m['files']}개, "
         f"모두 {', '.join(m['gpus'])}, (과제, 체크포인트)마다 host 하나({m['pairing_one_host_per_task_ckpt']}), seed 불일치 {m['seed_mismatch']}.")
ids = [r["none_vs_e11"] for r in S["tasks"].values()]
L.append(f"- 점검: td.8의 none이 E11의 none과 {sum(x['identical'] for x in ids)}/{sum(x['n'] for x in ids)} episode에서 비트 단위로 같다"
         " (return·성공). policy 경로는 dynamics·reward 층(dyn0·dyn1·rew0·rew1)의 호출 수가 모든 과제에서 0이다(`check_policy_calls.json`).")
L.append("- 비교 규칙(자료 전에 정함): E12에서 dither 구성(PT·PC × sh-iter·det+shift) 가운데 k\\*가 가장 낮은 것(같으면 k\\*에서의 D가 작은 것, 그다음 PT·PC, sh-iter·det+shift 순)의 D와 CI를 쓴다. 두 CI가 겹치지 않으면 \"낫다\"(dither CI 상한 < policy CI 하한) 또는 \"못하다\", 겹치면 \"같다\"로 센다. E12 값은 다른 pod(같은 A40, 같은 seed)의 결과라 짝이 아니다.")
L.append("- 출처: `results/analysis_x6/x6_summary.json`, `x6_cells.csv`, E12 입력 `results/analysis/td_e12_cells.csv`.")
L.append("")
L.append("| 과제 | none | policy | D_policy [CI] | E12 가장 낮은 dither 구성(k\\*) | D [CI] (짝) | dither 대 policy | 참고: 모든 반올림 중 가장 낮은 구성 |")
L.append("|---|---|---|---|---|---|---|---|")
for t, r in S["tasks"].items():
    mw = r["unit"] == "mw"
    d, b = r.get("e12_lowest_dither") or {}, r.get("e12_lowest_any") or {}
    nm = (lambda x: f"{x:.3f}") if mw else (lambda x: f"{x:.1f}")
    word = {"better": "낫다", "same": "같다", "worse": "못하다"}.get(r.get("dither_vs_policy"), "")
    L.append(f"| {SHORT.get(t, t)} | {nm(r['none_mean'])} | {nm(r['policy_mean'])} | {pp(r['D_policy'])} {ci(r['D_policy_ci'])} | "
             f"{d.get('range')} {RN.get(d.get('rounding'), d.get('rounding'))} k = {d.get('k')} | {pp(d['D'])} {ci(d['D_ci'])} ({d.get('n')}) | "
             f"{word} | {b.get('range')} {RN.get(b.get('rounding'), b.get('rounding'))} k = {b.get('k')}: {pp(b['D'])} |")
L.append("")
c = S["counts_dither_vs_policy"]
L.append(f"- 과제 수(dither 대 policy): MW 낫다 {c['mw']['better']} · 같다 {c['mw']['same']} · 못하다 {c['mw']['worse']}, "
         f"DMC 낫다 {c['dmc']['better']} · 같다 {c['dmc']['same']} · 못하다 {c['dmc']['worse']}.")
L.append("- D는 MW 성공률 %p, DMC 상대 return %다(none 대비). none은 td.8의 none(E11과 같음)이다.")
L.append("")
print("\n".join(L))
