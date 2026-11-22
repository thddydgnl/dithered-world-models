"""Formats REPORT.md section 11 (X5, D6) from results/analysis_x5/x5_summary.json (pure Python, no computation beyond
reading and counting). Usage: python x5_section.py <x5_summary.json> > section.md"""
import json
import sys

S = json.load(open(sys.argv[1]))
K1 = {"mw-push": 3, "mw-pick-place": 3, "mw-door-open": 4, "mw-drawer-open": 3, "mw-button-press-topdown": 3,
      "mw-peg-insert-side": 4, "mw-window-open": 4, "mw-sweep-into": 3}
SHORT = {"mw-button-press-topdown": "mw-button-press", "mw-peg-insert-side": "mw-peg-insert"}


def pp(x):
    return f"{100 * x:+.1f}"


def ci(c):
    return f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"


L = []
L.append("## 11. [사후] X5: MW det+noise 비트 곡선(D6)")
L.append("")
L.append("- 설계: DESIGN_tmlr.md §11.1 D6(자료 전 고정, commit 1d73e10; 사용자 승인 \"너의 추천대로 해줘\"). 기술 보고이며 판정 문턱이 없다.")
m = S["meta"]
L.append(f"- 실행: pod td.7(A40 SECURE, `l9usfc4n7b7evv`). MW 8과제 × det+noise × {{PT k1~5, PC k1~3}} × 90짝과 none 90짝, "
         f"E12의 위쪽 확장 규칙(가장 높은 k에서 D > τ = 0.10이면 k = 8까지). episode {S['n_episodes']:,}개, job 결과 {m['files']}개, "
         f"모두 {', '.join(m['gpus'])}, (과제, 체크포인트)마다 host 하나({m['pairing_one_host_per_task_ckpt']}), seed 불일치 {m['seed_mismatch']}. "
         f"적응 결정 {len(m['decisions'])}개(`raw/td.7/driver/decisions.jsonl`).")
L.append("- 범위: PT는 E0 파일, PC는 td.5·td.6에서 다시 잰 파일 24개를 그대로 복사(md5 24/24 일치). 코드는 td 본 실행 판(qlib.py md5 `fd1ceb9c…`, det+noise 구현 포함), 대기열만 `plan_x5.py`.")
L.append("- 점검: 처음 끝난 none 8 job이 E11(td.5)의 같은 job과 비트 단위로 같았다(아래 재현 표에 전체).")
L.append("- 출처: `results/analysis_x5/x5_summary.json`, `x5_kstar.csv`, `x5_curve.csv`, `x5_tables.md`. E12의 k\\*는 `results/analysis/td_e12_cells.csv`다. E12 값은 다른 pod(같은 A40)의 결과라 짝이 아니므로 k\\* 비교는 기술이다.")
L.append("")
L.append("### 11.1 k\\*(det+noise)와 E12 k\\*")
L.append("")
L.append("| 과제 | 범위 | k\\*(det+noise) | E12 k\\*: det / det+BC / sh-iter / det+shift | G_noise | noise ≤ sh-iter | noise ≤ min(sh-iter, det+shift) |")
L.append("|---|---|---|---|---|---|---|")
for key, x in S["kstar"].items():
    t, rk = key.split("|")
    yn = lambda b: "예" if b else "아니오"
    L.append(f"| {SHORT.get(t, t)} | {rk} | {x.get('kstar_detnoise')} | {x.get('kstar_det')} / {x.get('kstar_detbc')} / "
             f"{x.get('kstar_shiter')} / {x.get('kstar_detshift')} | {x.get('G_noise')} | {yn(x.get('noise_le_shiter'))} | "
             f"{yn(x.get('noise_le_min_shiter_detshift'))} |")
L.append("")
L.append("| 범위 | 과제 | k\\*(noise) ≤ k\\*(sh-iter) | k\\*(noise) ≤ min(k\\*(sh-iter), k\\*(det+shift)) | G_noise ≥ 1 |")
L.append("|---|---|---|---|---|")
for rk, c in S["counts"].items():
    L.append(f"| {rk} | {c['n_tasks']} | {c['noise_le_shiter']} | {c['noise_le_min_shiter_detshift']} | {c['G_noise_ge_1']} |")
L.append("")
L.append("- k\\* = 9는 k = 8까지도 D > τ라는 뜻이다(E12와 같은 정의).")
L.append("")
L.append("### 11.2 det+noise 비트–하락 곡선 (D [95% CI], 90짝)")
L.append("")
L.append("| 과제 | PT k: D [CI] | PC k: D [CI] |")
L.append("|---|---|---|")
for t in K1:
    cells = []
    for rk in ("PT", "PC"):
        cur = S["curves"].get(f"{t}|{rk}", {})
        cells.append("; ".join(f"{k}: {pp(v['D'])} {ci(v['D_ci'])}" for k, v in sorted(cur.items(), key=lambda kv: int(kv[0]))))
    L.append(f"| {SHORT.get(t, t)} | {cells[0]} | {cells[1]} |")
L.append("")
L.append("### 11.3 E11 재현 점검 (PT 1차 k의 det+noise와 none, 같은 seed, 같은 기종의 다른 pod)")
L.append("")
L.append("| 과제 | k | D_det+noise X5 [CI] | E11 [CI] | 같은 episode: none | det+noise |")
L.append("|---|---|---|---|---|---|")
for t, r in S["repro_e11"].items():
    x, e = r.get("x5"), r.get("e11")
    b = r["bit_identical_episodes"]
    L.append(f"| {SHORT.get(t, t)} | {r['k']} | {pp(x['D'])} {ci(x['D_ci'])} | {pp(e['D'])} {ci(e['D_ci'])} | "
             f"{b['none']['identical']}/{b['none']['n']} | {b['detnoise']['identical']}/{b['detnoise']['n']} |")
L.append("")
print("\n".join(L))
