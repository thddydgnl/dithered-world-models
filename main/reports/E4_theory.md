# E4 보고: 이론과 수치 모의 (논문 A 부록용)

- 작업자 E4, 2026-10-22. 기준: EXPERIMENT_DESIGN.md 4.1~4.4, 6.2~6.4, DESIGN_main.md v1~v4(특히 0.4, 4.2, 5, 6).
- 조정자 추가 지시(06:58Z) 두 건을 함께 다룬다: OEI 분해(2.9절), 비편향 dither의 비선형 2차 편향(2.8절).
- E1 결과 파일은 읽지 않았다. 실험 쪽 숫자는 파일럿 근거(5장)와 분석 **코드**의 출력 형식만 썼다.
- 코드 `main/code/theory/`, 결과 `main/results/E4/`, 작업 로그 `main/worklogs/E4_theory.md`(사전 선언 격자 07:24Z 포함).

## 요약

**핵심 유도**(2장, 수치 점검 포함)
1. **OEI 분해**: OEI = sel_last + drift(항등식).
   - sel_last는 마지막 평가 안의 선택이고, 같은 평가값으로 재면 winner's curse다.
   - $\text{sel\_last} = -\frac{\sigma_{e,T}^2+\rho\sigma_J\sigma_e}{\sigma_{\hat J,T}}S_{op}(N,k,\theta)$(2변량 정규에서 정확).
   - 분포 무관형은 Stein 항등식 $-\sigma_\perp^2\cdot$(선택 경계의 near-tie 밀도)다. 공통 오차 성분은 정확히 빠진다(CRN).
2. **새 잡음**(ind, sh-iter, det+shift)
   - drift의 기대값은 정확히 0이고, 실행 계획을 독립 draw로 다시 잰 sel_last도 0이다.
   - 같은 평가값의 sel_last는 0이 아니다. 그래서 **TD-MPC2 OEI(같은 평가값)는 새 잡음 변형에서도 음수로 예측**되고, DINO-WM `oei_ref_mf`(재평가)는 0으로 예측된다.
3. **지속장**(det, 고정 표)
   - 계획기는 모집단 std로 평활한 모델 비용의 최소점(SAA)으로 간다.
   - 약한 장에서 drift ≈ −σ_b² tr H⁻¹/ℓ² · (1+2σ_T²/ℓ²)^{−d/2−1} = −2 × regret이다. N에 1차로 무관하고, 반복과 함께 커지다 포화한다.
   - 장이 모집단 척도에서 거칠면 반복 사이에서 새 잡음처럼 보인다. 그래서 MPPI(σ_min 바닥)와 CEM(계속 붕괴)이 같은 det 오차를 다르게 이용한다.
4. **CRN**: 공유 offset 반올림 오차의 공분산은 브라운 다리 커널 min(f, f′) − ff′다.
   - 후보 간 상관은 ρ_cand(σ_y) = ½ + 3Σ e^{−4π²m²σ_y²}/(π²m²)이고, CRN 이득은 (1/6)/V(σ_y) ≥ 2다. near-tie에서 ≈ √π/(6σ_y)로 커진다(H4b 방향 지지).
5. **고정 표 크기**: 후보 간 분산의 정확식(2.6.1). 1/N_eff 법칙은 오프셋을 공유하는 항목이 같이 움직이는 일관 축에서만 성립하고, 비일관 축에서는 평탄하다.
6. **det+shift** = SR 오차 − (u − ½)(감산형 dither)다. (3)에 속하고, sh-iter보다 오프셋 편향 항과 Jensen 분산 변조가 없다. 예측: 손상 ≤ sh-iter, drift 0.
7. **Jensen 2차 편향**(조정자 2번)
   - $b_2 = \tfrac12\sum\Delta^2 H_{ii} v_i$이고, SR은 $v = f(1-f)$, det+shift는 1/12다. 비트에 대해 ∝ 4^{−k}이다.
   - 새 dither에서도 지속장처럼 이용된다. det drift 대비 비는 ≈ Δ²(H/g)²/60이다.

**모의 일치**(5장; 504 job, 계획기 실행 922,368회)
- 단발 선택(C1): 정규 (J, e)에서 명제 1은 상대 오차 0.03~0.14%(350칸)다. 실제 2차 첫 모집단에서는 2변량 식이 well-separated일 때 최대 6.4배 과소예측하고, near-tie 수 근사(상대 오차 0.85%)와 고정 J 정확식(1.2%)이 이를 고친다.
- 계획기 안의 선택 항(C2, 같은 평가값): CEM elite 평균 기울기 1.02, R² 0.998(5,247칸), CEM argmin(near-tie 수) 1.05, 0.995, MPPI(top-k + softmax + Gumbel) 0.97, 0.977(6,976칸). 지속장의 MPPI 마지막 반복만 약 20% 과대예측(공간 상관)이다.
- 분해(C3): 새 잡음에서 drift와 재평가 sel_last의 95% CI가 0을 포함하는 칸이 94.8~95.8%(명목 95%)다. 같은 평가 sel_last는 94~99.6% 칸에서 음수다. 지속 오차의 drift는 CEM 83%, MPPI 64% 칸에서 음수다.
- SAA(C4): 점진 평활 예측이 지속장의 drift를 맞힌다(기울기 0.92~1.02, R² 0.986~0.996). 수렴 위치는 못 맞힌다(강한 장, 국소 최소점이 여럿).
- CRN(C8): sel_last가 $(1-\rho_c)$에 비례한다(0.53, 0.105, 0.014 대 0.5, 0.1, 0.01). 고정 표 직접 측정의 ind/sh-iter 분산비는 near-tie에서 34(식 33)다.
- 고정 표(C9): 정확식 (2.6.1)이 측정과 맞는다(비 1.03, R² 0.985). 일관 toy에서 손상이 N_eff에 단조 감소(Spearman −1, 8~260배)하고, 비일관 toy에서는 평탄하다.
- 반복 수(C6): MPPI에서 새 잡음 손상은 줄고(n 6 → 12 비 0.75~0.91) 지속은 평탄(0.97~1.05)하다. 폐루프에서 (sh-step형 − sh-iter형) 손상 차는 n = 1에서 0(−0.01 ± 0.02)이고 n = 12에서 +0.22~0.39로 R2-14의 모양이다. CEM은 붕괴 때문에 새 잡음도 평균하지 못한다.
- det+shift(C10): 같은 비트에서 sh-iter보다 유의하게 나쁜 칸이 없다(0/32, 손상 비 중앙값 0.83~0.97). 오차 분산을 맞춘 비교에서는 1.1~1.2배다.
- Jensen(C11): $b_2 \propto 4^{-k}$(측정 기울기 −2.07 ~ −2.12/bit), det+shift는 상수장이다. toy에서는 drift가 검출 한계 아래이고, 파일럿 R2-13 수치로는 det drift의 1~3%다.
- 어긋남(6장): 지속 지수가 지속/새 오차를 가르지 못한다. 새 잡음 정상 손상식이 최대 약 3배 어긋난다.

**결정 사항**(7장. 정의와 문턱은 바꾸지 않았다)
- 7.1 H2a: TD-MPC2 OEI 정의상 sh-iter OEI는 이론적으로 음수다. 모의에서 새 잡음 OEI는 기본 설정 −6.25 σ_J0로 지속장 −9.03과 같은 급이다. 파일럿 기반 예측으로 TD-MPC2 세 과제 중 둘에서 sh-iter 항이 det 항과 같은 급이다. 선택지(등록대로 판정과 해설 / 분해 사후 보고 / E2에서 재평가 OEI 추가)는 7.1절이다.
- 7.2 H2c: 고정 elite 비율의 elite 평균 선택에서는 N 용량-반응이 약하다. 고정 elite 수에서는 |OEI|가 1.7~5.4배 오르지만, 고정 비율에서는 0.99~1.40배이고 손상은 오히려 준다. v5에서 elite 규칙을 정해야 한다.
- 7.3 H4a: 1/N_eff 법칙과 단조성은 공유 축의 일관성에 달려 있다. 일관 toy는 단조(Spearman −1), 비일관 toy는 평탄이다. 손상의 직접 예측 변수는 각 수준의 고정 표 후보 간 분산이다. E3에서 `var_cand`를 함께 잴지 결정해야 한다.


---

## 1. 무엇을 했나

1. **유도**(2장): 오차 모형 (1) 지속장, (2) 새 독립 잡음, (3) 새 공유 잡음, (4) 고정 공유 dither와 det+shift, 그리고 비선형 2차(Jensen) 편향. 계획기는 argmin, top-k 평균(CEM), top-k + softmax(MPPI), Gumbel 선택을 다룬다.
2. **모의**(4~5장): 서버에서 읽은 두 계획기 코드를 그대로 옮겼다.
   - TD-MPC2 MPPI: 후보 512(policy prior 24), elite 64, 반복 6, temperature 0.5, std [0.05, 2], [−1, 1] clamp, warm start, 마지막 Gumbel elite 선택
   - DINO-WM CEM: 후보 300, top-30, 반복 30, 첫 표본 = 평균, 불편 std, open loop
   - 2차 목적함수(차원 2~12, near-tie·well-separated) 위에서 사전 선언 격자 504 job, 계획기 실행 922,368회를 돌렸다.
3. **겹쳐 그리기 도구**(9장): E2·E3 분석 JSON을 이론·모의와 같은 축에 그린다. 합성 자료로만 시험했다.

---

## 2. 유도

### 2.1 설정과 기호

- 한 번의 평가(계획기 반복 t)에서 후보 $a_{t,1},\dots,a_{t,N}$가 모집단 $p_t$에서 나온다.
  - 참 비용 $J_i = J(a_{t,i})$, 비용 오차 $e_i$, 모델 비용 $\hat J_i = J_i + e_i$.
  - MPPI의 가치는 $V = -\hat J$이다. TD-MPC2에서는 $e = V_0 - V_q$, DINO-WM에서는 $e = J_v - J_{none}$으로 둘 다 "비용 오차"다.
- 계획기의 선택은 가중치 $w_i = W_i(\hat J_1,\dots,\hat J_N)$, $\sum_i w_i = 1$이다. 모델 비용에만 의존하고 이동 불변이다: $W(\hat J + c\mathbf 1) = W(\hat J)$.
  - argmin: 최솟값에 1
  - top-k 평균(CEM): elite k개에 $1/k$
  - MPPI: top-k elite에 $\propto \exp(-\lambda \hat J_i)$, λ = temperature
  - TD-MPC2의 실행 후보: 마지막 반복의 MPPI 가중치로 elite 하나를 Categorical 추출(gumbel_softmax_sample)한다. 기대값은 가중 평균과 같다.
- 갱신: $\mu_{t+1} = \sum_i w_i a_{t,i}$. std는 elite의 (가중) std이고, MPPI는 $[\sigma_{min}, \sigma_{max}] = [0.05, 2]$로 자른다.
- 모집단 기술자(같은 평가 안, 공통 성분 제거)
  - $\sigma_{J,t}$, $\sigma_{e,t}$ = 후보 간 std
  - $\rho_t$ = corr(J, e)
  - $\sigma_{\hat J,t}^2 = \sigma_{J,t}^2 + \sigma_{e,t}^2 + 2\rho_t\sigma_{J,t}\sigma_{e,t}$
  - $\theta_t = \lambda \sigma_{\hat J,t}$
  - $k_{eff} = 1/\sum w^2$

**OEI(실험 정의 그대로)**
- TD-MPC2(DESIGN_main 0.4): $\mathrm{OEI}^{MPPI} = e_T(a_{exec}) - \bar e_1^{samp}$
  - $e_T(a_{exec})$: 마지막 반복 T에서 실행 elite가 **선택에 쓰인 바로 그 평가**에서 가진 오차
  - $\bar e_1^{samp}$: 첫 반복 표본 후보(policy prior 제외)의 평균 오차
- DINO-WM(`oei_ref_mf`): $\mathrm{OEI}^{CEM} = \tilde e(\mu_T) - \bar e_1$
  - $\mu_T$는 최종 elite 평균이다.
  - $\tilde e$는 det·고정 표 변형이면 그 오차이고, 새 잡음 변형(ind, sh-iter, det+shift)이면 **독립 draw 한 번으로 다시 잰 오차**다(runner의 `Q.it = opt_steps`).

**분해(조정자 1번, 항등식)**

$$\mathrm{OEI} = \underbrace{e(\text{실행}) - \bar e_T}_{\text{sel\_last}} + \underbrace{\bar e_T - \bar e_1}_{\text{drift}}$$

- sel_last: 마지막 평가 안의 선택
- drift: 반복을 거치며 모집단이 유리한 오차 쪽으로 옮겨 간 양
- 모의는 sel_last를 두 가지로 기록한다. (a) 같은 평가값, (b) 실행 계획을 독립 draw로 다시 평가한 값.

### 2.2 한 번의 평가 안의 선택: winner's curse

**명제 1 (2변량 정규, 정확).** $(J_i, e_i)$가 후보마다 i.i.d. 결합 정규 $(\sigma_J, \sigma_e, \rho)$이면, 모델 비용에만 의존하는 모든 선택 규칙에 대해

$$\mathbb E\Big[\sum_i w_i e_i\Big] - \mathbb E[\bar e] = -\frac{\sigma_e^2 + \rho\sigma_J\sigma_e}{\sigma_{\hat J}}\, S_{op}, \qquad S_{op} = \mathbb E\Big[\sum_i w_i Z_i\Big],\ Z_i = \frac{\mu_{\hat J} - \hat J_i}{\sigma_{\hat J}}.$$

- 증명: $e_i = \mu_e + \beta(\hat J_i - \mu_{\hat J}) + \eta_i$, $\beta = \mathrm{Cov}(e,\hat J)/\sigma_{\hat J}^2$로 쓴다. $\eta$는 $\hat J$와, 따라서 $w$와 독립이다.
- 참 비용 쪽의 선택은 $-(\sigma_J^2 + \rho\sigma_J\sigma_e)S_{op}/\sigma_{\hat J}$로 감쇠한다.
- 선택 강도 $S_{op}$(정확 적분 또는 MC 20,000회, `s_tables.json`)

| 연산자 | $S_{op}$ | N = 64 | 512 | 1024 | N 의존 |
|---|---|---|---|---|---|
| argmin | $e_N = \mathbb E\max_{i\le N} Z_i \approx \sqrt{2\ln N} - \frac{\ln\ln N + \ln 4\pi}{2\sqrt{2\ln N}}$ | 2.344 | 3.044 | 3.248 | 64 → 1024에서 +39% |
| top-k 평균, q = k/N 고정 | $i_{k,N} = \frac1k\sum_{r\le k}\mathbb E Z_{(r:N)} \to \varphi(z_q)/q$ | 1.614 (q 12.5%) | 1.643 | 1.645 | +2%, 극한 1.647 |
| top-k 평균, k 고정 | $i_{k,N}$, q = k/N가 N과 함께 감소 | k = 64면 0(전부 elite) | 1.643 | 2.0 | 크다 |
| MPPI(top-k + softmax θ) | $i_{k,N}$(θ → 0)에서 $e_N$(θ → ∞)까지 | | 1.643(θ 0), 1.88(θ 1), 3.02(θ 10) | | θ에 달림 |

**명제 2 (Stein 형태, J 분포 무관).** e가 후보 간 공분산 $\Sigma_e$를 갖는 정규이고 J와 독립이면

$$\mathbb E\Big[\sum_i w_i e_i\Big] - \mathbb E[\bar e] = \sum_{i,j}(\Sigma_e)_{ij}\,\mathbb E\Big[\frac{\partial w_i}{\partial \hat J_j}\Big].$$

- 이동 불변성 때문에 $\sum_j \partial w_i/\partial\hat J_j = 0$이다. 그래서 **모든 후보에 공통인 오차 성분은 정확히 사라진다**(CRN).
- 후보 간 독립 성분(분산 $\sigma_\perp^2$)만 남으면 $-\sigma_\perp^2 D_{op}$이다. $D_{op}$는 "선택 경계에서 후보의 밀도"다.
  - top-k 평균: $D = f_{\hat J}(t_k)/q$($t_k$ = k번째 모델 비용)
  - MPPI: $D = \lambda(1 - \sum w^2) + N f_{\hat J}(t_k) w_{(k)}$
  - argmin: $D = N(N-1)\int f_{\hat J}^2 (1 - F_{\hat J})^{N-2}$
- 해석: winner's curse = (이용 가능한 오차 분산) × (선택 경계의 near-tie 밀도). "계획은 near-tie에서 이루어진다"(설계 1.2절)를 정량으로 옮긴 식이다.

**명제 3 (argmin의 near-tie 수 근사).** $\Delta_{argmin} \approx -\sigma_\perp\, e_{n_c}$, $n_c = \sum_j 2\Phi\big(-(J_j - J_{min})/(\sqrt2\sigma_\perp)\big)$.
- 오차 크기 안에 들어오는 경쟁 후보 수 $n_c$가 극값 통계의 N을 대신한다.
- $\sigma_\perp \gg$ 퍼짐이면 $n_c \to N$, gap ≫ $\sigma_\perp$이면 $n_c \to 1$이 되어 선택 편향이 0이다.
- 2차 목적함수의 첫 모집단처럼 아래가 막힌 비대칭 분포에서는 명제 1의 전역 $\sigma_J$가 near-tie 밀도를 과소평가한다. 이 근사가 그 오차를 고친다(5.1절).

**척도.** sel_last의 N·elite·반복 의존은 $S_{op}$(또는 $D_{op}$)와 마지막 모집단의 $(\sigma_{e,T}, \sigma_{J,T})$로 정해진다.
- N: argmin형은 $\sqrt{2\ln N}$로 커진다. 고정 elite 비율의 elite 평균은 거의 평탄하다(+2%). 고정 elite 수는 커진다.
- elite 수 k: k가 클수록 $S_{op}$가 작다.
- 반복 수 T
  - 모집단이 수렴하면 $\sigma_{J,T}$가 작아진다($\propto \sigma_T^2$). 그러면 $\kappa = \sigma_e/\sigma_{\hat J} \to 1$이고 $|\text{sel\_last}| \to \sigma_{e,T} S_{op}$로 포화한다.
  - $\sigma_{e,T}$는 백색 잡음이면 그대로다. 매끄러운 오차장(길이 ℓ)이면 $\sigma_T/\ell$에 비례해 준다.
- 파일럿 R2-8b 결정 수준 자료로 본 TD-MPC2 마지막 반복: $\sigma_{e}/\sigma_{J}$가 det 7~477, 확률적 변형 16~196(중앙값)이었다. 모든 양자화 변형이 깊은 near-tie에 있었다.

### 2.3 모형 (1) 지속 오차장 $e(a) = b(a)$ (det)

**(a) 고정점 = SAA.** 가우시안 표본 계획기의 elite 갱신은 1차에서 다음과 같다(명제 1을 $a = \mu + \sigma z$에 적용).

$$\mu_{t+1} - \mu_t \approx -\frac{S_{op}\,\sigma_t^2}{\sigma_{\hat J,t}}\nabla \hat J_{\sigma_t}(\mu_t), \qquad \hat J_\sigma(\mu) = \mathbb E_z[\hat J(\mu + \sigma z)] = J_\sigma + b_\sigma.$$

- 그래서 반복은 $\sigma_t$로 평활한 모델 비용의 최소점을 따라간다(점진 평활, graduated smoothing).
- CEM은 $\sigma_T \to 0$이라 $J + b$의 국소 최소점으로, MPPI는 $\sigma_T \ge \sigma_{min}$이라 $J + b_{\sigma_{min}}$의 최소점으로 수렴한다.
- 이것이 SAA다. 계획기는 교란된 문제를 정확히 푼다.

**(b) 약한 장(이동 ≪ ℓ), 2차 J.** $\delta = \mu^* - a^* \approx -H^{-1}\nabla b_{\sigma_T}(a^*)$이고 regret은 $\tfrac12\nabla b_\sigma^\top H^{-1}\nabla b_\sigma$이다.
- b가 분산 $\sigma_b^2$, 길이 ℓ의 정상 가우시안 장이면 $\mathbb E[\nabla b_\sigma\nabla b_\sigma^\top] = (\sigma_b^2/\ell^2)(1 + 2\sigma^2/\ell^2)^{-d/2-1}I$다. 그래서 다음이 나온다.

$$\mathbb E[\text{regret}] = \frac{\sigma_b^2\,\mathrm{tr}H^{-1}}{2\ell^2}\Big(1 + \frac{2\sigma_T^2}{\ell^2}\Big)^{-d/2-1}, \qquad \mathbb E[\text{drift}] \approx -\frac{\sigma_b^2\,\mathrm{tr}H^{-1}}{\ell^2}\Big(1 + \frac{2\sigma_T^2}{\ell^2}\Big)^{-d/2-1} = -2\,\mathbb E[\text{regret}].$$

- 최종 평균 한 점의 오차(CEM `oei_ref_mf`)는 평활 인자가 하나라 $\mathbb E[\mathrm{OEI}_{mf}] \approx -\frac{\sigma_b^2\mathrm{tr}H^{-1}}{\ell^2}(1 + \sigma_T^2/\ell^2)^{-d/2-1}$이다(`rff_saa_moments`).

- drift = $\mathbb E_{p_T}[b] - \mathbb E_{p_1}[b] \approx b_{\sigma_T}(\mu^*) - b_{\sigma_1}(\mu_0)$이다. 인스턴스 평균에서 $\mathbb E\, b_{\sigma_1}(\mu_0) = 0$이다.
- sel_last(같은 평가 = 재평가)는 명제 1을 마지막 모집단에 적용한 것이다. 매끄러운 장이면 $\sigma_{b,T}\approx\sigma_T\|\nabla b\|$이고 $\rho_T \ne 0$이다.
- CEM `oei_ref_mf`는 수렴($\sigma_T \to 0$)하면 sel_last_mf → 0이다. 그래서 $\mathrm{OEI}^{CEM} \approx$ drift다.

**(c) 강한 장(이동 ≳ ℓ).** 계획기는 J가 장의 크기만큼만 변하는 영역 안의 국소 최소점 하나에 머문다.
- drift는 그 깊이 $\approx -\sigma_b\sqrt{2\ln n_{loc}}$이다(극값 형태).
- $n_{loc}$는 평활 일정이 훑은 상관 칸의 수, 대략 $(R/\ell)^d$이고 $R = \sqrt{2\sigma_b/\bar\lambda}$다.
- 정량 예측은 모의에서 실제 $\sigma_t$ 궤적으로 점진 평활을 따라간 수치해를 쓴다(`saa_track`).

**(d) 거친 장과 매끄러운 장.** 기준은 ℓ과 그 반복 모집단 퍼짐 $\sigma_t$의 비다.
- $\ell \ll \sigma_t$이면 $b_\sigma$의 인자 $(1 + 2\sigma^2/\ell^2)^{-d/2-1}$이 장을 지운다. 그래서 drift가 없고 반복 안의 선택(sel_last)만 남는다. 반복 사이에서는 새 잡음처럼 보인다.
- $\sigma_t$가 ℓ 아래로 내려가면(CEM의 붕괴) SAA 이용이 시작된다.
- det 반올림장의 ℓ은 행동 공간에서 활성이 격자 한 칸을 지나는 거리 $\approx \Delta/\|\nabla_a x\|$다. 그래서 det를 얼마나 이용하는지는 계획기의 모집단 일정이 정한다. MPPI는 $\sigma_{min}$에서 멈추고, CEM은 계속 붕괴한다.

**(e) 척도(H2c 관련)**
- N: sel_last는 $S_{op}(N)$을 따른다. drift는 고정 elite 비율에서 1차로 N과 무관하다. 고정점이 N에 달려 있지 않고, 유한 N은 잡음만 더한다.
- 반복 수: drift는 평활이 진행되는 동안 커지다가 SAA 점에서 포화한다. 지속 오차는 반복으로 평균되지 않는다.
- elite 비율: 붕괴 속도를 통해 약하게 작용한다.

### 2.4 모형 (2) 반복마다 새 독립 잡음 (ind)

**(a) drift는 정확히 0.**
- $e_t(a) = m(a) + \varepsilon_t(a)$이고 $\varepsilon_t$가 반복 t 전의 모든 것과 독립이면 $\mathbb E[\text{drift}] = \mathbb E[\bar m_{p_T} - \bar m_{p_1}]$이다.
- 이유: 마지막 반복 후보의 위치는 계획기 표본과 $\varepsilon_1..\varepsilon_{T-1}$의 함수라 $\varepsilon_T$와 독립이다.
- 1차 비편향 잡음($m$ = 상수)이면 N, k, T, 계획기에 관계없이 $\mathbb E[\text{drift}] = 0$이다. $m(a)$가 상수가 아닌 경우는 2.8절(Jensen)이다.

**(b) sel_last**
- **같은 평가값**: 실행 elite는 $\hat J_T = J + \varepsilon_T$로 골랐다. 그래서 $-\sigma_\varepsilon^2 S_{op}/\sigma_{\hat J,T} \ne 0$이다. 늦은 반복의 near-tie에서는 $\approx -\sigma_\varepsilon S_{op}$다.
- **독립 재평가**: $\mathbb E = 0$이다.
- 따라서 실험 정의로는 다음과 같다.
  - TD-MPC2: $\mathbb E[\mathrm{OEI}_{ind}] = \mathbb E[\text{sel\_last}] < 0$
  - DINO-WM: $\mathbb E[\mathrm{OEI}_{ind}] = 0$

**(c) 계획에 주는 영향(SA)**
- 고정점 편향이 없다($\mu = a^*$에서 기대 갱신이 0).
- 효과는 두 가지다.
  - 하강 이득 감쇠: $\mu_{t+1} - \mu_t \approx -(S\sigma_t^2/\sigma_{\hat J,t})\nabla J$. 잡음이 없을 때의 $\sigma_{J,t}$ 대신 $\sigma_{\hat J,t}$로 나눈다.
  - 확산: $\mathrm{Var}(\mu_{t+1}\mid\mu_t) \approx \sigma_t^2/k_{eff}$
- AR(1): $\delta_{t+1} = (I - \alpha_t H)\delta_t + \xi_t$, $\alpha_t = \sigma_t^2 S_t/\sigma_{\hat J,t}$
- MPPI처럼 $\sigma_t = \sigma_{min}$에 머물면 정상 상태는 다음과 같다.

$$\mathrm{Var}\,\delta_\infty = \frac{\sigma_{\hat J}}{2 S k_{eff}}H^{-1}, \qquad \mathbb E[\text{regret}(\mu)] = \frac{d\,\sigma_{\hat J}}{4 S k_{eff}} \approx \frac{d\,\sigma_\varepsilon}{4 S k_{eff}}, \qquad \tau = \frac{\sigma_\varepsilon}{\sigma_{min}^2 S \lambda_{min}}.$$

- 조건
  - 반복 간 독립
  - $m$ = 상수
  - 모집단 std가 아래로 유계($\sigma_{min} > 0$)이거나 천천히 준다. 빨리 붕괴하면 평균이 그 전에 얼어붙는다.
  - 1차 선형 응답(σ가 곡률 척도보다 작음)
  - $k_{eff} \ge 2$
- **설계 문구 "1/(elite 수 × 반복 수)"의 정정**
  - momentum 없는 MPPI·CEM은 반복 평균(감소 보폭 SA)을 하지 않는다. 그래서 잔여 손상은 반복 수에 1/T로 줄지 않는다.
  - $T \ll \tau$에서는 감쇠된 하강이 따라잡으며 줄어든다. $T \gg \tau$에서는 $\propto \sigma_\varepsilon/(S k_{eff})$에서 포화한다.
  - 반복이 만드는 차이는 지속 오차와의 대비에 있다. 반복마다 같은 크기 p로 미는 지속 오차는 $p/\alpha$까지 쌓인다(SAA 편향). 새 잡음은 rms $p/\sqrt{2\alpha}$까지만 쌓인다.

### 2.5 모형 (3) 반복마다 새 공유 잡음 (sh-iter, CRN)

- 명제 2에 따라 공통 성분 $c_t\mathbf 1$은 모든 선택에서 사라진다. 그래서 다음과 같다.
  - sel_last(같은 평가) = $-\sigma_{\perp,T}^2 S_{op}/\sigma_{\hat J,T}$
  - 재평가 sel_last = 0
  - drift: 기대값 0
- 공통 성분은 OEI에 편향이 아니라 분산($c_T - c_1$)을 더한다.

**반올림 모형에서의 CRN 크기**
- 오프셋 u를 공유하는 확률적 반올림 오차(격자 단위)는 $r(y;u) = \lfloor y+u\rfloor - y = (u - \tfrac12) + \mathrm{saw}(y+u)$, $\mathrm{saw}(z) = \sum_{m\ge1}\sin(2\pi m z)/(\pi m)$다.
- 같은 u를 쓰는 두 값의 공분산은 **브라운 다리 커널**이다.

$$\mathbb E_u[r(y;u)\,r(y';u)] = \min(f, f') - f f', \qquad f = y \bmod 1.$$

- 모집단의 활성 퍼짐이 $\sigma_y$(격자 단위, 위상 균등)일 때 한 항목 오차의 후보 간 상관은 다음과 같다.

$$\rho_{cand}(\sigma_y) = \tfrac12 + 3\sum_{m\ge1}\frac{e^{-4\pi^2m^2\sigma_y^2}}{\pi^2m^2}, \qquad \rho_{cand}(0) = 1,\ \rho_{cand}(\infty) = \tfrac12.$$

- 이용 가능한 분산은 다음과 같다.
  - sh-iter: $V(\sigma_y) = \tfrac{1-\rho}{6} = \tfrac1{12} - \sum_m \tfrac{e^{-4\pi^2m^2\sigma_y^2}}{2\pi^2m^2} \approx \sigma_y/\sqrt\pi$($\sigma_y \ll 1$)
  - ind: $\mathbb E f(1-f) = 1/6$
- **CRN 이득** $G_{CRN} = (1/6)/V(\sigma_y) \ge 2$이고, near-tie($\sigma_y \to 0$)에서 $\approx \sqrt\pi/(6\sigma_y)$로 커진다.
  - H4b(near-tie 밀도가 높은 과제에서 CRN 이득이 크다)의 예측식이다.
  - 상수 $\sigma_y$ = 마지막 반복 모집단에서 양자화 층 활성의 후보 간 std / Δ다. held 분할에서 잴 수 있다.
- 후보에 따라 민감도 $G_i(a)$가 변하면 $(u - \tfrac12)$ 항이 $(\Delta^2/12)\mathrm{Var}_a[\sum_i G_i(a)]$만큼 남는다. CRN은 민감도 변화는 지우지 못한다.

### 2.6 모형 (4) 고정 공유 dither와 표 크기 $N_{eff}$

- 표를 고정하면 오차는 계획 안에서 입력의 결정적 함수다. 그래서 **모형 (1)과 같다.** 2.3절이 그대로 적용되고, $\sigma_b^2$는 고정 표 오차의 후보 간 분산이다.

**정확식(선형화 readout).**
- $e(a) = (c\Delta/K)\sum_m G_m(a)\, r(y_m(a); u_{g(m)})$이고, 같은 group g의 항목은 오프셋 $u_g$를 공유한다.
- $\mathbb E_{u_g}[r] = 0$이 모든 y에서 성립한다. 그래서 group 사이 공분산은 표에 대한 기대에서 0이다.
- 브라운 다리 커널의 Fourier 형태 $C(y,y') = \sum_p [1 + \cos 2\pi p(y-y') - \cos 2\pi p y - \cos 2\pi p y']/(2\pi^2p^2)$를 쓰면 다음이 된다.

$$\mathbb E_u\,\mathrm{Var}_a[e] = \Big(\frac{c\Delta}{K}\Big)^2\sum_g\sum_{p\ge1}\frac{\mathrm{Var}_a\big[Z_{g,p}(a) - S_g(a)\big]}{2\pi^2p^2}, \quad Z_{g,p} = \sum_{m\in g}G_m e^{2\pi i p y_m},\ S_g = \sum_{m\in g}G_m. \tag{2.6.1}$$

**두 극한**
- **일관 group**(구성원이 y mod 1과 민감도 변화를 함께 움직임)
  - $\mathbb E_u\mathrm{Var} = (c\Delta \bar G)^2 V(\sigma_y)/N_{eff}$에 $(c\Delta)^2\mathrm{Var}_a(G)/(12 N_{eff})$가 더해진다.
  - 설계의 **$1/N_{eff}$ 법칙**이다. $A = (c\Delta)^2[\bar G^2 V(\sigma_y) + \mathrm{Var}(G)/12]$
- **비일관 구성원**(위상 독립·퍼짐 ≫ 1 또는 독립 민감도): 교차 항이 사라진다. 공유 방식과 **무관한 바닥** B가 된다.
- 일반형은 일관 축을 따라 $A/N_{eff} + B$, 비일관 축을 따라 평탄이다.
- group을 합칠 때 분산은 구성원 사이 교차 공분산만큼 바뀐다. 이 공분산은 양(일관), 0(비일관), 음(반대 방향 민감도)이 다 가능하다. 그래서 **단조성은 보장되지 않는다.**

**두 모델의 $N_{eff}$와 readout**
- TD-MPC2 sh-step/sh-ep 표: (층 4개: dyn0, dyn1, rew0, rew1) × horizon 3 × unit 512
  - 비용 평가 하나가 독립 오프셋 6,144개를 쓴다(항목마다 하나).
  - E3 sweep은 층당 1,536 → 512 → 32 → 1이다.
  - latent가 벡터 하나라 token 평균이 없다. unit은 노름을 보존하는 선형층으로 섞여 스칼라 가치로 간다.
  - 그래서 $G_m \sim O(1/\sqrt M)$이다. 비일관 바닥도 det 오차도 폭과 함께 줄지 않는다.
- DINO-WM 표: 층·horizon step마다 (token 196·nh) × unit 2048(0.4~1.2M)
  - 비용이 최종 프레임 patch 196 × 384 차원을 평균한다. 그래서 $G_m \sim O(1/M)$이고 비일관 성분은 1/M로 상쇄된다.
  - det의 **일관 성분**(GELU 출력이 0 근처 한 칸에 몰린 군집, R2-13)은 상쇄되지 않는다.
- R2-8b(TD-MPC2 sh-step ≈ det)와 R2-13(DINO-WM sh-fixed 무해)의 차이에 대한 이 이론의 설명
  - 오프셋 개수 자체가 아니다.
  - (readout이 평균하는가) × (det 오차가 일관된가) × (공유가 일관 축을 따르는가)다.

**E3 예측식(명시)**
- $\sigma^2_{fix}(N_{eff}) = A/N_{eff} + B$(일관 축)이다. 손상과 OEI는 2.3절의 지속장 식에 $\sigma_b^2 = \sigma^2_{fix}$를 넣는다.
  - sel_last ∝ $\sigma^2_{fix}/\sigma_{\hat J}\cdot S$
  - 매끄러운 성분의 drift ∝ $\sigma^2_{fix}/\ell^2$
  - 그래서 near-tie에서는 $|\mathrm{OEI}| \propto (A/N_{eff} + B)^{1/2}$, drift형 손상은 $\propto A/N_{eff} + B$로 예측한다.
- 상수
  - A, B: 고정 표의 후보 간 오차 분산을 held 분할에서 두 수준 이상 재어 맞춘다.
  - ℓ: 오차의 행동 공간 상관 길이. 자유 상수이고, 추가 측정이 필요하다.
  - $S$, $\sigma_{J,T}$, $\theta_T$, $k_{eff}$: 실험 기록에서 잰다.

### 2.7 det+shift의 위치

- $q = \mathrm{round}(y - o) + o$, $o \sim U[-\tfrac12, \tfrac12)$(후보 공유, forward마다 새로)
- 오차는 $\mathrm{round}(y-o) - (y-o) = \mathrm{saw}(y + u)$, $u = \tfrac12 - o$다. 즉 **SR 오차에서 $(u-\tfrac12)$를 뺀 톱니 부분만** 남는다.
  - o에 대해 모든 입력에서 평균 0이다.
  - 모든 y에서 $[-\tfrac12, \tfrac12)$ 균등이라 분산 1/12가 입력과 무관하다. 감산형 dither다.
- 한 평가 안에서는 입력의 결정적 함수다(det와 같다). 평가 사이에서는 새롭다(sh-iter와 같다). 그래서 **(3)에 속한다.**
- 이용 가능한(비공통) 분산은 sh-iter의 B 항뿐이다. 오프셋 편향 × 민감도 변화 항(A 항)은 없다.
- 항목 MSE는 Δ²/12로 SR의 Δ²/6의 절반이다. 오차 분산이 입력과 무관해 Jensen 분산 변조장(2.8절)도 없다.
- 예측: $\mathbb E[\text{drift}] = 0$, 재평가 sel_last = 0, 같은 평가 sel_last의 크기 ≤ sh-iter, $D_{det+shift} \le D_{sh\text{-}iter}$
  - "지속 편향만 깨도 충분한가"(4.2절 표)에 대한 이론의 답은 **그렇다**다. 반복 안의 이용은 sh-iter와 같은 수준으로 남는다.
- shift를 계획 동안 고정하면 (1)이 된다. 무작위로 이동한 격자 위의 det다.

### 2.8 비편향 dither의 비선형 2차 편향장 (조정자 2번)

- $\hat J(a;\varepsilon)$를 모든 양자화 항목의 활성 오차가 ε일 때의 모델 비용이라 하자.
- 비편향 dither는 $\mathbb E_u\varepsilon_i = 0$이다. SR(비감산형)은 1차 모멘트가 비편향이고, det+shift(감산형)는 오차가 입력과 독립이다.
- 2차 전개는 다음과 같다.

$$b_2(a) \equiv \mathbb E_u[\hat J(a;\varepsilon)] - \hat J_0(a) = \tfrac12\sum_i H_{ii}(a)\,\mathrm{Var}_u(\varepsilon_i) + \tfrac12\sum_{i\ne j}H_{ij}\,\mathrm{Cov}_u(\varepsilon_i,\varepsilon_j) + O(\Delta^3).$$

- H는 하류 전체(뒤 층과 비용 head)의 헤시안이다.
- 한 후보 안에서 항목들의 오프셋은 ind와 sh-iter 모두 독립이다(sh-iter는 후보 사이에서 공유할 뿐이다). 그래서 교차 항이 사라지고 다음이 남는다.

$$b_2(a) = \tfrac12\sum_\ell \Delta_\ell^2\sum_{i\in\ell}H_{ii}(a)\,v_i(a), \qquad v_i = f_i(1-f_i)\ \text{(SR)},\quad v_i = \tfrac1{12}\ \text{(det+shift)}.$$

- 두 성분
  - 곡률 지도 $\tfrac12\Delta^2\bar v\sum H_{ii}(a)$: 매끄럽고, 두 방식 모두에 있다.
  - 분산 변조 $\tfrac12\Delta^2\sum H_{ii}(f_i(1-f_i) - \tfrac16)$: SR에만 있다. 활성 공간에서 격자 주기로 진동하므로 상관 길이가 det 오차와 같은 급이다.
- **비트 척도**: $\Delta_\ell = (hi - lo)/(2^k-1)$이므로 $b_2 \propto 4^{-k}$다. 1차 새 잡음과 det의 비일관 오차는 $\propto 2^{-k}$다. det의 일관 성분은 군집이 해상되기 전까지 줄지 않는다.
- **이용**: $b_2$는 기대값에서 a의 결정적 함수다. 그래서 새 dither에서도 계획기는 $J + b_2 + \varepsilon_t$를 본다(ε에는 SA, $b_2$에는 SAA). 기대 drift는 후보 간 std $\sigma_{b_2}$인 지속장의 drift(2.3절)와 같다.
- **det drift 대비 크기**(비일관 항목, SAA 영역, 같은 상관 길이)

$$\frac{\mathrm{sd}(b_2)}{\mathrm{sd}(e_{det})} = \frac{\tfrac12\Delta^2\sqrt{\sum H_{ii}^2/180}}{\Delta\sqrt{\sum g_i^2/12}} = \frac{\Delta\,(H/g)}{2\sqrt{15}}, \qquad \frac{\text{drift}_{b_2}}{\text{drift}_{det}} \approx \frac{\Delta^2(H/g)^2}{60} \propto 4^{-k}.$$

  - $g_i = \partial\hat J/\partial x_i$이고 $H/g$는 활성 하나에 대한 곡률/기울기 비(활성 단위)다.
  - $H/g \sim 1/R_x$(활성 범위)면 비는 $\approx 1/(60(2^k-1)^2)$이다. k = 3에서 $3\times10^{-4}$, k = 4에서 $7\times10^{-5}$다.
  - 목표 근처처럼 1차 민감도 $g$가 0으로 가는 곳에서는 비가 커진다. MSE 목표 비용 head에서 $g \propto$ 잔차다.
- 측정 제안(E2): held 분할 후보 집합의 모델 비용을 새 draw 여러 번(예: 64회)으로 평균해 $b_2$를 얻는다. 그 후보 간 std와 drift를 SR과 det+shift에서 비교한다.

### 2.9 OEI 분해 요약 (조정자 1번)

| 모형 | sel_last, 같은 평가값 | sel_last, 독립 재평가 | drift | N | elite k | 반복 T |
|---|---|---|---|---|---|---|
| (1) 지속장 b(det) | $-\frac{\sigma_{b,T}^2+\rho\sigma\sigma}{\sigma_{\hat J,T}}S_{op}$ | = 같은 평가값 | < 0. 약한 장 $\approx -\frac{\sigma_b^2\mathrm{tr}H^{-1}}{\ell^2}f(\sigma_T)$, 강한 장 극값 | sel: $S_{op}(N)$. drift: 고정 q에서 평탄 | sel: k↑이면 작아짐. drift: 약함 | drift는 커지다 포화(SAA). sel은 $\sigma_{b,T}S$로 포화 |
| (2) 새 독립(ind) | $-\frac{\sigma_\varepsilon^2}{\sigma_{\hat J,T}}S_{op}$ (≠ 0) | 0 | 0(정확), Jensen만 | $S_{op}(N)$ | k↑이면 작아짐 | near-tie로 가면 $\to\sigma_\varepsilon S$ |
| (3) 새 공유(sh-iter) | $-\frac{\sigma_{\perp,T}^2}{\sigma_{\hat J,T}}S_{op}$ | 0 | 0, Jensen만 | 〃 | 〃 | 매끄러운 성분은 붕괴와 함께 줄어듦 |
| (4) 고정 표(sh-step/ep) | (1)과 같고 $\sigma^2 = A/N_{eff}+B$ | = 같은 평가값 | (1)과 같음 | 〃 | 〃 | 〃 |
| det+shift | (3)에서 B 항만 | 0 | 0, 곡률 지도만 | 〃 | 〃 | 〃 |

- **실험 OEI로 옮기면**
  - TD-MPC2 OEI = sel_last(같은 평가) + drift다. 새 잡음 변형에서도 $\mathbb E[\mathrm{OEI}] = $ 마지막 반복 winner's curse ≠ 0이다(7.1절 결정 사항).
  - DINO-WM `oei_ref_mf` = sel_last(최종 평균 재평가) + drift다.
    - 새 잡음 변형: $\mathbb E = 0$(Jensen 제외)
    - 지속 변형: 붕괴 뒤 ≈ drift
  - DINO-WM 보조 `oei_ref_elite`(마지막 top-30 평균의 같은 평가 오차)와 `oei_ref_argmin`은 같은 평가 선택을 포함한다. 그래서 새 잡음에서도 음수다.

### 2.10 폐루프와 warm start

- warm start는 다음 계획을 이동한 이전 평균에서 시작한다.
- 제어 step을 넘어 지속하는 오차(det, sh-ep)는 이용된 이동이 다음 step으로 넘어간다. 그래서 n = 1에서도 step을 거쳐 누적된다(R2-14의 n = 1 det 손상).
- sh-step은 계획 안에서만 고정이다. 계획 안의 SAA(n > 1)는 있지만, 같은 장이 다음 step으로 넘어가지는 않는다(warm start로 편향된 평균만 넘어간다).
- 예측: (sh-step − sh-iter) 손상 차는 n = 1에서 0이고 n과 함께 커진다. warm start가 있으면 n = 1에서 det형 > sh-step형이다.

---

## 3. 예측식과 상수

**상수 표기**
- [기록] E1·E2 기록이나 held 분할에서 바로 얻는 상수
- [추정] held 분할에서 추가 측정이 필요한 상수
- [자유] 학습된 모델에서는 직접 얻을 수 없는 상수. 비교는 비(ratio)로 해서 이 상수를 지운다.

| # | 예측 | 식 | 상수 |
|---|---|---|---|
| P1 | 마지막 평가 안의 선택(같은 평가값) | $\text{sel\_last} = -\frac{\sigma_{e,T}^2 + \rho_T\sigma_{J,T}\sigma_{e,T}}{\sigma_{\hat J,T}}S_{op}(N,k,\theta_T)$ | $\sigma_{e,T}$, $\sigma_{J,T}$ [기록: TD-MPC2 `l_e_rand_sd`, `l_sd_v0`, DINO-WM 마지막 모집단 `spec_last`·J_none]. $\theta_T = \lambda\sigma_{\hat J,T}$ [기록 근사]. $\rho_T$ [추정, 없으면 0]. $S_{op}$는 표(정확) |
| P1′ | 같은 항, 분포 무관 | $-\sigma_{\perp,T}^2[\lambda(1-\sum w^2) + N f_{\hat J}(t_k)w_{(k)}]$(MPPI), $-\sigma_{\perp,T}^2 f_{\hat J}(t_k)/q$(CEM) | $f_{\hat J}(t_k)$ = elite 경계의 모델 비용 밀도 [기록: DINO-WM pops 저장. TD-MPC2는 추가 기록 필요] |
| P1″ | argmin, near-tie | $-\sigma_\perp e_{n_c}$, $n_c = \sum_j 2\Phi(-(J_j-J_{min})/(\sqrt2\sigma_\perp))$ | 모집단의 무양자화 비용 [기록: DINO-WM pops] |
| P2 | 새 잡음의 drift와 재평가 sel_last | $\mathbb E = 0$(정확). Jensen 항만 남음 | — |
| P3 | 지속장 drift(약한 장) | $-\frac{\sigma_b^2\mathrm{tr}H^{-1}}{\ell^2}(1+2\sigma_T^2/\ell^2)^{-d/2-1}$(= −2 × regret), 강한 장은 $-\sigma_b\sqrt{2\ln n_{loc}}$ | $\sigma_b$ [기록 `f_e_rand_sd`], $\sigma_T$ [기록], ℓ·H [자유]. 같은 ℓ·H의 변형 사이에서는 drift ∝ $\sigma_b^2$(비 예측) |
| P4 | 반복 수 | 새 잡음: 손상이 $d\sigma_\varepsilon/(4Sk_{eff})$ 바닥으로 감소·포화(MPPI, std 바닥이 있을 때. 정량은 최대 약 3배 어긋남, 5.5). 지속: 손상이 SAA 점에서 평탄 | $k_{eff}$ [기록 가능], $\sigma_\varepsilon$ [기록] |
| P5 | N 용량-반응(H2c) | $\frac{|\mathrm{OEI}|(N)}{|\mathrm{OEI}|(N_0)} \approx w_{sel}\frac{S_{op}(N)}{S_{op}(N_0)} + w_{drift}\cdot 1$, $w$ = 기준 N에서 두 항의 몫 | $S_{op}$ [표], 두 항의 몫 [기록: 분해] |
| P6 | CRN 이득(H4b) | $G_{CRN} = \frac{1/6}{V(\sigma_y)}$, $V(\sigma_y) = \frac1{12} - \sum_m\frac{e^{-4\pi^2m^2\sigma_y^2}}{2\pi^2m^2}$ | $\sigma_y$ = 마지막 모집단의 양자화 층 활성 후보 간 std / Δ [추정, held] |
| P7 | 고정 표 크기(H4a) | $\sigma^2_{fix}(N_{eff}) = A/N_{eff} + B$(일관 축만), 손상·OEI는 P1·P3에 $\sigma_b^2 = \sigma_{fix}^2$를 넣음. 정확식 (2.6.1) | A, B [추정: 수준 2개 이상에서 고정 표 후보 간 분산을 잼] |
| P8 | det+shift | drift = 0, $|\text{sel\_last}| \le$ sh-iter, $D \le D_{sh\text{-}iter}$ | — |
| P9 | Jensen 편향 | $b_2 = \tfrac12\sum\Delta^2H_{ii}v_i$, $\frac{\text{drift}_{b_2}}{\text{drift}_{det}} \approx \frac{\Delta^2(H/g)^2}{60}\propto4^{-k}$ | H/g [자유, 여러 draw 평균으로 추정 가능] |

---

## 4. 모의 격자 (사전 선언, worklogs/E4_theory.md 07:24Z)

- **공통**
  - 2차 목적함수 $J = \tfrac12(a-a^*)^\top H(a-a^*)$, $a^* \sim U[-0.5, 0.5]^d$
  - sep: 고유값 log-균등 [1, 10]
  - tie: 절반을 [0.01, 0.1]로 둔다(평평한 방향).
  - d ∈ {2, 6, 12}(horizon × action = 1×2, 3×2, 3×4)
- 오차 척도 r = (첫 모집단 후보 간 오차 std) / (첫 모집단 near-tie gap g0)이고, r ∈ {0.5, 2, 8, 32, 128}이다.
  - g0/σ_J0는 MPPI sep 0.070, tie 0.013, CEM sep 0.035, tie 0.009였다.
  - 그래서 r의 범위는 σ_e0/σ_J0 ≈ 0.004~9다. 파일럿 TD-MPC2는 σ_e0/σ_J0 ≈ 1, gap/std ≈ 0.005였다.
- **오차 모형**(블록 C·E·R·J를 빼면 모두 첫 모집단 후보 간 std를 r·g0로 맞춤)
  - M1 지속 RFF 장(ℓ 0.3)
  - M2 새 i.i.d.
  - M3 새 RFF 장 + 공통 충격(ρ_c 0.5)
  - 반올림 toy(512 entry, ν = 3칸/단위 행동, 선형 readout 1 + γ·민감도 변화): R-det, R-shiter, R-fixed(entry마다 계획 고정 = sh-step/sh-ep), R-detshift
- **블록**

| 블록 | 내용 | 계획기 실행 |
|---|---|---|
| M | {MPPI, CEM} × {sep, tie} × N {64, 128, 256, 512, 1024} × elite {q 2%, 12.5%, 25%, k 고정(64/30)} × 모형 7 × r 5, d = 6 | 737,280 |
| D | 차원 d {2, 6, 12}, 기본 계획기(512/64, 300/30), r {8, 32} | 38,400 |
| L | M1·M3의 ℓ {0.03, 0.1, 1, 3} | 17,408 |
| C | CRN: M3 ρ_c {0, 0.5, 0.9, 0.99}와 M2를 같은 주변 진폭으로 | 11,264 |
| E | 고정 표 크기: toy 16 token × 256 unit, N_eff {4096, 1024, 256, 64, 16, 4, 1}, 일관/비일관, 같은 비트 + det·ind·shiter·detshift | 47,104 |
| R | 같은 비트 반올림 변형 5종 × 특징 {균등 위상, 군집 위상 + 방향 일관} | 22,528 |
| J | Jensen: 2차 head toy × 비트 {3, 4, 5, 6} × κ {1, 10} × {det, ind, shiter, detshift, fixed} | 41,984 |
| V | 장기 반복(MPPI 100회, k {8, 32, 64, 128}; CEM 60회), M1·M2 | 6,400 |
| A | 단발 선택(정규 (J, e) 20,000회; 2차 첫 모집단 2,000개) | — |
| W | 폐루프 warm start(LQ 적분기, 25 step, n {1, 3, 6, 12} × warm start {on, off} × {M1j, Mstep, M3, M2}) | 64 episode × 칸 |
| EV / JV | 고정 표 분산 직접 측정(정확식 비교) / Jensen 장 직접 측정 | — |

- 인스턴스 128개 × 계획기 seed 2개(재실행 편차·지속 지수)
- 모든 오차 모형이 같은 계획기 난수열로 짝을 이룬다. 오차 draw는 척도 사이에서 공유한다.

---

## 5. 결과: 이론과 모의의 일치

- 전체 표: `results/E4/e4_tables.md`. 칸별 pooling 통계: `cells_mppi.csv.gz`, `cells_cem.csv.gz`
- 칸 값은 인스턴스 128개 × seed 2개(= 256 계획) 평균이다. SE는 그 표준오차다.
- 적합 지표
  - 기울기: 원점을 지나는 회귀
  - R²
  - 상대 오차 중앙값: 측정이 2 SE 밖으로 0과 다른 칸만
- **사후 수치 유효성 필터**(08:58Z): CEM 모집단이 SAA 고정점 위로 붕괴한 칸에서는 ρ → −1이고 모델 비용이 1차로 평평하다. 그래서 2변량 식의 $1/\sigma_{\hat J}$가 발산한다. Stein k-NN 밀도도 동률 후보에서 발산한다. 예측이 측정의 10³배를 넘는 칸은 제외하고 개수를 적었다. 2변량 식 1칸, Stein 최대 169칸이다.
- **유도 수치 점검**(본 실행 전, worklogs 07:27Z)
  - Stein softmax 항등식: MC −0.9245 대 식 −0.9248(비정규 J)
  - 공유 offset의 V(σ): MC와 일치(σ 0.02~1)
  - det+shift = SR − (u − ½) 항등식과 [−½, ½) 균등 오차(분산 0.0835 대 1/12)

### 5.1 C1 단발 선택 (명제 1·3, 그림 `fig1_singleshot.png`)
- 정규 (J, e)
  - 연산자 5종(argmin, top-2%, 12.5%, 25%, MPPI θ 0.1·1·10) × N 5 × ρ 2 × 비 5 = 350칸이다.
  - 상대 오차 중앙값 0.03~0.14%, 기울기 1.00이다. 명제 1은 정확하다.
- 실제 2차 첫 모집단(아래가 막힌 비대칭 J), argmin

| 모집단 | σ_e/σ_J | 측정 | 2변량 | 고정 J 정확식 | near-tie 수 |
|---|---|---|---|---|---|
| MPPI sep | 0.1 | −0.076 | −0.030 | −0.080 | −0.070 |
| MPPI sep | 0.3 | −0.50 | −0.26 | −0.51 | −0.50 |
| MPPI sep | 1 | −2.56 | −2.14 | −2.56 | −2.56 |
| MPPI tie | 0.1 | −0.168 | −0.030 | −0.171 | −0.168 |
| CEM tie | 0.3 | −0.694 | −0.247 | −0.690 | −0.697 |
| CEM sep | 10 | −29.2 | −28.6 | −28.7 | −28.5 |

- 20칸 전체의 상대 오차 중앙값은 2변량 22%, 고정 J 정확식 1.2%(2 SE 안 100%), near-tie 수 근사 0.85%(95%)다.
- 2변량 식은 σ_e ≪ σ_J일 때 2.5~6.4배 과소예측한다. 전역 σ_J가 선택 경계의 near-tie 밀도를 과소평가하기 때문이다. σ_e = σ_J에서는 16~25% 낮고, σ_e ≥ 3σ_J에서는 3% 안이다.

### 5.2 C2 평가 안 선택 항(같은 평가값) 대 예측 (명제 1·2·3, `fig2_selection_vs_pred.png`)

| 비교 | 칸 | 기울기 | R² | 상대 오차 중앙값 |
|---|---|---|---|---|
| CEM elite 평균, 2변량, 전체 | 5,247 | 1.02 | 0.998 | 2.4% |
| CEM elite 평균, 2변량, 새 잡음·마지막 반복 | 982 | 1.01 | 1.000 | 0.4% |
| CEM elite 평균, 2변량, 지속·마지막 반복 | 765 | 1.00 | 0.999 | 6.2% |
| CEM argmin, 2변량 | 5,247 | 1.05 | 0.994 | 4.9% |
| CEM argmin, near-tie 수 | 5,248 | 1.05 | 0.995 | 2.8% |
| MPPI(Gumbel, $S_{MPPI}(N,k,\theta_T)$), 2변량, 전체 | 6,976 | 0.97 | 0.977 | 8.6% |
| MPPI, 2변량, 새 잡음·마지막 | 972 | 1.06 | 0.974 | 2.7% |
| MPPI, 2변량, 지속·마지막 | 760 | 0.81 | 0.924 | 14% |
| MPPI, Stein, 새 잡음(처음·중간·마지막) | 961~1,965 | 0.93~1.03 | 0.94~0.98 | 6~14% |
| MPPI, Stein, 지속 | 748~1,532 | 0.46~0.51 | < 0 | 17~39% |

- 2변량 식은 두 계획기의 실제 선택(elite 평균, argmin, top-k + softmax + Gumbel 추출)을 1~6% 오차로 맞춘다.
- 지속장의 MPPI 마지막 반복에서는 약 20% 과대예측한다. 지속장은 후보 사이에서 공간적으로 상관되어 있어 "후보 간 독립" 가정을 깨기 때문이다. Stein 식은 독립을 직접 가정하므로 더 크게(약 2배) 과대예측한다.

### 5.3 C3 OEI 분해 (2.4~2.9절, `fig3_decomposition.png`)

| 계획기 / 모형군 | 칸 | drift 95% CI가 0 포함 | drift CI < 0 | 재평가 sel_last CI가 0 포함 | 같은 평가 sel_last CI < 0 |
|---|---|---|---|---|---|
| CEM / 새 잡음 | 2,804 | **94.8%** | 1.8% | **95.2%** | **99.6%** |
| CEM / 지속 | 2,204 | 16.9% | **82.7%** | 28.7% | 98.8% |
| MPPI / 새 잡음 | 3,720 | **95.8%** | 2.5% | **95.4%** | **94.4%** |
| MPPI / 지속 | 2,936 | 36.1% | 63.6% | (= 같은 평가값) | 94.4% |

- 새 잡음은 2.4절과 정확히 맞는다. drift와 재평가 sel_last의 CI 포함률이 명목 95%이고, 같은 평가값의 sel_last는 거의 모든 칸에서 음수다.
- 지속 오차의 drift는 CEM에서 83%, MPPI에서 64% 칸이 음수다.
  - MPPI는 $\sigma_{min}$ 바닥 때문에 반올림장이 모집단 척도에서 거칠게 남는다. 그래서 반복 사이 이용이 약하다(2.3(d)).
- **기본 설정 분해**(σ_J0 단위, sep, r = 32)

| 계획기 | 모형 | OEI | sel_last 같은 평가 | sel_last 재평가 | drift | 손상 |
|---|---|---|---|---|---|---|
| MPPI 512/64, n = 6 | M1 지속장 | −9.03 | −1.13 | −1.13 | **−7.90** | 0.67 |
| | R-det | −7.00 | −5.08 | −5.08 | −1.92 | 0.69 |
| | R-fixed(sh-step형) | −7.22 | −4.25 | −4.25 | −2.97 | 0.97 |
| | M2 새 독립 | **−6.25** | −6.25 | −0.04 | −0.005 | 0.35 |
| | M3 새 공유 | −4.77 | −4.71 | −0.09 | −0.06 | 0.34 |
| | R-shiter | −4.12 | −4.31 | 0.14 | 0.20 | 0.46 |
| | R-detshift | −5.34 | −5.37 | 0.11 | 0.03 | 0.50 |
| CEM 300/30, n = 30 | M1 | −4.72 | −0.0005 | −0.0005 | **−4.72** | 0.19 |
| | R-det | −0.91 | −1.56 | −0.50 | −0.41 | 0.08 |
| | R-fixed | −3.09 | −1.58 | −0.59 | −2.50 | 0.87 |
| | M2 | 0.13 | −2.01 | 0.12 | 0.01 | 0.03 |
| | R-shiter | 0.02 | −1.34 | 0.02 | 0.002 | 0.10 |
| | R-detshift | 0.02 | −1.59 | 0.01 | 0.008 | 0.07 |

  - TD-MPC2 정의(MPPI)의 OEI는 새 잡음에서도 지속장과 같은 크기의 음수다(M2 −6.25 대 M1 −9.03).
  - DINO-WM 정의(CEM `oei_ref_mf`)에서는 새 잡음이 ≈ 0이다.

### 5.4 C4 지속장의 SAA (2.3절)
- 점진 평활 SAA 예측(실제 σ_t 궤적으로 J + b_σ의 최소점을 추적)
  - drift: MPPI 기울기 1.01~1.02, R² 0.990~0.996. CEM 기울기 0.92~0.93, R² 0.986
  - CEM OEI_mf: 기울기 0.86~0.89, R² 0.93~0.96
- 수렴 위치는 맞히지 못한다. 이동의 회귀 기울기 중앙값 0.11~0.15, R² 중앙값 < 0이다. 가장 약한 r = 0.5에서도 기울기 0.15~0.34다.
  - 이유: 이 격자에서는 장이 만든 이동이 ℓ과 같은 급이라 강한 장 영역이다. 국소 최소점이 여럿이고 계획기와 추적이 다른 최소점에 간다. **이용된 깊이(drift)는 예측되지만 어느 최소점인지는 seed에 달렸다**(5.12의 지속 지수와 같은 결론).

### 5.5 C5 새 독립 잡음의 정상 손상 (2.4(c), 블록 V, MPPI 100회)

| k | r | k_eff | 측정 손상(평균 계획) | 예측 $d\sigma_\varepsilon/(4Sk_{eff})$ |
|---|---|---|---|---|
| 8 | 8 | 6.9 | 0.22 ± 0.008 | 0.19 |
| 32 | 8 | 24 | 0.11 ± 0.004 | 0.066 |
| 64 | 8 | 43 | 0.10 ± 0.004 | 0.042 |
| 128 | 8 | 73 | 0.092 ± 0.003 | 0.028 |
| 8~128 | 32 | 3.0~4.3 | 0.82~0.90 | 1.1~1.6 |

- 정성은 맞는다. 정상 바닥이 있고, n = 12에서 이미 거의 정상이며(n = 12 대 100의 차가 작음), r = 32에서 θ가 커서 k_eff ≈ 3~4로 elite 수와 무관하다.
- 정량은 틀린다. 기울기 0.68, R² 0.32이고, k_eff에 대한 log-log 기울기는 −0.38(r 8), −0.14(r 32)로 예측 −1보다 훨씬 얕다.
- 모집단 std는 0.07~0.125로 σ_min 0.05까지 줄지 않는다.
- 가능한 원인(검증 안 함): 고정 policy-prior 제안 24개가 잡음 지배 선택에서 계속 elite에 들어가 평균을 prior 중심 쪽으로 당긴다. elite가 많을수록 더 당긴다. 이 당김이 AR(1) 식에 없다.

### 5.6 C6 반복 수 (2.4·2.10절, `fig5_iterations.png`)
- **반복 후반 손상 추세**(사후 요약, 블록 M)
  - MPPI n 6 → 12
    - 새 잡음 4종의 손상 비 중앙값 0.75~0.91(줄어든 group 74~85%)
    - 지속: M1 1.05(줄어든 group 18%), R-det·R-fixed 0.97~0.98
  - CEM n 10 → 30
    - 지속 1.33~1.86(줄어든 group ≤ 4%)
    - 새 잡음도 0.98~1.52
    - CEM은 std 바닥이 없다. 붕괴가 잡음이 만든 평균 변위를 얼려서, 반복이 새 잡음을 평균하지 못한다. 2.4(c)의 "std가 아래로 유계" 조건이 실제로 갈림길이다.
- **폐루프**(블록 W, 상대 손상 = 폐루프 비용 증가/기준)

| warm start | r | Mstep − M3 (n = 1) | n = 3 | n = 6 | n = 12 |
|---|---|---|---|---|---|
| on | 8 | −0.012 ± 0.020 | −0.016 ± 0.018 | 0.071 ± 0.025 | **0.223 ± 0.047** |
| on | 32 | −0.071 ± 0.037 | 0.107 ± 0.042 | 0.130 ± 0.059 | **0.391 ± 0.097** |
| off | 8 | 0.004 ± 0.027 | 0.020 ± 0.020 | 0.103 ± 0.041 | **0.271 ± 0.066** |
| off | 32 | 0.025 ± 0.049 | −0.026 ± 0.040 | 0.071 ± 0.064 | **0.340 ± 0.073** |

  - 2.10절 예측(n = 1에서 0, n과 함께 증가)과 R2-14의 cheetah 모양이 그대로 나온다.
  - 지속형(M1j, Mstep)의 상대 손상은 n과 함께 계속 커진다(r 8, warm start: 0.16 → 0.63).
  - 새 잡음형은 n 3~6에서 정점을 찍고 준다(M2: 0.18 → 0.40 → 0.23).
  - warm start가 n = 1의 지속형 손상을 키운다는 예측은 이 toy에서 보이지 않았다(M1j 0.160 대 0.183). 상태가 움직이면서 장이 제어 step 사이에서 상관을 잃기 때문으로 본다.

### 5.7 C7 N 용량-반응 (H2c 예측, `fig4_dose_N.png`)

| 계획기 / elite 규칙 / 모형군 | |OEI| 비(N 1024/64) 중앙값 | Spearman(N, \|OEI\|) ≥ 0.8인 group | Spearman(N, 손상) 중앙값 | Spearman(손상, \|OEI\|) ≥ 0.8 |
|---|---|---|---|---|
| MPPI / k 고정(64) / 지속 | **5.42** | 100% | +0.9 | 53% |
| MPPI / k 고정 / 새 잡음 | 4.33 | 98% | +0.1 | 20% |
| MPPI / q 12.5% / 지속 | **1.11** | 43% | −0.7 | 30% |
| MPPI / q 12.5% / 새 잡음 | 1.12 | 50% | −0.7 | 5% |
| MPPI / q 2% / 지속 | 1.23 | 83% | −0.9 | 13% |
| MPPI / q 25% / 지속 | 1.12 | 50% | −0.15 | 23% |
| CEM / k 고정(30) / 지속 | **1.72** | 97% | +1.0 | 80% |
| CEM / q 12.5% / 지속 | **1.06** | 60% | −0.85 | 10% |
| CEM / q 25% / 지속 | 0.99 | 10% | −0.9 | 37% |
| CEM / q 2% / 지속 | 1.40 | 77% | −1.0 | 3% |

- 연산자만의 예측(2.2절)과 방향이 같다. 고정 elite 수에서는 크게 오르고(+70~440%), 고정 elite 비율에서는 조금 오른다(−1~+40%).
- 고정 비율에서 **손상은 오히려 N과 함께 준다**(Spearman 중앙값 −0.7 ~ −1). elite가 늘면 거친 오차가 평균되기 때문이다.
- CEM + 거친 반올림 det(R-det, q 12.5%, r 32)에서는 |OEI|도 N과 함께 크게 준다(30.5 → 0.71).
- 칸마다 계획 256개로 실험보다 잡음이 훨씬 적다. 그런데도 고정 비율에서는 Spearman ≥ 0.8이 group의 10~83%에서만 성립한다.

### 5.8 C8 CRN (2.5절, 블록 C, 같은 주변 진폭)

| 계획기 / regime / r | ρ_c 0 → 0.5 → 0.9 → 0.99: sel_last / sel_last(0) | 예측 (1−ρ_c) | 손상 |
|---|---|---|---|
| CEM / sep / 32 | 1 → 0.53 → 0.105 → 0.014 | 0.5, 0.1, 0.01 | 1.01 → 0.85 → 0.45 → 0.16 |
| MPPI / sep / 32 | 1 → 0.65 → 0.24 → 0.051 | (near-tie면 √(1−ρ_c): 0.71, 0.32, 0.10) | 1.50 → 1.13 → 0.62 → 0.25 |

- 공통 성분은 선택에서 사라지고, 손상은 비공통 분산과 함께 준다.
- 같은 주변 진폭의 백색 잡음(M2)은 sel_last가 크다. 그런데 손상은 매끄러운 새 장(ρ_c 0)보다 작다(CEM 0.29 대 1.01).
  - 매끄러운 새 오차는 elite 집합 전체를 한 방향으로 민다. 반복마다 무작위 기울기이고, elite 수로 평균되지 않는다.

### 5.9 C9 고정 표 크기 N_eff (2.6절, `fig6_neff.png`)
- **정확식 (2.6.1) 대 측정**(표 32개 평균, 1,344칸): 측정/정확 비 중앙값 1.03, R² 0.985
  - 48차 절단 때문에 정확식이 1~2% 낮다.
  - 2 SE 안 비율은 77%다. 작은 N_eff에서 표 사이 분포가 크게 비대칭이라 32개 표의 SE가 작게 잡힌다.
- **일관 toy**(token 위상·민감도 같음, unit 간 민감도 변화 공유)
  - Var(N_eff = 1)/Var(4096) = 17(퍼짐 0.003) ~ 1,520(퍼짐 1)
  - log-log 기울기 −0.29 ~ −0.85
  - A/N_eff + B의 R²는 0.16(near-tie) ~ 1.00(퍼짐 ≥ 0.3)이다.
  - near-tie에서는 두 구간 모양이다. 일관한 token 공유 구간(4096 → 256)은 가파르고, 비일관한 unit 공유 구간(256 → 1)은 평탄하다. 정확식은 둘 다 맞힌다.
- **비일관 toy**: 모든 퍼짐에서 평탄(비 0.98~1.02)
- **계획기 손상**

| 계획기 / regime / toy / r | N_eff 1 | 16 | 256 | 4096 | Spearman(N_eff, 손상) | det | sh-iter | det+shift |
|---|---|---|---|---|---|---|---|---|
| CEM / sep / 일관 / 32 | 168 | 42.6 | 13.5 | 1.42 | **−1** | 6.87 | 0.74 | 0.73 |
| MPPI / sep / 일관 / 32 | 9.86 | 7.54 | 6.23 | 1.24 | **−1** | 6.69 | 1.03 | 1.10 |
| CEM / tie / 일관 / 8 | 2.64 | 0.385 | 0.116 | 0.028 | **−1** | 0.088 | 0.012 | 0.012 |
| CEM / sep / 비일관 / 32 | 8.12 | 10.0 | 10.4 | 10.6 | **+0.75** | 0.69 | 1.33 | 0.63 |
| MPPI / sep / 비일관 / 8 | 0.70 | 0.75 | 0.74 | 0.73 | +0.04 | 0.32 | 0.39 | 0.32 |

  - 일관 toy에서는 8칸 모두 Spearman −1이고, N_eff 1 → 4096에서 손상이 8~260배 준다.
  - 비일관 toy에서는 평탄하거나 오히려 오른다(Spearman −0.32 ~ +0.89).
  - (사후) 일관 toy에서 log 손상 대 log(측정 표 분산)의 기울기는 0.28~0.84다. 2.6절의 ½(near-tie sel)~1(drift) 범위와 맞는다.
  - **비일관 toy에서 고정 표는 N_eff와 무관하게 det보다 2~15배 해롭다**(CEM sep r 32: 10.6 대 0.69).
    - 고정 $(u-\tfrac12)$ × 민감도 변화(A 항)가 매끄러운 지속장을 만들고, det에는 이 항이 없다.
    - R2-8b의 "sh-ep가 det보다 나쁘다"에 대한 기전 후보다.

### 5.10 C10 det+shift 대 sh-iter (2.7절)

| 비교 | 칸 | det+shift 손상 ≤ sh-iter | 유의하게 나쁨(2 SE) | 손상 비 중앙값 |
|---|---|---|---|---|
| 같은 비트(블록 E·R), CEM | 16 | 88% | 0% | 0.83~0.97 |
| 같은 비트(블록 E·R), MPPI | 16 | 63% | 0% | 0.97 |
| 오차 분산 맞춤(블록 M·D), CEM | 212 | 33~67% | 0~39% | 0.94~1.10 |
| 오차 분산 맞춤(블록 M·D), MPPI | 208 | 0~17% | 32~63% | 1.11~1.21 |

- **같은 비트에서는 det+shift가 sh-iter보다 유의하게 나쁜 칸이 없다**(0/32, 손상 비 중앙값 0.83~0.97). 예측(≤)과 맞는다. A 항이 큰 비일관 toy에서는 det+shift 손상이 sh-iter의 절반이다.
- 첫 모집단 오차 분산을 맞춘 비교(블록 M·D)에서는 det+shift가 조금 나쁘다(1.1~1.2배).
  - det+shift의 오차는 전부 톱니 성분이다. 모집단이 줄어도 후보 간 분산이 선형으로만 준다. sh-iter의 매끄러운 A 항은 더 빨리 준다.
  - 그래서 이 비교는 2.7절의 같은 비트 예측을 시험하지 않는다.

### 5.11 C11 Jensen 2차 편향 (2.8절, `fig7_jensen.png`)
- **직접 측정**(JV)
  - log2 sd(b₂)/bit는 −2.07 ~ −2.12(퍼짐 ≥ 0.1, 예측 −2)이고, 가장 좁은 퍼짐 0.03에서 −1.65다.
  - det 오차는 −1.05 ~ −1.09(κ 1, 예측 −1)다.
  - det+shift의 b₂는 상수다(후보 간 sd ~10⁻¹⁶).
- sd(b₂)/sd(det)는 4×10⁻⁴(퍼짐 1, k 6, κ 1) ~ 0.11(퍼짐 0.03~0.1, κ 10)이다. 최적점 근처(1차 민감도가 0으로 감)와 강한 곡률에서 가장 크다(2.8절).
- **계획기 drift**(블록 J)
  - det drift는 비트당 약 4.7배 준다(CEM sep κ 10: −11.4, −2.36, −0.50, −0.11 = $\propto 4^{-k}$).
  - SR 변형의 drift는 32칸 중 30칸에서 |drift| < 2 SE로 0과 구별되지 않는다. 예측 비 $(sd\ 비)^2$는 10⁻⁷~10⁻³이다. det+shift drift도 ≈ 0이다.
  - **이 toy에서 Jensen drift는 검출 한계 아래다.**
- 파일럿 적용(사후, 6.2절): R2-13의 확률적 변형 셋은 같은 drift를 공유한다(Wall −0.06~−0.07, PushT −0.002~−0.005). det drift의 3%, 0.6~1.4%이고 Jensen 장의 형태와 맞는다. toy보다 크다(MSE 목표 head는 목표 근처에서 H/g가 큼).

### 5.12 C12 지속 지수와 계획 편차

| 계획기 / 모형군 | 지속 지수 중앙값(예측: 지속 ≈ 1, 새 ≈ 0) | 첫 행동 편차/seed 바닥 | 재실행 퍼짐/seed 바닥 |
|---|---|---|---|
| CEM / 지속 | 0.10 | 11.1 | 15.5 |
| CEM / 새 | 0.06 | 6.2 | 8.6 |
| MPPI / 지속 | 0.17 | 3.7 | 4.0 |
| MPPI / 새 | 0.30 | 2.5 | 3.2 |

- 편차 크기는 지속 오차가 크다(1.4~1.8배). 지속 지수는 둘을 가르지 못한다(6.1절).

### 5.13 C13 (사후) 손상의 예측 변수
- 오차 분산을 맞춘 같은 척도 안(블록 M, 모형 7개)에서 Spearman(손상, 예측 변수)은 다음과 같다.
  - −OEI 0.61(MPPI), 0.64(CEM)
  - −drift 0.57, 0.64
  - −sel_last(같은 평가) **−0.43, −0.59**
  - 같은 평가의 선택 항만 보면 해로운 모형을 거꾸로 순위 매긴다.
- 같은 비트(블록 E·R, 변형들)에서는 −OEI 0.55~0.97, −drift 0.45~0.94다.
- 척도를 섞으면 오차 크기가 순위를 지배한다(MPPI −OEI 0.98).

---

## 6. 이론과 모의가 어긋나는 곳

### 6.1 어긋남
1. **지속 지수는 지속 오차와 새 오차를 가르지 못한다**(C12)
   - 예측은 지속 ≈ 1, 새 ≈ 0이었다. 측정 중앙값은 지속 0.10~0.17, 새 0.06~0.30이다.
   - 원인 (a): 강한 지속장은 국소 최소점이 여럿이라 seed마다 다른 최소점에 간다. 5.4의 위치 예측 실패와 같은 원인이다.
   - 원인 (b): 반복 12~30회의 과도 구간에서 새 잡음도 감쇠된 하강 때문에 seed가 공유하는 체계적 지연을 만든다.
   - 2.4절의 "새 잡음은 고정점을 편향시키지 않는다"는 정상 상태 명제다. 유한 반복의 과도 구간까지 덮지 않는다.
   - 실무 함의: 지속 이용을 재는 양으로는 지속 지수보다 drift가 낫다.
2. **SAA 수렴 위치**(C4): 깊이(drift R² 0.93~0.996)는 맞고 위치(R² < 0)는 틀린다. 격자 전체가 강한 장 영역이었다. 약한 장 선형식(`rff_saa_moments`)은 이 격자에서 시험되지 않은 셈이다.
3. **새 독립 잡음의 정상 손상식**(C5)
   - 최대 약 3배 어긋난다. elite 수 의존이 예측($1/k_{eff}$)보다 훨씬 약하다.
   - 원인 후보(검증 안 함)는 policy prior 제안의 당김이다.
4. **2변량 winner's curse**
   - well-separated 단발(C1)에서 최대 6.4배 과소예측한다. near-tie 수 근사와 고정 J 정확식이 고친다.
   - MPPI 마지막 반복의 지속장(C2)에서는 약 20% 과대예측한다. 공간 상관 때문이다.
5. **CEM에서는 반복이 새 잡음을 평균하지 못한다**(C6)
   - 붕괴가 평균을 얼린다. 2.4(c)에 적은 조건(std 하한)이 실제로 결정적이다.
   - 설계 4.4절 2번 문구 "평균 갱신에 들어가는 잡음은 반복에 걸쳐 평균된다"는 MPPI(std 바닥)에만 맞는다.
6. **오차 분산을 맞춘 비교에서 det+shift가 sh-iter보다 조금 나쁘다**(C10). 이론의 예측은 같은 비트에 대한 것이고, 같은 비트에서는 예측대로다.
7. **고정 비율 N 증가에서 거친 det의 |OEI|가 크게 준다**(C7, CEM + R-det). 이론은 평탄 또는 감소를 예측했지만 감소 폭(30.5 → 0.71)은 예상보다 컸다. elite가 늘면 거친 지속장이 반복 사이에서 평균되어 이용되지 않는다.
8. **Jensen drift**(C11): toy에서는 검출 한계 아래였다(예측과 맞음). 파일럿 R2-13에서는 det drift의 1~3% 크기로 보인다. toy의 H/g가 실제 MSE 목표 head보다 작기 때문으로 본다. 모순이라기보다 상수 보정의 차이다.
9. **warm start 누적**(C6): 예측한 n = 1의 지속형 손상 증가가 LQ toy에서 보이지 않았다.
10. **A/N_eff + B**(C9): near-tie에서 맞지 않는다(R² 0.16~0.47). 일관 축과 비일관 축이 섞인 두 구간 모양 때문이다. 정확식 (2.6.1)은 이 모양까지 맞힌다(측정/정확 1.03).

### 6.2 파일럿 수치와의 정합 (사후, 이론의 파일럿 적용. E1 아님)

- R2-13 보고서 2.3절 사후 표(비용 오차 ΔJ = J_변형 − J_none의 평균: 0단계 무작위 / 마지막 모집단 / 선택 후보 = argmin)를 분해하면 다음과 같다.
  - drift = 마지막 − 0단계
  - sel_argmin = 선택 − 마지막(같은 평가값)

| 과제 | 변형 | OEI_argmin | drift | sel_argmin | drift 몫 |
|---|---|---|---|---|---|
| PushT | det | −0.386 | −0.350 | −0.036 | 91% |
| PushT | ind | −0.007 | 0.000 | −0.007 | — |
| PushT | sh-iter | −0.007 | −0.002 | −0.005 | — |
| PushT | sh-fixed | −0.007 | −0.005 | −0.002 | — |
| Wall | det | −2.326 | −2.040 | −0.286 | 88% |
| Wall | ind | −0.123 | −0.068 | −0.055 | — |
| Wall | sh-iter | −0.100 | −0.058 | −0.042 | — |
| Wall | sh-fixed | −0.124 | −0.071 | −0.053 | — |

- det: CEM 붕괴 뒤 OEI의 약 90%가 drift(지속 이용)다. 2.3절(c)·(d)의 예측과 같다.
- 확률적 변형 셋
  - drift가 Wall에서 −0.06~−0.07로 거의 같고, PushT에서 −0.002~−0.005다.
  - 2.8절의 Jensen 장(비감산형 변형 모두에 같다)이 예측하는 형태다. 크기는 det drift의 약 3%(Wall), 0.6~1.4%(PushT)다.
  - 이론의 비일관 어림 $\Delta^2(H/g)^2/60$(k = 3에서 3×10⁻⁴)보다 훨씬 크다. MSE 목표 비용 head는 목표 근처에서 1차 민감도 g가 작아져 H/g가 커진다. 2.8절의 경고와 맞는다.
- 예측(E2에서 검사 가능): det+shift(감산형)는 분산 변조 성분이 없다. 그래서 drift 크기가 sh-iter보다 작아야 한다(곡률 지도 성분만 남음, 항목 분산도 절반).

---

## 7. 사전등록 가설에 대한 함의와 결정 사항

**나는 가설, 정의, 문턱을 하나도 바꾸지 않았다.** 아래는 이론과 모의가 사전등록 문구와 맞지 않는 곳이다. 조정자가 결정할 사항으로 올린다.

### 7.1 [결정 사항 1] H2a의 sh-iter 조건과 TD-MPC2 OEI 정의
- **유도**(2.4, 2.5, 2.9절)
  - TD-MPC2 OEI(DESIGN_main 0.4, v4 6.2)는 실행 elite의 오차를 그 elite를 고른 바로 그 평가의 값으로 잰다.
  - 그래서 OEI = sel_last(같은 평가) + drift이고, 새 잡음 변형에서도 $\mathbb E[\mathrm{OEI}] = -\sigma_{\perp,T}^2 S_{MPPI}/\sigma_{\hat J,T} < 0$이다.
  - 마지막 반복은 깊은 near-tie에 있으므로 이 값은 대략 $-\sigma_{\perp,T}S$다.
  - "sh-iter OEI의 95% CI가 0을 포함"이라는 H2a 조건은, sh-iter가 반복을 넘어 이용되지 않더라도 표본이 많으면 실패하게 정의되어 있다.
- **크기**: 파일럿 R2-8b 결정 수준 자료(마지막 반복 중앙값)로 계산한 예측이다. E1 결과가 아니다.
  - sh-iter의 같은 평가 sel_last는 mw-push −8.9, pick-place −38, cheetah(k4) −3.2(가치 단위)다.
  - det의 반복 안 항은 −98, −42, −3.4다.
  - 세 H2a 과제 중 두 과제에서 sh-iter 항이 det 항과 같은 급이다.
- **모의**: MPPI(TD-MPC2 정의)에서 새 잡음 모형 3,720칸 가운데 같은 평가 sel_last의 95% CI가 0보다 작은 칸이 94.4%다. drift CI는 95.8%, 독립 재평가 sel_last CI는 95.4% 칸에서 0을 포함한다(5.3). 기본 설정(512/64, n 6, sep, r 32)에서 새 독립 잡음의 OEI는 −6.25 σ_J0로 지속장 M1의 −9.03과 같은 급이다. 같은 칸의 CEM `oei_ref_mf`는 +0.13이다.
- **DINO-WM `oei_ref_mf`**는 새 잡음 변형의 최종 평균을 독립 draw로 다시 잰다. 그래서 이론상 기대값이 0이고(Jensen 항 제외) H2a의 의도와 맞는다. 두 계열의 OEI 정의가 이 점에서 다르다.
- **선택지**
  - (a) 등록대로 판정하고, 실패하면 이 유도를 함께 보고한다.
  - (b) 사후 보조로 분해를 보고한다. 지속 이용을 직접 가르는 양은 drift다.
    - drift = `l_e_rand − f_e_rand`, sel_last = `e_chosen − l_e_rand`. 둘 다 E1 episode 기록에 있다.
    - 예측: sh-iter drift의 CI는 0을 포함하고 det drift는 음수다.
  - (c) E2에서 측정 계획마다 실행 elite를 독립 dither draw로 한 번 더 평가한다(forward 1회 추가). 그러면 DINO-WM과 같은 재평가 OEI를 기록할 수 있다.
- **H2b**도 같은 OEI 정의를 쓴다. 새 잡음 변형의 −OEI가 커서 손상과의 순위 상관이 떨어질 수 있다. 모의(C13, 사후)에서는 이 우려가 크지 않았다. 오차 분산을 맞춘 같은 척도 안에서 Spearman(손상, −OEI)이 0.61(MPPI)·0.64(CEM)이고, 같은 비트(블록 E·R)에서 0.55~0.97이다. 같은 평가 선택 항만 따로 보면 −0.43~−0.59(거꾸로)다. 그래서 OEI의 손상 예측력은 drift 성분에서 온다. H2b가 이 정의 때문에 실패할 것이라고 예측하지는 않는다.

### 7.2 [결정 사항 2] H2c(후보 수 용량-반응)의 형태
- **유도**
  - |OEI|의 N 의존은 선택 연산자에서만 온다.
    - sel_last ∝ $S_{op}(N)$
    - drift(지속 이용)는 SAA 고정점이 N과 무관해서 1차로 평탄하다.
  - 연산자만의 예측(64 → 1024)
    - argmin +39%
    - elite 평균, 고정 비율: q 10% +0.6%, 12.5% +1.9%, 25% +1.4%, 2% +12%
    - MPPI q 12.5%: θ_T 0.1이면 +2%, 1이면 +4.5%, 3이면 +25%, 10이면 +38%
    - 고정 elite 수: k 30이면 ×2.7, k 64면 N = 64에서 절단이 없다(S = 0).
  - 파일럿 TD-MPC2 마지막 반복 θ_T는 det 0.05~16, sh-fwd 0.1~6.5였다.
- **함의**
  - 고정 elite 비율로 N을 바꾸면 det |OEI|의 증가가 작다. CEM `oei_ref_mf`는 drift가 지배해 거의 평탄하다. "N과 함께 커진다"(H2c)는 약하거나 성립하지 않을 수 있다.
  - Spearman ≥ 0.8(5수준) 문턱은 약한 신호에서는 잡음에 지배된다.
  - det 손상이 N과 함께 커진다는 예측도 이론에서 나오지 않는다. 지속 성분은 평탄하고, 거친 성분은 elite가 늘면 오히려 평균되어 줄어든다.
- **모의**: 5.7절. 고정 elite 수(MPPI 64, CEM 30)에서는 |OEI|가 N 64 → 1024에서 1.7~5.4배가 되고, Spearman ≥ 0.8이 97~100% group에서 성립한다. 고정 elite 비율에서는 0.99~1.40배이고 Spearman ≥ 0.8이 10~83% group에서만 성립한다. 손상의 Spearman 중앙값은 −0.7 ~ −1(N과 함께 감소)이다. 칸당 계획 256개로 실험보다 잡음이 훨씬 적은데도 그렇다.
- **결정 필요**
  - v5(E2 sweep 고정)에서 N을 바꿀 때 elite를 비율로 둘지 수로 둘지
  - H2c를 분해(sel_last의 $S_{op}$ 예측, drift)와 함께 보고할지

### 7.3 [결정 사항 3] H4a(고정 dither 표 크기)의 예측식
- **유도**(2.6절, 정확식 2.6.1)
  - $1/N_{eff}$ 법칙은 오프셋을 공유하는 항목들의 양자화 위상과 민감도 변화가 후보 사이에서 같이 움직이는 "일관 축"에서만 성립한다.
  - 비일관 축에서는 분산이 공유 방식과 무관하다. group을 합치는 단계에서 비단조도 가능하다.
- **모의**: 5.9절. 일관 toy에서는 고정 표 손상이 N_eff와 함께 단조 감소한다(8칸 모두 Spearman −1, 1 → 4096에서 8~260배). 비일관 toy에서는 평탄하거나 증가한다(Spearman −0.32 ~ +0.89). 후보 간 분산의 정확식은 두 경우를 모두 맞힌다(측정/정확 1.03, R² 0.985).
- **함의**
  - E3 sweep에서 일관 축일 가능성이 높은 단계는 token×unit → unit(DINO-WM)과 horizon×unit → unit(TD-MPC2)이다.
  - unit → group → 층은 unit 사이의 위상·민감도 일관성에 달렸다. 예를 들어 0 근처 GELU·Mish 군집이 일관성을 만든다.
  - 그래서 "N_eff가 커질수록 단조 감소"(H4a)를 이론이 보장하지 않는다.
  - 손상의 직접 예측 변수는 N_eff 자체가 아니라 그 수준에서 잰 고정 표의 후보 간 오차 분산이다. 블록 E 사후 기울기: 일관 toy에서 log 손상 대 log(측정 표 분산)의 기울기는 0.28~0.84다(사후).
- **결정 필요**: E3에서 수준마다 `var_cand`를 함께 재어 보조 예측 변수로 보고할지. 정의 변경이 아니라 측정을 하나 더하는 것이다.

### 7.4 H4b(near-tie 밀도와 CRN 이득): 이론은 지지 방향
- $G_{CRN} = (1/6)/V(\sigma_y)$는 2 이상이고, $\sigma_y \to 0$(near-tie, 늦은 반복)에서 $\approx\sqrt\pi/(6\sigma_y)$로 커진다.
- 측정 변수는 마지막 반복의 양자화 층 활성 후보 간 std / Δ다.
- 모의: 고정 표 직접 측정(EV)에서 ind/sh-iter 오차 분산비는 near-tie 퍼짐(σ_y 0.009)에서 34, 퍼짐 σ_y 3에서 1.3~1.4였다. 식 $G_{CRN} = (1/6)/V(\sigma_y)$의 near-tie 값은 33이다. 블록 C에서는 같은 주변 진폭에서 sel_last가 $(1-\rho_c)$에 비례해 줄었고(CEM 0.53, 0.105, 0.014 대 0.5, 0.1, 0.01), 손상도 함께 줄었다.

### 7.5 설계 문구 정정(가설에는 영향 없음)
- 4.4절 2번 "남는 비용은 분산이고 1/(elite 수 × 반복 수)에 비례해 준다"
- MPPI·CEM은 반복 평균을 하지 않는다. 그래서 잔여 손상은 $\propto\sigma_\varepsilon/(S k_{eff})$(1/elite 수)에서 포화하고, 반복 수에는 그 바닥으로 다가가는 꼴로 준다(2.4절).
- 모의: MPPI에서 새 잡음 손상은 n 6 → 12에서 준다(중앙값 0.75~0.91). 장기 반복(블록 V)에서는 n = 12와 100이 비슷한 정상 바닥이고, elite 의존은 예측보다 약하다. CEM에서는 붕괴 때문에 새 잡음 손상이 n 10 → 30에서 줄지 않는다(0.98~1.52).

### 7.6 det+shift(4.2절 대조)
- 이론: (3)에 속한다. sh-iter보다 손상이 작거나 같고 drift는 0이다. 감산형이라 Jensen 분산 변조가 없다.
- 모의: 같은 비트(블록 E·R)에서 det+shift ≤ sh-iter인 칸은 CEM 88%, MPPI 63%이고 유의하게 나쁜 칸은 0이다. A 항이 큰 비일관 toy에서는 det+shift 손상이 sh-iter의 절반이다(0.63 대 1.33). 오차 분산을 맞춘 비교에서는 1.1~1.2배 나쁘다(5.10).
- 실험에서 det+shift가 sh-iter보다 뚜렷이 나쁘면 모형 밖 요인일 것이다. 격자 끝 clamp, offset 때문에 생기는 2^k 수준의 범위 이동이 후보다.

---

## 8. 편차와 한계

- **사전 선언 전에 바꾼 것**(worklogs 07:00~07:22Z)
  - 오차 척도 격자를 g0 단위 4수준에서 5수준 {0.5, 2, 8, 32, 128}로 넓혔다. 근거는 보정 결과(g0/σ_J0)다.
  - 단발 선택의 정밀 예측식 세 개(고정 J 정확식, near-tie 수, Stein)를 비교에 더했다. 근거는 스모크 블록 A다.
- **사후 추가**(결과를 보며 더한 것, 표에 "사후"로 표시)
  - 블록 E에서 손상 대 고정 표 측정 분산의 기울기
  - C13(손상 예측 변수 비교)
  - 파일럿 수치에 이론 적용(R2-8b의 H2a 예측, R2-13 분해)
- **모형화의 한계**
  - 2차 목적함수와 선형화 readout을 쓴 toy 위의 모의다. 학습된 가치·비용 head의 곡률 H와 오차 상관 길이 ℓ은 자유 상수로 남는다. 그래서 절대 크기가 아니라 비와 추세를 예측한다.
  - 오차 모형 대응
    - (1) = M1(RFF 장) + R-det
    - (2) = M2 + 블록 E·R의 ind
    - (3) = M3 + R-shiter
    - (4) = R-fixed + 블록 E 고정 표
    - det+shift = R-detshift
  - 주 격자의 반올림 변형은 첫 모집단 후보 간 오차 std를 맞춘 비교(구조만 비교)다. 같은 비트 비교는 블록 E, R, J에 있다.
  - MPPI policy prior는 a* 근처의 정규 제안으로 두었다. TD-MPC2는 prior 궤적을 양자화된 dynamics로 만든다. 그 경로는 모의하지 않았다.
  - policy prior 수는 N과 무관하게 24다(N = 64면 표본 40). E2가 다르게 정하면 N = 64 칸의 비교에 영향이 있다.
  - MPPI temperature sweep {0.1, 0.5, 2}는 격자에 넣지 않았다(과제 지시 격자에 없음). temperature는 $\theta = \lambda\sigma_{\hat J}$를 통해 $S_{MPPI}$에 들어가고, 표에 θ 전 범위가 있다.
  - 반복 수 수준은 한 실행에서 기록했다. 반복 상태가 총 반복 수와 무관하므로 따로 돌린 것과 같다.
  - 폐루프(블록 W)는 선형-2차 적분기 toy다.
- E1 결과 파일은 읽지 않았다. 실험 쪽 숫자는 파일럿 보고서·CSV(5장 근거)와 분석 코드의 형식만 썼다.

---

## 9. 겹쳐 그리기 도구 (E2·E3용)

- 코드: `main/code/theory/e4_overlay.py`
  - `python e4_overlay.py <e4_comparisons.json> <s_tables.json> <out_dir> <E2·E3 분석 JSON …>`
  - 시험: `--selftest`. 모의 곡선으로 만든 합성 JSON 세 개(`SYNTHETIC_*.json`)로만 돌렸다. 결과는 `results/E4/overlay_test/`이다.
- **입력 해석**: E1 분석 JSON 두 형식을 그대로 읽는다.
  - TD-MPC2 `{"cells": {"<task>:<k>": {"variants": {v: cell}}}}`
  - DINO-WM `{"tasks": {task: {"confirmatory": {k: {v: cell}}}}}`
  - OEI 키(`oei` 또는 `oei_ref_mf`)를 가진 dict가 칸이다.
  - sweep 변수는 경로 키(`mw-push:3:N512:q0.125:it6:T0.5:neff256`, `{"N": {"512": …}}`)나 칸·부모의 필드(N/num_samples, num_elites/topk/elite_ratio, iterations/opt_steps, temperature, n_eff, warm_start)에서 읽는다.
- **그림 네 가지**(실측은 점, 모의는 선, 이론은 검은 선. 단위가 다르므로 기준 수준에서 정규화한다)
  - `overlay_dose_N.png`(H2c): |OEI|와 손상 대 N. TD-MPC2는 N = 512, DINO-WM은 N = 300으로 정규화한다. 이론 $S_{op}(N)$ 비는 고정 q와 고정 k 두 가지다.
  - `overlay_iterations.png`: 손상 대 반복 수. 모의 지속형·새 잡음형과 폐루프 곡선을 함께 그린다.
  - `overlay_neff.png`(H4a): 고정 표 후보 간 분산과 손상 대 N_eff. 정확식과 1/N_eff를 함께 그린다.
  - `overlay_decomposition.png`: 칸별 sel_last와 drift, 그리고 측정 상수로 계산한 winner's curse 예측
- **E2·E3 분석이 칸에 넣어 주면 좋은 필드**(E1 칸 집계에는 아직 없는 것 포함)
  - TD-MPC2
    - `e_chosen`, `l_e_rand`, `f_e_rand`: 분해용. E1 episode 기록에는 있으나 칸 집계에 없다.
    - `l_e_rand_sd`, `l_sd_v0`, `f_e_rand_sd`: winner's curse 상수
    - 가능하면 실행 elite를 독립 dither draw로 다시 잰 오차(재평가 sel_last, 7.1절)
  - DINO-WM: `mf_dJ`, `last_pop_dJ`, `it0_rand_dJ`, `last_elite_dJ`, `spec_last`, `it0_rand_dJ_sd`
  - E3: 수준마다 고정 표의 후보 간 오차 분산 `var_cand`. P7의 A, B 추정에 쓴다.

---

## 10. CPU 시간

- 서버 CPU만 썼다(GPU 미사용, `CUDA_VISIBLE_DEVICES=""`). 작업자 8개(각 1 thread, nice 10), tmux `A-E4-sim`
- 선택 강도 표 16 s, 보정 14 s, 스모크 71 job 약 2분(CPU 845 s)
- 본 실행: 07:25:24~08:52:13Z(벽시계 1시간 27분), job 504개 CPU 합 41,636 s(11.6 CPU-시간), 계획기 실행 922,368회, 오류 0
- 분석: 분석 1회 14 s(서버, 4 thread), 세 번 돌렸다(필터·그림 수정). 겹쳐 그리기 자체 시험 수 초

---

## 11. 파일

- 코드(`main/code/theory/`, 서버 `~/pmx/A/theory/code/`와 같음)
  - `e4_formulas.py`: 예측식, 선택 강도 표
  - `e4_sim.py`: 목적함수, 오차 모형, 두 계획기
  - `e4_jobs.py`: 격자, 보정, 작업자
  - `e4_run.sh`: 실행기
  - `e4_analyze.py`: 비교 C1~C13, 표, 그림
  - `e4_report_tables.py`: 이 보고서의 표
  - `e4_overlay.py`: 겹쳐 그리기 도구
- 결과(`main/results/E4/`)
  - `calib.json`, `s_tables.json`, `manifest.json`
  - `e4_comparisons.json`, `e4_tables.md`
  - `cells_mppi.csv.gz`, `cells_cem.csv.gz`: 칸 pooling 통계
  - `fig1_singleshot.png` ~ `fig7_jensen.png`
  - `overlay_test/`: 합성 입력 `SYNTHETIC_*.json`, `overlay_*.png`
- job별 원자료(job 504개, 서버 `~/pmx/A/theory/results/E4/jobs/`, 약 176 MB)는 맥북에 옮기지 않았다.
- 작업 로그: `main/worklogs/E4_theory.md`

