## [사후] X17: PushT 4-bit E1 분할(eval6) 재실행, RTX 4090 (D5)

| 변형 | 성공 | D %p [CI] | red [CI] | E1 성공 | E1과 episode 일치 |
|---|---|---|---|---|---|
| none | 83% | — | — | 83% | 98% |
| det | 62% | +21 [12, 30] | — | 59% | 95% |
| sh-iter | 83% | +0 [-4, 4] | 1.00 [0.80, 1.23] | 83% | 96% |

- 판정(D5, 미리 적은 기술): **(i)** split difference (eval8) explains the non-replication. D_det +21 [12, 30] (E1 +24 [14, 34]); 한쪽 p 9.999e-05
- E1 none과 비교: 이 pod 83/100, E1 83/100, episode 일치 98%, McNemar p 1
