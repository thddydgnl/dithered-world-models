# 참고문헌 확인 기록 (RSS 2027 Stage 1 초록)

- RSS 2027은 환각 인용이 있으면 desk reject한다. 인용 여부는 자동으로 검사한다.
- 그래서 `refs.bib`의 40개 항목을 모두 원 출처와 대조했다. 대조 시각은 2026-11-01(KST)이다.
- 표기
  - **A**: 조정자가 직접 확인했다.
  - **R**: 관련 연구 재점검 작업자가 abs 페이지나 OpenReview API 기록을 다시 열어 확인했다. 기록은 `rss_ext/related_2026.bib`의 항목별 `% verified` 주석에 있다.
  - **K**: 표준 문헌이다. 서지는 출판사 기록과 맞는 것으로 알고 있다.
- 남은 일: OpenReview 7편은 조정자가 다시 확인하지 못했다. OpenReview API가 봇 확인(challenge)을 요구하기 때문이다. 제출 전에 브라우저로 열어 제목을 한 번 더 확인한다.

| 키 | 출처 | 확인 |
|---|---|---|
| hansen2024tdmpc2 | ICLR 2024 | K |
| zhou2025dinowm | arXiv:2411.04983 | K(JKAIA 원고에서도 같은 서지) |
| williams2017mppi | ICRA 2017, 1714–1721 | K |
| rubinstein1999cem | MCAP 1(2):127–190 | K |
| smith2006curse | Management Science 52(3):311–322 | K |
| kleinman1999crn | Management Science 45(11):1570–1578, DOI 10.1287/mnsc.45.11.1570 | A(RePEc/INFORMS 기록, 11-01) |
| kleywegt2002saa | SIAM J. Optim. 12(2):479–502 | K |
| gupta2015limited | ICML 2015, 1737–1746 | K |
| croci2022sr | R. Soc. Open Sci. 9(3):211631 | K |
| roberts1962picture | IRE Trans. Inf. Theory 8(2):145–154 | K |
| schuchman1964dither | IEEE Trans. Commun. Technol. 12(4):162–165 | A(인용 기록, 11-01) |
| ha2018worldmodels | NeurIPS 2018 | K |
| boney2019dae | NeurIPS 2019 | K |
| yu2019metaworld | CoRL 2019 | K |
| tassa2018dmc | arXiv:1801.00690 | K |
| rouhani2023mx | arXiv:2310.10537 | K |
| nvidia2025nvfp4 | arXiv:2509.25149 | A(abs 페이지, 11-01: 제목, 첫 저자들, v1 2025-09-29) |
| min2026lowerror | arXiv:2610.05550 | R, A(abs 페이지: 제목, 저자 4명, v1 2026-10-04 일치) |
| fu2026wmquant | arXiv:2602.02110 | R |
| xu2026prede | arXiv:2609.19441 | R |
| you2026predictability | arXiv:2607.10362 | R |
| wang2026drpe | arXiv:2609.32322v2 | R |
| obst2026cem | arXiv:2610.00921 | R |
| vlaquantbench2026 | arXiv:2609.25376 | R |
| qvla2026 | ICLR 2026(arXiv:2602.03782, OpenReview TpL2nXanru) | R |
| khilar2026dither | arXiv:2609.39185 | R, A(abs 페이지: 제목, 단독 저자, v1 2026-09-30 일치) |
| nvidia2026nemotron | arXiv:2604.12374 | R |
| chatelain2026sr | arXiv:2610.01889v2 | R(둘째 저자 성은 "de Oliveira Castro"로 표기) |
| wu2026decisions | arXiv:2608.06564v5 | R |
| wang2026steerquant | arXiv:2609.39056 | R |
| park2026qcrl | arXiv:2610.15675 | R |
| ranganath2026bits | arXiv:2602.11882 | R |
| kuzmenko2025tdmpcopt | arXiv:2507.01823 | R |
| anon2026modulemse | ICLR 2027 투고, OpenReview 3VSa1Kx7HD | R(조정자 재확인은 challenge로 막힘) |
| anon2026elitecert | ICLR 2027 투고, OpenReview cVCKr4bBU2 | R(같음) |
| anon2026resolution | ICLR 2027 투고, OpenReview eTu4mzKor3 | R(같음) |
| anon2026evaluations | ICLR 2027 투고, OpenReview 1vmM17t1R8 | R(같음) |
| anon2026magnitude | ICLR 2027 투고, OpenReview 5alhlhb7zd | R(같음) |
| anon2026loopquant | ICLR 2027 투고, OpenReview AhG6FcJBHt | R(같음) |
| anon2026wamptq | ICLR 2027 투고, OpenReview hdBfu8lIC7 | R(같음) |

- 본문에서 이 문헌들을 서술한 문장은 초록 수준의 정보에 기대고 있다. OpenReview 7편은 본문 PDF를 읽지 못했다.
  - 위협이 큰 두 편은 사용자가 본문을 읽고 서술이 맞는지 확인한다. 두 편은 3VSa1Kx7HD("보정 목표로 4-bit 가중치에서 FP 수준 계획")와 cVCKr4bBU2("CEM 반복에 따라 인증된 elite 판정 감소")다.
- ICLR 2027 토론 기간에 이 투고들의 수정본이나 arXiv판이 나올 수 있다. 제출 직전(12월 초)에 다시 확인한다.
