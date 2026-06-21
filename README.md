# paper-summary

AI와 함께 공부한 논문 내용을 한 곳에 모아서 정리하고, 나중에 복습 자료로 활용한다.

## Recommender Systems Submission Plan

Last checked: 2026-06-21.

추천시스템 숏페이퍼/짧은 원고 제출을 목표로, 2026년 기준 주요 학회와 인접 저널 후보를 정리한다. 날짜가 확정된 항목은 공식 페이지 기준으로 적고, 아직 공식 2026 일정이 확인되지 않은 항목은 전년도 일정 기반 추정으로 표시한다.

### Immediate Opportunities

| Priority | Venue | Deadline | Best fit |
| --- | --- | --- | --- |
| 1 | [ACML 2026 Conference Track](https://www.acml-conf.org/2026/calls/papers/) | 2026-06-26 AoE | 추천 알고리즘 자체보다 ML 방법론 기여가 강하고, 16-page PMLR 포맷으로 빠르게 정리할 수 있을 때 |
| 2 | [RecSys 2026 Research and Practice Notes](https://recsys.acm.org/recsys26/call/) | 2026-07-15 AoE | 아직 완성도가 낮거나 preliminary result 중심인 추천시스템 숏페이퍼 |
| 3 | [RecSys 2026 Demo](https://recsys.acm.org/recsys26/call/) | 2026-07-15 AoE | 실제 동작하는 추천 데모, 인터랙션 UI, 라이브 시스템, 코드 라이브러리가 있을 때 |
| 4 | [ACM IUI 2027](https://iui.acm.org/2027/) | Abstract: 2026-08-13, full: 2026-08-20 AoE | human-centered, interactive, conversational, explainable recommender |
| 5 | [AAAI-27](https://aaai.org/conference/aaai/aaai-27/) | Abstract: 2026-07-21, full: 2026-07-28 | 추천 문제를 일반 AI/ML 문제로 강하게 포장할 수 있을 때 |
| 6 | [TMLR](https://jmlr.org/tmlr/) | Rolling | conference-format 원고를 저널로 돌리고 싶고, 고정 마감에 맞추기 어렵다면 |

### Major Conference Schedule

| Venue | 2026 status | Submission dates | Event dates | Notes |
| --- | --- | --- | --- | --- |
| [RecSys 2026](https://recsys.acm.org/recsys26/call/) | Main short closed; R&P/Demo open | Main short abstract: 2026-04-14, paper: 2026-04-21. R&P Notes/Demo: 2026-07-15 AoE | 2026-09-28 to 2026-10-02 | 추천시스템 원고에는 가장 직접적인 venue. R&P Notes는 2 pages plus one page for references/tables/figures. Demo는 2 pages plus up to one extra page. |
| CIKM 2026 | Official 2026 schedule not found | [CIKM 2025](https://cikm2025.org/) 기준 short abstract: 2025-05-30, short paper: 2025-06-06. 2026도 late May to early June 추정 | Not confirmed | 2025 공식 사이트의 key dates 기준 추정. 2026 일정은 재확인 필요. |
| [The Web Conference 2026](https://www2026.thewebconf.org/) | Closed | Short paper abstract: 2025-11-10, paper: 2025-11-17 | 2026-06-29 to 2026-07-03 | web-scale recommender, personalization, search/recommendation 융합에 적합. |
| [KDD 2026](https://kdd2026.kdd.org/) | Closed for short-like option | Blue Sky Ideas paper: 2026-03-12 AoE | 2026-08-09 to 2026-08-13 | 일반 research track은 short paper 성격이 아님. Blue Sky는 5-page position-style track이었으나 이미 마감. |
| SIGIR 2026 | Closed | Short paper abstract: 2026-02-05, paper: 2026-02-12 | 2026-07-20 to 2026-07-24 | IR/retrieval-heavy recommender라면 강한 후보. 공식 사이트 접근성 문제로 URL/일정은 재확인 필요. |
| WSDM 2026 | Already held | Short abstract: 2025-09-18, full: 2025-09-25 | 2026-02-22 to 2026-02-26 | 2027 submission은 2026년 9월 전후로 예상되지만 공식 CFP 확인 필요. |
| [NeurIPS 2026](https://neurips.cc/Conferences/2026/Dates) | Main closed; workshops later | Main abstract: 2026-05-04, full: 2026-05-06. Workshop suggested contribution date: 2026-08-29 | 2026-12-06 to 2026-12-12 | 추천시스템보다는 일반 ML novelty가 강해야 함. 워크샵은 논문 성격에 따라 현실적인 대안. |
| [ICML 2026](https://icml.cc/Conferences/2026/Dates) | Closed | Abstract: 2026-01-23, full: 2026-01-28 | 2026-07-06 to 2026-07-11 | 일반 ML contribution 중심일 때만 적합. |
| [AAAI-26 / AAAI-27](https://aaai.org/conference/aaai/aaai-27/) | AAAI-26 already held; AAAI-27 open soon | AAAI-27 abstract: 2026-07-21, full: 2026-07-28 | AAAI-26: 2026-01-20 to 2026-01-27. AAAI-27: 2027-02-16 to 2027-02-23 | AI 문제로 framing 가능한 추천 논문이면 고려. |
| [UMAP 2026](https://www.um.org/umap2026/) | Closed and already held | Paper due: 2026-01-29 | 2026-06-08 to 2026-06-11 | user modeling, adaptation, personalization에 매우 적합하지만 올해 일정은 종료. |
| [ACML 2026](https://www.acml-conf.org/2026/) | Conference track open very soon | Conference track: 2026-06-26 AoE. Journal track: 2026-06-20 AoE. Encore track: rolling until 2026-10-18 AoE | 2026-12-01 to 2026-12-04 | collaborative filtering, learning to rank, representation learning 등 ML 기여가 선명하면 우선 검토. |
| [ACM IUI 2027](https://iui.acm.org/2027/) | Open later this year | Workshop proposal: 2026-07-06. Paper abstract: 2026-08-13, full: 2026-08-20 AoE | 2027-02-08 to 2027-02-11 | 사용자 인터페이스, interactive recommendation, explanation, conversational agent 쪽이면 적합. |

### Journal Candidates

| Journal | Deadline pattern | Fit |
| --- | --- | --- |
| [ACM Transactions on Recommender Systems (TORS)](https://dl.acm.org/journal/tors) | Rolling | 추천시스템 전용 저널. 숏페이퍼보다는 확장된 full journal paper에 적합. |
| [ACM Transactions on Information Systems (TOIS)](https://dl.acm.org/journal/tois) | Rolling | IR/search/recommendation 평가가 강한 논문에 적합. |
| [User Modeling and User-Adapted Interaction (UMUAI)](https://link.springer.com/journal/11257) | Rolling | personalization, user modeling, adaptive systems, recommender systems에 적합. |
| [IEEE TKDE](https://www.computer.org/csdl/journal/tk) | Rolling | data mining, database, large-scale recommender system contribution이 강할 때. |
| [ACM TKDD](https://dl.acm.org/journal/tkdd) | Rolling | knowledge discovery/data mining 관점의 추천 논문에 적합. |
| [ACM TIST](https://dl.acm.org/journal/tist) | Rolling | AI system, intelligent system, applied AI contribution이 강할 때. |
| [Information Processing & Management](https://www.sciencedirect.com/journal/information-processing-and-management/about/aims-and-scope) | Rolling | information retrieval, search, personalization, user behavior 분석과 맞음. |
| [JMLR](https://www.jmlr.org/author-info.html) | Rolling | 범용 ML 방법론 기여가 강해야 하며, 단순 적용형 추천 논문은 부적합할 수 있음. |
| [TMLR](https://jmlr.org/tmlr/) | Rolling | 짧은 conference-format manuscript와 빠른 review cycle에 상대적으로 잘 맞음. |
| [JAIR](https://www.jair.org/index.php/jair/about) | Rolling | AI 전반 기여가 있는 논문. technical note/research note도 가능. |
| [Machine Learning](https://link.springer.com/journal/10994) | Rolling plus special calls | 일반 ML 저널. 2026 special call 중 Federated Learning for Critical Applications는 2026-07-15 deadline. |
| [Data Mining and Knowledge Discovery](https://link.springer.com/journal/10618) | Rolling | data mining/application 중심의 강한 실험 논문에 적합. |
| [ACM Computing Surveys (CSUR)](https://dl.acm.org/journal/csur) | Rolling | survey/tutorial 논문용. 숏페이퍼 제출 목적과는 거리가 있음. |
| [Artificial Intelligence Review](https://link.springer.com/journal/10462) | Rolling | survey/review/tutorial 논문용. |

### Recommended Preparation Order

1. Decide whether the paper is closer to an ML method paper or a recommender systems result note.
   - Strong new algorithm, generalizable representation learning, ranking objective, theory, or broad benchmark: prioritize ACML first.
   - Preliminary recommender result, new direction, partial experiment, case study, evaluation idea: prioritize RecSys R&P Notes.
   - Working system, user-facing interaction, prototype, visualization, library, or live demo: prioritize RecSys Demo.

2. If targeting ACML 2026, freeze the scope immediately.
   - Convert the draft to the ACML 16-page format.
   - Anonymize the manuscript for double-blind review.
   - Check the reviewer/area-chair nomination requirement before submission.
   - Submit by 2026-06-26 AoE only if the central claim is already experimentally supported.

3. If targeting RecSys R&P Notes or Demo, prepare a compact artifact by early July.
   - R&P Notes: write a 2-page anonymous extended abstract with one clear research question, one key result, and one future-work path.
   - Demo: prepare a 2-page description, working link if possible, and required video.
   - Submit by 2026-07-15 AoE.

4. Keep IUI 2027 as the next polished target.
   - Use July to add user-centered framing: interaction design, explainability, controllability, user study, qualitative feedback, or conversational flow.
   - Prepare abstract by 2026-08-13 and full paper by 2026-08-20 AoE.

5. Use rolling journals as fallback or expansion targets.
   - TMLR is the most natural fallback for a conference-style ML manuscript.
   - TORS/UMUAI/TOIS/IP&M/TKDD/TKDE usually require a more complete journal version, broader related work, and deeper analysis.

6. Track the next annual conference cycle.
   - WSDM 2027 is likely around September 2026 for submission, based on WSDM 2026.
   - ICLR 2027 is likely around September 2026 for submission, based on ICLR 2026.
   - CIKM 2026 needs official schedule confirmation; if it follows CIKM 2025, short paper deadlines may fall around late May to early June.
