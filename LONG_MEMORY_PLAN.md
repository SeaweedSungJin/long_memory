# 현재 장기기억 연구 계획

## 완료한 V9 목적함수 대조 실험 (2026-09-16)

[V9 실제 생성 action 보조 학습](docs/LONG_MEMORY_V9_DEPLOYMENT_OBJECTIVE_PLAN.md)을
구현하고 CPU / 실제 GPU gradient·checkpoint·재개 검증을 완료했다. 같은 V7 archive 구조에서
flow-only continuation과 flow + generated-prefix loss를 동일 조건의128updates로 비교 완료했다.
둘 다 초기 모델보다 검증 action 오차가 높아 이 설정의 장기 학습은 보류한다. 새 memory
architecture나 learned admission writer를 추가한 실험은 아니다. 고정 step128의 실제
validation rollout까지 완료했으나 초기 archive1250 대비 성공률 개선은 확인하지 못했다.

동시에 [같은 checkpoint의 READ on/off 대조](docs/LONG_MEMORY_ARCHIVE_READ_CONTROL_V7.md)를
완료했다. 원본160episodes 38성공 / archive160episodes 41성공은 기존 평가의 모든 CSV 필드와
일치했다. 같은 AE의 READ-off는34/160이었다. READ-on 효과는+4.375pp이나 McNemar
p=0.0654이며 원본 대비 우위는 입증하지 못했다. 아직 **test30%를 달성한 모델은 없다.**
Loss와 실제 성공률의 순서가 달라, V9 두 fixed-step128 모델도 all16 val10에서
별도로 평가했고 두 run 모두 baseline / READ-on / 같은 AE READ-off 각각160개,
480/480행을 완료했다(exit0, 실패·중단 없음). 성공 수는 flow38/38/37,
auxiliary38/39/34다. Baseline CSV는 기존16개 파일과 byte-for-byte 동일하고, 모든 paired
episode/scenario seed·instruction 및 context가 일치했다. 각 run의 source141개,
benchmark70개, base12개, checkpoint4개 파일 해시도 변경 없이 검증됐다.

READ-on의 원본 대비 변화는 flow0.000pp (95% CI [−4.375,+4.375], p=1),
auxiliary+0.625pp ([−3.125,+4.375], p=1)다. 같은 AE READ-off 대비는 각각
+0.625pp ([−3.750,+5.000], wins/losses7/6, p=1),
+3.125pp ([−0.625,+6.875], 8/3, p=0.226562)다. 초기 archive1250의41성공 대비는
flow−1.875pp ([−6.250,+2.500], 5/8, p=0.581055),
auxiliary−1.250pp ([−5.625,+2.500], 5/7, p=0.774414)였다.
CI는 같은160episodes의 within-task paired bootstrap이며 모든 구간이0을 포함한다.
APPEND/READ 진단 누락이나 reset 이상은 없고, 두 READ-off의 conditioning 변화는 정확히0이었다.
따라서 개선도 확실한 악화도 입증하지 못했으며 **현재 V9 장기 continuation을 확대할 근거는
없다.** 초기 archive1250을 보존한다. 이 결과는 validation이며 test30% 달성 주장이 아니다.

[V10 출력층 적응 대조](docs/LONG_MEMORY_V10_PROJECTOR_PILOT.md)는 별도 파일에 구현하고
학습/저장/추론/평가 CPU 73 tests 및 실제 base/cache read-only preflight를 통과했다.
같은 archive/flow/LoRA에서 마지막 AE Linear의 zero-initialized delta만 추가 학습하는 후보이며
V10 실제 GPU 2-step pause → 4-step resume 스모크도 통과했다. Memory/LoRA/projector가
finite gradient로 업데이트되고 원본/CVOM은 보존됐다. **V10 최종 paired 평가도 완료했다.**
동일 조건 128-update 대조 학습을 완료했으나 두 arm 모두 proxy best는 초기step0이다.
고정step128 두 모델의 실제 simulator smoke를 각각6/6 완료했고 all16 val10 READ-on/off
평가를 완료했다. READ-on160은 각각 대조군40성공(25.00%)/출력층 후보37성공(23.125%)이며
동일 AE READ-off는 두 arm 모두38성공(23.75%)이다. READ 효과의95% CI는 각각
[−2.50,+5.00]pp/[−3.75,+2.50]pp로0을 포함한다. 출력층 후보 확대를 채택하지 않으며
기존 코드/checkpoint를 유지한다. 두 run 모두 실패·중단·진단 누락이 없다.
실제 BF16 Action Expert의 CPU zero-delta 수치/gradient 검사도 통과했다. 다음 평가의 원본
결과 재사용은 provenance를 검증하고 `REUSED`로 구분한다. 과거 image의 공간 정보 보존은
[별도 조건부 후보](docs/LONG_MEMORY_VISUAL_PATCH_CANDIDATE.md)에 core/replay/학습/저장/
추론/평가를 새 파일로 구현하고 CPU70 tests를 통과했다. 실제 고정 AE에서 저장 encoder와
Q/K/V까지 gradient를 확인했다. 첫 V11 pilot은 archive1250/AE를 고정하고 새 visual 모듈만
512updates 학습을 완료했다. 실제 GPU pause/resume·최장prefix·6episode 추론스모크를 통과했고,
고정VAL prefix MSE기준best384가초기대비4.19% 낮다(관절MSE는11.91% 악화).
`v11_visual_best384_val_n10_seed6`에서 고정모델의 all16 val10 READ-on/off 평가를 완료했다.
Visual-on은160회를 완료해29성공(18.125%)이며 원본38성공(23.75%)보다 낮다.
동일 frozen archive/AE의 visual-off는41성공(25.625%)이다. On−off는−7.50pp,
95% paired bootstrap CI [−11.875,−3.125]pp, p=0.00753784다. 원driver SIGTERM후
같은identity로미완료부분만재개했으며최종exit0이다. 고정32-query proxy 개선도
단일query에 크게 의존했고, 선택32개를 제외한 추가 VAL292개 진단도 완료했다.
추가292개 중82개 개선/200개 악화/10개 동일이며, 평균MSE 개선은 한query를 제외하면
악화로 바뀐다. 전체MAE는3.93% 악화했다. Flow loss 감소를 성공률 개선으로 해석하지 않는다.
이 후보를 성능 개선 모델로 채택하지 않으며, 최종test30% 달성 결과는 없다.

다음 [V12 differential READ](docs/LONG_MEMORY_V12_DIFFERENTIAL_READ.md)는 새 파일로 구현했다.
V11의 patch-common residual 중앙 에너지94.4% 진단을 근거로, content-only value의
과거 READ에서 현재 self-READ를 빼는 후보와 독립 학습 current-only adapter를 비교한다.
기존 archive/AE는 고정하고, 두 arm의 parameter·초기값·query/noise 예산을 맞춘다.
선택VAL을128개로 늘리고 generated-prefix MAE를 사전 선택 지표로 정한다.
Core/replay/training/checkpoint/policy/evaluator/verifier CPU95 tests와 실제BF16 AE의
두arm별2-update GPU wiring, 2→4-step 저장/재개, 총8episode 추론 smoke를 통과했다.
동일초기값·zero-init원래Euler4일치·과거/현재gradient·원본가중치불변을확인했다.
Differential512 학습은 완료했고 고정VAL128의MAE는6.02% 감소했으나, 개선이 두query와
gripper에 집중돼 일반화나 task성공률 향상 근거로 충분하지 않다. 같은512예산 current-only
대조 학습도 완료했고 best384의MAE는0.353% 감소했다. 각MAE-best512/384를 고정하여
추가196 cache-query 진단 및 all16×10 VAL READ-on/off/current-only rollout을 진행한다.
V12 differential의별도VAL160은31성공(19.375%)으로원본38성공(23.75%)보다낮았다.
READ-off/current-only 전체평가도 완료했다. 성공 수는 baseline38/differential31/current-only31/
visual-off41 (각160 VAL)이다. Differential의 off 대비 −6.25pp 구간[−10.625,−1.875]pp,
current-only 대비0pp 구간[−4.375,+4.375]pp이며 V12를 개선후보로 채택하지 않는다.
결과는 `runs/eval/robomme/v12_paired_best_val_n10_seed6/`에 paired report로 보존한다.
추가196cache-query에서도parent대비MAE개선0.05%의구간이0을포함했다.
최종TEST30% 달성 결과는 아직 없다.

[V13 demo-tail 입력 후보](docs/LONG_MEMORY_V13_DEMO_TAIL.md)는 별도파일의입력/CPU검증
도구를구현하고 CPU49개 검사/실제metadata preflight를 통과했다. 실제로빠지는시연마지막15frames를visualbank에만추가하고기존HAMLET/
parent/action/RNG는보존하는계획이다. 새로운관측의효과와검색구조의효과를구분한다.
실제12-frame native/direct/cached GPU 특징은 exact 검증을 통과했고,
9 TRAIN episodes의135frames proof 후 전체1,600episodes/13,500frames를
`demo_tail_full1600_v13_20260917`에 추출 완료했다. 기존 cache와 모델은 변경하지 않았다.
Batch/online 수치 차이를 확인해 새 V13 replay만 framewise로 통일했고,
실제 frozen AE에서 학습 전/후5경우의 기억값·native Euler4 행동 일치 검사를 통과했다.
Distinct V13 checkpoint/학습/RGB-ingest/policy/server/client를 구현했고
재개 및 live 경로를 CPU 검사했다. **두 arm 모두4/512회 저장용 smoke를 완료했고,
실제 RGB/session144개 검사에서 parent 상태·행동·난수 보존을 확인했다.
새 폴더로 각각512회 재개/최종 저장을 완료했다. Tail의 offline MAE는parent보다1.18% 나쁘고,
canonical의6.11% 개선은 두query에 집중돼 있다. 실제 VAL160의 tail은32성공(20.0%),
canonical은35성공(21.875%)로 원본38성공(23.75%)보다 낮았다. 실제 off160은41성공(25.625%)이다.
전체 split report가 COMPLETE이며 tail의 off 대비 −5.625pp 구간[−10.0,−1.25]pp다.**
Raw extra preprocessing은 global RNG를 소비하므로 online 경로는 반드시 난수 상태를
보존해야 한다. 단순한 입력/캐시 검증을 저장·검색 성공률 개선으로 보고하지 않는다.

현재 [V14 공간 기억 + AE 공동학습](docs/LONG_MEMORY_V14_VISUAL_EXPERT.md)을 NEW 파일로 구현했다.
동일 V13 framewise/tail/query512 일정에서 visual14와 기존 AE LoRA256 tensors만 함께 학습한다.
Visual LR1e-4 / expert LR1e-5, 원본 base/archive/CVOM 고정이며 admission은 APPEND 규칙이다.
CPU30개 검사와 실제 데이터 read-only preflight 및 GPU0의4-update 검사를 통과했다.
270개 tensor 업데이트/원본 보존/초기값 동일성을 확인하고 NEW 폴더로512까지 재개 완료했다.
최종 on MAE0.0144397/off0.0142809/초기0.0142281로 offline 개선은 확인되지 않았다.
V14의 off 대조는 새 adapted expert를 유지하며, 실제6episode 실행 검사와 전체 VAL을 완료했다.
ON33/160(20.625%),OFF37/160(23.125%),원본38/160(23.75%)이다. OFF→ON은−2.5pp,
95% CI[−6.875,+1.875]pp로 개선을 입증하지 못했다. 이 후보를 개선 모델로 채택하지 않는다.
원본/체크포인트/전체source/실제READ·RPC/episode identity 감사 및 최종 paired report가 모두
완료됐으며 두 GPU를 반환했다. 결과는 `runs/eval/robomme/v14_paired_fixed512_val_n10_seed6/`다.
기존 archive1250과 모든 실험은 보존했다. 최종TEST30% 달성 결과는 아직 없다.
[입력/검색 후속 감사](docs/LONG_MEMORY_INPUT_RETRIEVAL_AUDIT_20260917.md)에서 language
정답 누출 가설을 배제하고, weak evidence 대응 및 강한 demo telemetry 조건의 제한을 기록했다.

다음 [V15 과거 구간 검색 학습 검사](docs/LONG_MEMORY_V15_SEGMENT_RETRIEVAL.md)는
관측 입력을 유지한 채 PatternLock의 약한 demo/execution 구간 대응을 loss에만 사용한다.
기존 visual Q/K Linear 두 개만 별도 복사본에서 학습하고, 시간 기준 및 content 이동 대조로
검색 신호를 검사했다. TRAIN53/cache-VAL16 episodes,48 CPU tests 및4-update GPU smoke를
통과했고 고정256updates도 정상 완료했다. 정상 cache-VAL loss2.301→1.260은 개선됐으나
time-only0.734보다 나쁘고, 내용 이동 조건에서 uniform-demo 우위를 입증하지 못했다.
사전 판정은 **no-go**이며 이 probe를 로봇 정책에 연결하지 않는다.
다음 [V16 대조](docs/LONG_MEMORY_V16_PROJECTION_PROBE.md)에서는 Q/K-only와 image projection
한tensor를 함께 학습하는 경우를 동일 예산으로 비교한다. 공유 P는 현재 이미지·short query도
바꾸므로 과거 storage만의 효과로 해석하지 않는다. 아직 새 로봇 정책이나 성공률 개선 결과가 아니며, 실제 action 학습과
전체 paired rollout이 남는다. 기존 V7–V15 모델 및 결과는 변경하지 않는다.
V16은79 CPU tests, 양 arm의 real preflight/4-update GPU smoke를 통과했고 실제 TRAIN query에서
current/past 양쪽 P gradient도 확인했다. 같은 초기값·일정의 fixed256 대조도 정상 완료했다.
QKP의 정상loss0.957/이동내용loss1.545는 QK1.260/1.903보다 좋고, primary 내용 이동 개선
+0.357926 nats의95% episode CI[0.047308,0.601779]는0을 제외한다.
그러나 정상 time-only 및 이동내용 uniform-demo 대비 우위를 입증하지 못해 최종판정은
**no-go**다. 표현 학습의 탐색적 개선이지 robot success가 아니며 실제 정책으로 승격하지 않는다.
보고서는 `runs/long_memory/v16_projection_pair256_20260917/`에 저장했다.
Learned admission/CVOM을 추가한 실험이 아니며 새 policy 결과로 보고하지 않는다.

## 완료한 V8 실험 (2026-09-16)

[V8 사건별 저장/검색 설계](docs/LONG_MEMORY_V8_EVENT_PLAN.md)를 구현하고,
전체 데이터 3-epoch 학습과 예정된 RoboMME val 비교를 완료했다.
V7에서 관찰한 슬롯 동질화를 피하기 위해 사건별 moment 표현을 독립적으로 저장하고,
저장 encoder·cross-attention 검색기·fusion·AE LoRA를 함께 학습한다.
128-event FIFO의 저장/삭제 결정은 규칙이며 CVOM 학습과 구분한다.

- [학습·실행 기록·실시간 그래프](docs/LONG_MEMORY_V8_TRAINING.md)
- [RoboMME 평가와 대조군](docs/LONG_MEMORY_V8_EVALUATION.md)
- [완료된 학습 및 실제 평가 진행 기록](docs/LONG_MEMORY_V8_PROGRESS_20260916.md)
- Val 16×10: 원본 23.75%, archive 최고 25.625%, V8 최고 23.125%.
  개선을 확정할 증거는 없으며 [같은 archive의 READ-off 대조](docs/LONG_MEMORY_ARCHIVE_READ_CONTROL_V7.md)를 완료했다.
- 30%는 목표이며 아직 달성한 결과가 아니다. V7 이하 코드/체크포인트는 보존한다.

## 이전 계획: V7 (기록 보존)

2026-09-15 기준 다음 구현 명세는
[V7 recurrent memory 설계 문서](docs/LONG_MEMORY_V7_RECURRENT_DESIGN.md)에 정리했다.

- HAMLET short tokens → shared cross-attention READ/WRITE → 고정 크기 recurrent memory.
- Stage 1: 미래 action loss로 WRITE/READ/gates/fusion/AE LoRA 공동학습.
- Stage 2: KEEP/UPDATE의 미래 기여도를 학습하는 저장 CVOM.
- 모듈 형태·차원·수식·sampling·gradient·평가·보존 규칙은 위 문서를 기준으로 한다.
- [학습 실행·실시간 그래프](docs/LONG_MEMORY_V7_TRAINING.md), [RoboMME 평가](docs/LONG_MEMORY_V7_EVALUATION.md).
- 같은-short archive/AE-only 대조군 및 인접 query pairing을 검토 후 추가했다.
- **V7 코드 구현 및 짧은 실행 검증. 전체 학습·새 성공률 개선 결과는 아직 없음.**

원래 `ideas.md`, backup의 `goal.md`, v1–v6 문서/코드/체크포인트/결과는 보존한다.
구현하면서 설계가 바뀌면 위 문서의 변경 기록과 실험 run에 함께 남긴다.
