# CVoM 저장 승인 실험 — frozen V19, fixed VAL160

## 목적과 변경 범위

기존 검색/행동 모델을 바꾸지 않고 **어떤 사건을 메모리에 남길지 학습하는 것**의 효과를 측정한다.
기존 모델이 정보를 전혀 활용하지 못하면 writer만으로 해결되지 않을 수 있다. 이 실험은 성능 향상을 보장하지 않는다.

- 고정 parent: `runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072`
- 고정 cache: `runs/long_memory/cache_full1600_v1`
- HAMLET, 저장 encoder, reader, fusion, AE 및 AE LoRA: 모두 동결.
- 온라인 feature precision: `native`. 정밀도 A/B를 이번 실험에 섞지 않는다.
- 원본 HAMLET 결과: `runs/eval/robomme/archive_read_best1250_val_n10_seed6`의 baseline 38/160을 strict provenance 검사 후 재사용.
- 저장 단위: **4개의 encoded token으로 이루어진 사건**. 토큰 하나를 프레임/물체 하나로 해석하지 않는다.
- 용량: 32개 사건. 가득 차기 전에는 모든 사건을 그대로 append.
- 가득 찬 뒤: `KEEP`(새 사건 버림) 또는 `REPLACE-OLDEST`(가장 오래된 사건 제거 후 새 사건 append).
- 이번에는 병합, 임의 victim 선택, 새 policy loss, metadata 보조 loss, 정책 재학습을 추가하지 않는다.

## 학습하는 모듈

`CVoMAdmission` 하나만 학습한다.

```text
현재 candidate / bank 평균 / oldest victim / candidate-bank 차이
+ cosine 유사도 3개 / fill 비율 1개
  → LayerNorm(1028)
  → Linear(1028, 128) → SiLU
  → Linear(128, 128) → SiLU
  ├─ Linear(128, 1): signed utility
  └─ Linear(128, 1) → sigmoid: write probability
```

stored event의 전체 token은 수정하지 않는다. 위 pooling은 writer 판단용일 뿐이다.
시간/demo 정보는 parent encoder가 이미 encoded token에 넣던 것을 유지한다.
미래 관측, GT action, subgoal, 성공 여부는 온라인 writer 입력에 들어가지 않는다.
write probability ≥ 0.5 **그리고** utility ≥ 0이면 replace한다.
두 출력 head가 0인 초기값은 tie rule에 의해 정확히 FIFO가 된다.

## CVoM teacher와 비교군

FIFO로 채운 bank의 oldest event를 `v`, 현재 새 사건을 `e`, 나머지 과거 사건의 부분집합을 `S`라 하자.
동일한 미래 관측/GT action/noise/timestep에서 다음 signed utility를 계산한다.

`gain(S) = action_loss([v] + S) - action_loss(S + [e])`

양수면 **같은 저장 예산에서 oldest 대신 candidate를 보유하는 편**이 미래 행동 예측에 유리하다.
두 branch의 사건/token 수가 같고, 각 branch 안에서 시간 순서를 유지한다.
후보 사건의 추가 이득뿐 아니라 기존 사건의 삭제 비용도 포함한다.

| arm | teacher 조건 | 온라인 정책 |
| --- | --- | --- |
| FIFO | 학습 없음 | 항상 oldest 교체 |
| single | 나머지 31개 사건을 모두 둔 조건을 독립 noise로 4회 측정 | 두 head의 승인에 따라 KEEP/REPLACE |
| coalitional | full 조건 1개 + 부분집합 조건 3개에서 측정한 gain 평균 | single과 같은 구조/판정 규칙 |

- 각 조건에서 short window 밖의 미래 action query 2개, matched noise 2개.
- single/coalitional은 같은 초기 가중치, minibatch 순서, 2,000 update, 학습률, decoder 용량을 사용한다.
- 조건당 KEEP/REPLACE 각 1회: arm별 logical AE forward 32회/context.
- 첫 full 조건의 측정은 두 arm이 공유하므로 physical forward는 합계 56회/context.
- 부분집합 크기는 non-full cardinality에서 균등 표집. 이것을 exact Shapley 추정치라고 주장하지 않는다.
- 미래 query에서 bank는 해당 저장 시점의 counterfactual snapshot으로 고정한다. 중간/미래 event는 추가하지 않는다.

따라서 이 라벨은 **고정된 teacher와 bank에서의 조건부 미래 action-flow utility**이다.
미래 simulator 성공률, terminal reward, 실제 FIFO continuation의 수익이 아니다.
사용하는 loss는 기존 prefix-weighted flow objective(tail weight 0.25)이며 별도 policy loss를 도입하지 않는다.
teacher의 저장 판단 context는 FIFO history에서 만든다. 학습된 writer가 사건을 거부하기 시작하면
온라인 bank 분포는 이 학습 context와 달라질 수 있다. 그 차이를 offline validation만으로 해결했다고 하지 않는다.

writer의 학습 loss는 `SmoothL1(normalized utility) + 0.25 × BCE(write label)`이다.
TRAIN에서 구한 공통 robust scale을 두 arm에 사용한다.
`abs(gain) > max(1e-6, 1.96 × conditional noise SE)`를 만족하지 못하는 모호한 라벨은 두 loss에서 제외한다.
이 기준은 적은 noise 표본에 대한 휴리스틱이지 episode-level 95% CI가 아니다.
어느 arm이든 confident TRAIN label이 하나도 없으면 임의 학습 대신 중단한다.
두 arm의 ambiguous mask가 달라 실제 loss에 참여하는 라벨 수는 다를 수 있다. `label_audit.json`에 별도 기록한다.
noise guard는 coalition 간 변동을 반영한 신뢰구간이 아니므로 coalition 변동도 원시 라벨에 함께 보존한다.
기존 BF16 cache 특징과 native 온라인 특징 사이의 차이를 이번 writer 구현이 해결했다고 주장하지 않는다.

## 데이터와 사전 고정 평가

기존 TRAIN 1,276 / cache-VAL 324 episode split을 유지한다. TEST는 사용하지 않는다.
성공률이나 teacher gain을 보기 전에 candidate/query를 seed `192109`로 선택한다.
기본값은 **조건에 맞는 모든 TRAIN episode에서 후보 저장 시점 1개**이다.

2026-09-21 실제 preflight:

- TRAIN: 280개 episode/context. FIFO32가 이미 찼고, short window 밖에 미래 action query가 최소 2개 남는 episode 전부.
- cache-VAL: 64개 episode/context. TRAIN과 episode ID 중복 없음.
- 조건을 만족하지 않는 짧은 episode는 정책을 달리할 overflow가 없거나 teacher 미래 query가 부족하여 학습 대상이 아니다.
- 이것은 전체 policy/모든 frame 재학습이 아니다. 캐시를 반복 사용하는 작은 writer 학습이다.
- 종료/체크포인트 선택: 미리 정한 2,000 update의 마지막 checkpoint. VAL 최고 step 탐색 안 함.

온라인 평가는 이전과 동일한 16 tasks × 10 scenarios = **160 episodes/arm**, dataset `val`, inference seed `6`.
action interval 16, max episode steps 1300, 기존 denoising/session/read-before-write를 유지한다.
FIFO, single, coalitional을 각각 평가하므로 전체 **480개 신규 rollout**이다.
기존 V19 37/160은 참고 기록이며, 이번 source/runtime의 FIFO를 다시 실행해 세 arm을 직접 비교한다.
원본 HAMLET 38/160은 새로 실행하지 않는다.

결과를 보고 subset, threshold, checkpoint, 종료 조건을 바꾸지 않는다.
개발용 VAL160을 반복 사용했으므로 최종 일반화 성능의 확증으로 주장하지 않는다.
평가 중에는 연결된 policy/server/evaluator/core 파일을 수정하지 않는다.

## 실행 순서

아직 시작하지 않은 새 run이라면 한 번에:

```bash
cd /home/sjkim/HAMLET-Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 bash run_scripts/robomme/run_cvom_admission.sh all
```

분리 실행:

```bash
bash run_scripts/robomme/run_cvom_admission.sh preflight
CUDA_VISIBLE_DEVICES=0 bash run_scripts/robomme/run_cvom_admission.sh prepare
bash run_scripts/robomme/run_cvom_admission.sh train
CUDA_VISIBLE_DEVICES=0 bash run_scripts/robomme/run_cvom_admission.sh verify
bash run_scripts/robomme/run_cvom_admission.sh eval-preflight
CUDA_VISIBLE_DEVICES=0 bash run_scripts/robomme/run_cvom_admission.sh eval
bash run_scripts/robomme/run_cvom_admission.sh compare
```

기본 출력:

- 학습: `runs/long_memory/cvom_admission_full_v1`
- 평가: `runs/eval/robomme/cvom_admission_full_v1_val160_seed6`
- 최종 3-arm 비교: 평가 경로 아래 `summary/comparison.txt`, `comparison.json`, `tasks.csv`
- 개별 arm vs 원본 HAMLET: `single/comparison_summary.txt`, `coalitional/comparison_summary.txt`
- 실시간 학습 기록: 학습 경로 아래 각 arm의 `metrics.jsonl`; 정적 그래프도 저장.

실시간 TensorBoard:

```bash
bash run_scripts/robomme/run_cvom_admission.sh monitor
```

기본 port는 6007이다. `single/train`, `single/val`, `coalitional/train`, `coalitional/val`을 선택한다.
write accuracy는 **teacher의 저장 판단 라벨을 맞힌 비율**이지 로봇 성공률이 아니다.
`conditional_gain_vs_fifo` 역시 offline teacher 지표이며 실제 성공률은 rollout summary로 확인한다.

재실행 규칙:

- prepare는 **같은 source/data/parent/protocol**인 경우 완료된 context를 검증하고 이어서 계산한다.
- train은 기존 arm 디렉터리를 덮어쓰지 않는다. 부분 writer 학습 재개는 이번 CLI에서 지원하지 않는다.
- 완료된 학습은 전체 wrapper에서 체크포인트 검증 후 재사용한다.
- eval은 기존 manifest와 source가 일치하는 완료 task만 재사용한다. 서로 다른 실험은 새 출력 경로를 사용한다.
- `CVOM_RUN_DIR`, `CVOM_EVAL_DIR`로 새 실험 경로를 명시할 수 있다.
- 동시에 같은 wrapper를 실행하면 lock으로 중복 실행을 거부한다.
- GPU teacher 준비/회귀 검사 전에 사용 현황을 확인하며 다른 작업을 강제 종료하지 않는다.

## 코드 구성

| 파일 | 책임 |
| --- | --- |
| `run_scripts/robomme/cvom_admission_core.py` | causal features, MLP, 결정/메모리 갱신 |
| `run_scripts/robomme/cvom_admission_teacher.py` | 사전 context 선택, matched counterfactual coalition labels |
| `run_scripts/robomme/cvom_admission_checkpoint.py` | frozen parent와 writer sidecar 무결성/저장/로딩 |
| `run_scripts/robomme/train_cvom_admission.py` | preflight, GPU teacher label 준비, CPU writer 학습, metrics |
| `run_scripts/robomme/verify_cvom_admission.py` | 고정 관측 replay, 실제 AE, WRITE/RNG/READ-off 회귀 |
| `run_scripts/robomme/eval_cvom_admission.py` | 기존 evaluator의 fixed VAL160 wrapper |
| `run_scripts/robomme/summarize_cvom_admission.py` | 엄격 provenance 검증, 세 arm의 episode-paired CI/반전 집계 |
| `run_scripts/robomme/run_cvom_admission.sh` | 단계별/일괄 실행 |

기존 `policy_representation_v18.py`, `serve_representation_v18.py`, `eval_representation_v18.py`에는
명시적인 `--cvom-admission` 선택 경로를 추가했다. 기본 동작은 바꾸지 않는다.
원래 checkpoint/cache/results는 수정하지 않고 writer만 별도 sidecar로 저장한다.

## 실행 검증 기록 (2026-09-21)

- 관련 회귀/unit test **76개 통과**. matplotlib 의존성 deprecation warning 14개, 실패 없음.
- 실제 parent/cache로 smoke labels TRAIN 8 / VAL 4 생성, writer 두 arm 각각 20 update 및 별도 checkpoint 저장 완료.
- smoke의 confident VAL label이 0개인 것은 기록하되 성공 주장/threshold 변경 근거로 사용하지 않았다.
- 실제 native VLM/AE 고정 관측 replay: episode 1355의 72 endpoints와 626의 6 endpoints를 ON/OFF 각각 검사.
  총 156 endpoint 처리, 실제 AE 생성 22회. bank, native short, WRITE 결정, RNG, reset, read-before-write 검증 통과.
- teacher, reader, writer의 해당 검증 전후 state hash 동일.
- simulator/policy 환경 및 원본 HAMLET reference strict preflight 통과.
- 상세 smoke 결과: `runs/diagnostics/cvom_admission_smoke_20260921/completed.json`.

이 회귀 검증은 **실제 simulator closed-loop 성공률 측정이 아니다**.
학습과 rollout 완료 여부는 각 run의 status/manifest/summary를 확인해야 한다.

### 본 실행 시작 기록

2026-09-21, 다음 본 실행을 실제로 수행했다. 과거 결과/가중치는 덮어쓰지 않았다.

- label 준비: TRAIN 280 + VAL 64 = 344 contexts 완료, 약 306초(라벨 계산 구간).
- 준비 전후 frozen core/head state hash 동일.
- single/coalitional 모두 2,000 update 완료, 각각 `checkpoint-002000` 저장.
- confident TRAIN label: single 45/280(positive 24, negative 21), coalitional 61/280(positive 29, negative 32).
- confident VAL label: single 12/64, coalitional 21/64.
- final VAL write-label accuracy: single 50%, coalitional 약 52.38%. **로봇 성공률이 아니며 강한 일반화 증거도 아니다.**
- 전체 64개 offline VAL context에서 final replace 비율: single 43.75%, coalitional 40.625%.
- 본 coalitional checkpoint로 native 실제 AE 회귀 재실행: 156 endpoints, 22 AE 생성, PASS.
- 세 arm의 실제 simulator 평가를 시작했다. 이 문서 작성 시점에는 진행 중이며 최종 성공률은 아직 없다.

실제 실행 명령:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash run_scripts/robomme/run_cvom_admission.sh prepare

# 위 label 준비 완료 후, 분리된 세션으로 나머지 단계를 연속 실행했다.
setsid --fork nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash run_scripts/robomme/run_cvom_admission.sh all \
  >> runs/long_memory/cvom_admission_full_v1_launch.log 2>&1 < /dev/null
```

시작 당시 pipeline PID는 `256851`이었다. 실행 중이면 중복 실행하지 않는다.
통합 로그:

```bash
tail -f runs/long_memory/cvom_admission_full_v1_launch.log
```

학습 근거: `label_audit.json`, `training_summary.json`, `runtime_verification/completed.json`.
평가가 모두 끝나면 위의 `summary/comparison.txt`가 생성된다.

## 결과 해석

1. 동일 frozen actor의 learned writer vs FIFO: 학습한 저장 승인이 실제 행동에 도움이 되는가?
2. coalitional vs single: 같은 학습/teacher 계산 예산에서 여러 memory 조합을 고려한 라벨이 도움이 되는가?
3. 원본 HAMLET vs 세 arm: 기존 단기 모델 대비 최종 시스템의 이득인가?

task별 성공 수, 같은 scenario에서의 win/loss/same, episode-paired bootstrap CI를 함께 보고한다.
CI가 0을 포함하면 작은 표본에서 방향이 불확실하다는 뜻이지 효과가 없다는 증명이 아니다.
writer가 전부 KEEP하거나 전부 REPLACE하면 runtime count와 라벨 분포에 그대로 드러내며,
성능을 보고 threshold/평가 패널을 바꿔 유리한 결과만 보고하지 않는다.
