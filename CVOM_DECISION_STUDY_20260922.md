# CVoM 저장 결정 비교: noise, admission, eviction, min-fill

> 후속 실행 완료: 사용자의 추가 요청으로 min_fill32 VAL160을 실제 실행했다.
> 36/160 → 31/160이며, 상세 결과는 [후속 rollout 보고서](CVOM_MIN_FILL32_RESULT_20260922.md)에 있다.
> 아래의 "아직 실행하지 않았다"는 이 문서를 최초 작성했을 때의 상태를 보존한 기록이다.

## 이번 변경과 실행 범위

HEAD `9a63ad64a8a4e4772e755cf423c8529b10165e30`에서 시작했다.
기존 학습/추론/probe 파일과 모델·캐시는 변경하지 않고 새 진단 및 명시적 runtime wrapper만 추가했다.
`all` 재학습은 하지 않았다. 이번 실제 GPU 실행은 noise8 offline probe와 native min-fill 회귀 검사다.
**새 min_fill32의 VAL160 rollout은 아직 실행하지 않았다.**

| 요청 | 코드/결과 |
|---|---|
| 같은 teacher에서 noise 2→8 | 동일 16 episodes·context·미래 query로 실제 완료. 공통 draw와 생성 결과의 정확한 재현 검증 |
| bank 점유·포화 전 거부 | 기존 FIFO/learned 320개 완료 rollout session의 per-call telemetry 감사 완료 |
| writer 지표 분리 | 후보 admission / 기존 슬롯 eviction ranking / 선택한 결정의 heldout 이득을 별도 집계 |
| min_fill=32 | 명시적 추론 wrapper, native 실제 AE 회귀, strict fixed VAL160 evaluator와 preflight 완료 |
| teacher 확인 후 writer→reader 적응 | 현재 evidence gate 미통과. 학습 시작하지 않으며, 실제 후속 학습 루프 확장은 보류 |

## 1. Noise 2→8 실제 결과

Checkpoint: `runs/long_memory/echo_cvom_full_v1/refresh/checkpoint-001000`.
Cache: `runs/long_memory/cache_full1600_v1`.
VAL **16 episodes / 6 eligible tasks**, episode당 full FIFO32 context 하나.
이는 simulator의 16 tasks×10=VAL160과 다른 **offline cache-VAL 진단**이다.

변경한 것은 noise 반복 수 2→8뿐이다. 다음은 정확히 같았다:

- checkpoint 및 in-memory actor hash, cache 및 선택된 파일 signature, source, 환경.
- episode/context, selection·heldout query, GT 전체 50-step chunk 비중첩 간격.
- 33개 same-budget operation, frozen manager 결정/점수, bank provenance.
- 포함 관계에 있는 공통 flow draw의 5종 오차와 모든 generated action 오차.

기존 noise2 결과는 수정/재실행하지 않고 재사용했다.
noise8은 flow 25,360회, 4-step Euler action generation 1,072회 실행했다.
진단 계산·종료 감사 약 437초, 로딩/사전 검사 시간은 별도다.

| 지표 | Noise 2 | Noise 8 |
|---|---:|---:|
| 독립 noise panel에서 동일 최선 선택 | 2/16 | 2/16 |
| 전체 operation-pair 순위 일치 | 50.52% | 52.28% |
| 위 순위 95% episode-bootstrap CI | [47.19,53.82]% | [48.03,56.75]% |
| 선택 A의 heldout generated MSE gain vs FIFO | +0.000001213 | +0.000007746 |
| 위 gain 95% CI | [-0.000001965,+0.000005602] | [-0.000003134,+0.000024332] |

**Teacher의 별도 미래 유용성이 확인되었다고 판단하지 않는다.**
Noise8의 점추정은 일부 개선됐지만 불확실성이 크고, 순위는 안정적이지 않다.
CI가 0을 포함한다는 사실이 CVoM/기억이 무용하다는 증명은 아니다.

두 설정에서 고른 결정을 **동일한 noise8 heldout panel**로 다시 비교했다.
따라서 선택이 달라진 효과와 평가 loss의 Monte Carlo 표본 수 변경을 분리했다.

- Generated MSE의 noise8-choice 이득: `+0.000006533`, CI `[-0.000002320,+0.000022339]`.
- Flow-prefix 이득: `-0.000439809`, CI `[-0.001322844,+0.000003657]`.
- Flow 평균 악화는 BinFill episode 1290의 큰 차이 `-0.007047954`가 지배한다.
  이 사례를 삭제하거나 패널을 다시 고르지 않았다. `paired_selection.csv`에 전 사례가 있다.
- 이 숫자는 **로봇 성공률이 아니다**. Flow와 실제 generated action 평가가 다른 방향일 수 있다.

정식 비교 파일: `runs/long_memory/cvom_noise2_vs8_20260922_v2/`.
초기 비교본 `cvom_noise2_vs8_20260922/`도 보존했다. v2는 CLI를 명시적으로 2→8 전용으로 제한한
최종 분석 소스로 동일 raw evidence를 재분석한 것이다. 모델/probe 재실행이나 결과 선별이 아니다.

## 2. 기존 rollout의 저장 동작

출력: `runs/long_memory/echo_storage_audit_20260922/`.
97개 원본 파일 hash, episode/scenario/seed 및 completed-session identity를 검사했다.

| 지표 | FIFO | Learned min_fill4 |
|---|---:|---:|
| 포화 전 거부 KEEP / 포화 전 모든 저장 기회 | 0/4,042 | **1,751/5,289 = 33.106%** |
| 포화 후 거부 KEEP / 포화 후 모든 저장 기회 | 0/2,437 | 524/1,357 = 38.615% |
| 한 번이라도 full 도달 | 77/160 episodes | 53/160 episodes |
| episode별 최종 bank 수의 평균 | 25.2625 | 22.1125 |
| policy call별 WRITE 전 bank 수 평균 | 20.3385 | 17.5895 |
| policy call별 WRITE 후 bank 수 평균 | 20.9623 | 18.1219 |

145/160 learned episodes에서 포화 전 거부가 있었다. Occupancy 4부터 거부가 시작됐고,
모든 pre-full KEEP은 candidate probability<0.5였다. min_fill 미만 거부는 없었다.
Demo pre-full 거부는 625/1,806, execution은 1,126/3,483이다.

이것은 의도된 min_fill4 gate가 실제로 작동했다는 증거다. 조기 거부가 성능 저하 원인인지는
아직 모른다. 저장 수 증가가 자동으로 좋은 기억/좋은 행동을 뜻하지도 않는다.
최종 occupancy와 call-average occupancy의 분모를 혼동하지 않는다.
기존 로그만으로 min_fill32 이후의 bank/actions/success를 재구성할 수 없다.

최초 logical hash 수정 전 감사 출력은 `echo_storage_audit_20260922_initial`에 보존했다.
최종 경로는 output/source/logical summary hash 재검증을 완료한 별도 결과다.

## 3. Writer 지표 정의와 결과

새 분석: `cvom_writer_decision_metrics.py`.
출력: `runs/long_memory/cvom_writer_noise2_v1/`, `cvom_writer_noise8_v1/`.

### 후보 admission

각 context의 **새 후보 하나**의 `write_probability[-1]`만 평가한다. 기존 슬롯의 logit은 섞지 않는다.
Selection A에서 가장 낮은 loss를 보인 OLD-slot 교체를 고른 뒤,
`KEEP loss - 선택한 교체 loss`가 양수인지 surrogate label로 사용한다.
절대 차이 `<=1e-6`은 불확실로 제외하며 개수를 명시한다. 이는 정식 통계 유의성 기준은 아니다.

- Noise2: 7/16 =43.75%, 불확실 0개.
- Noise8: 7/15 =46.67%, 불확실 1개.
- 양쪽 모두 남은 label이 전부 ADMIT이어서 always-admit baseline=100%, balanced accuracy는 **정의되지 않음**.

이는 admission 문제가 해결되지 않았다는 경고지만, 독립된 참 정답 정확도가 아니다.
32개 중 가장 좋아 보이는 교체를 선택했기 때문에 label의 낙관적 편향도 있다.
해당 **동일 교체**의 B/heldout 이득을 각 context에 같이 기록했다. Heldout에서 교체 대상을 다시 고르지 않는다.

### 교체 순위

OLD 32개만 비교한다. 낮은 utility/retention 점수는 버릴 우선순위이고,
그 항목을 버렸을 때의 낮은 future loss와 순서가 맞는지 검사한다.
Utility와 heuristic이 포함된 retention을 따로 보고한다.
오차 차이 `<=1e-6`의 target tie는 제외, predicted tie는 0.5점이며 개수를 모두 남긴다.

- Heldout retention 순위 정확도: noise2 51.54%, noise8 52.95%.
- Noise8 CI `[48.83,56.82]%`: 아직 불확실.
- Top1은 exact/target-tie-set hit를 나눠 기록한다. 관측 hit가 0일 때 bootstrap [0,0]은
  모집단 확률이 0이라는 뜻이 아니다.

### 선택한 결정의 별도 미래 이득

실제 manager 선택, 강제 retention 교체, utility 교체, A-teacher 교체, candidate gate를
A-teacher 교체에 결합한 선택을 고정한 뒤 heldout에서 KEEP/FIFO 대비 평가한다.
Joint/gripper 및 pure-noise generated prefix를 별도로 저장한다.
큰 전체 utility correlation 하나로 writer 성능을 판정하지 않는다.

## 4. Min-fill32의 실제 평가 실행법

새 wrapper는 checkpoint를 먼저 그대로 검증/로드한 다음 **런타임 min_fill만 32**로 바꾼다.
학습 config는 min_fill4로 남고, `effective_echo_config`, `effective_min_fill`,
`min_fill_override`, wrapper source hash를 manifest와 모든 policy-call info에 별도로 기록한다.

가득 차기 전에는 모두 APPEND, 가득 찬 뒤에는 기존 learned admission/교체 규칙을 그대로 쓴다.
원본 HAMLET short, AE LoRA, precision=native, action16, seed6, denoising/종료 기준은 유지한다.

실제 native regression 완료:

- episode1355의 72 endpoints × parent/default wrapper/min_fill32 =216 endpoints.
- 실제 AE 생성 15회, decoded action 포함 비교.
- 기본 wrapper는 parent와 action·bank·session RNG·기존 info가 정확히 같았다.
- min_fill32 첫 32회 모두 APPEND, pre-full 거부 0. Weight/RNG 변경 없음.
- 결과: `runs/long_memory/echo_min_fill_regression_20260922/completed.json`.

이제 사용자가 실행할 단계:

```bash
cd /home/sjkim/HAMLET-Isaac-GR00T

# 읽기 전용 확인 (이미 통과했지만 실행 직전 다시 확인 가능)
bash run_scripts/robomme/run_cvom_decision_study.sh minfill-preflight

# 새 min_fill32만 16 tasks ×10 VAL/seed6 평가. 학습 아님.
# 기존 learned min_fill4 36/160을 엄격 검증 후 재사용한다.
bash run_scripts/robomme/run_cvom_decision_study.sh minfill-eval

# 완료 후 성공률/paired 차이 집계
bash run_scripts/robomme/run_cvom_decision_study.sh minfill-report

# 점유·거부율까지 같은 scenario끼리 비교
bash run_scripts/robomme/run_cvom_decision_study.sh storage-compare
```

새 결과: `runs/eval/robomme/echo_min_fill32_val160_seed6/comparison_summary.txt` 및 JSON.
Task별 successes, matched wins/losses/same, paired CI, runtime override 증거가 저장된다.
중단 후 같은 `minfill-eval`을 다시 실행하면 같은 manifest에서 남은 평가만 이어간다.
Checkpoint/env/source/scenario가 바뀌면 기존 결과를 강제로 재사용하지 않고 거부한다.
기존 `comparison_manifest.json`과 episode 결과를 수정하지 않는다.

기존 strict validator는 그대로 두고 새 wrapper에 대해서만 별도 검증 경로를 쓴다.
원본 코드·checkpoint·환경은 그대로이고, 추가 server wrapper 및 min_fill 필드만 명시적으로 다르다.
원본 baseline과 기존 ECHO control의 metadata/seed/완료 상태를 검증하며, native regression이 없으면 거부한다.

## 5. 이후 학습은 왜 아직 실행하지 않는가

```bash
bash run_scripts/robomme/run_cvom_decision_study.sh teacher-review
```

동일 raw evidence를 다시 검사하여 게이트를 계산한다. 현재 `eligible_for_manual_training_review=false`다.
판정 항목은 충분한 episodes, A/B rank CI>0.5, 별도 미래 flow/generated의 FIFO/KEEP 대비 양의 CI다.
작은 개발용 표본의 탐색적 최소 요건이지 절대적인 과학적 필요조건은 아니다.
Gate 실패는 증거 부족이며, CVoM 불가능 판정이 아니다.
Gate 통과 시에도 **자동 학습하지 않는다**. `all`, `train`, `adapt`는 명시적으로 중단된다.

사용자가 요청한 조건이 아직 충족되지 않아 이번에는 reader 적응 루프를 새로 확장하지 않았다.
기존 `train_echo_cvom.py warmup --resume stage2`를 reader 적응 명령으로 쓰면 안 된다:
warmup은 Stage1/FIFO와 기존 학습 모듈 구성을 강제하기 때문에 다른 실험이다.
또 기존 labels/writer 명령은 min_fill32 runtime override를 자동 상속하지 않는다.

향후 teacher가 확인되면 다음 세 arm을 분리해야 한다:

1. 현재 actor+writer control 보존.
2. 같은 actor를 동결하고 TRAIN-only budget label로 writer만 재학습.
3. 2번 writer/encoder/AE는 동결하고 **learned bank 분포에서 reader/fusion만 짧게 적응**.

단계별 동일 VAL160으로 비교하며, writer 변화와 reader 적응 효과를 합쳐 주장하지 않는다.
이후 budget-label export/within-bank ranking 학습 및 learned-bank reader-only 루프는
그 조건이 충족된 뒤 명시적으로 구현/검증할 항목이다. 이번 산출물에 구현 완료라고 포함하지 않는다.

## 재실행/출력 보호

이미 수행한 noise/분석을 확인하려면 `status` 또는 `teacher-review`만 실행하면 된다.
`noise8`은 동일 protocol 결과가 있으면 검증 후 재사용하며 재학습하지 않는다.
`noise-compare`, `storage-audit`, `writer2`, `writer8`은 기존 출력 경로를 덮어쓰지 않는다.
재분석 시 `CVOM_COMPARISON_OUTPUT`, `CVOM_STORAGE_OUTPUT`, `CVOM_WRITER_OUTPUT`으로 새 경로를 지정한다.
`minfill-verify`도 기존 검사 결과를 덮어쓰지 않는다.

검증: 신규 comparison/writer/runtime/storage/evaluator와 기존 budget 테스트를 합쳐 **123개 통과**.
Shell 문법, 실제 noise8 실행/strict 비교, native AE 회귀, min_fill VAL160 preflight를 확인했다.
이번 작업에서 모델 학습, 추가 writer/reader 체크포인트 생성, min_fill32 simulator rollout은 수행하지 않았다.
