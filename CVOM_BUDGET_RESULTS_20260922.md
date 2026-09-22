# 고정 용량 CVoM teacher 검사 — 실제 실행 결과

## 결론

검사 코드는 정상 실행됐지만, **이 설정의 정답으로 writer를 바로 다시 학습할 근거는 확보되지 않았다.**
같은 미래 query에서 noise panel만 바꿔도 operation 순위가 불안정했고,
선택한 operation의 별도 미래 action 예측 이득은 FIFO 대비 불확실했다.
이는 작은 frozen-actor/offline 진단이다. CVoM이 불가능하다거나 RoboMME 성공률이
바뀌지 않는다는 증명으로 해석하지 않는다.

## 실제로 실행한 것

```bash
cd /home/sjkim/HAMLET-Isaac-GR00T
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python run_scripts/robomme/cvom_budget_probe.py preflight \
  --contexts 16 --output-dir runs/long_memory/cvom_budget_val16_v1
bash run_scripts/robomme/run_cvom_budget_probe.sh smoke
bash run_scripts/robomme/run_cvom_budget_probe.sh resume
```

- `echo_cvom_full_v1/refresh/checkpoint-001000`, actor 전체 동결.
- 계획을 사전 고정한 cache-VAL **16 episodes / 6 tasks**; TRAIN/TEST 사용 안 함.
- 각 full FIFO32 bank에 candidate를 추가한 뒤, 33개 drop 선택을 같은 용량에서 전수 비교.
- 선택 future 2개 × noise 2회, 같은 future의 독립 noise 재검사, 별도 미래 2개 × noise 2회.
- 미래 query 사이 50-step GT chunk 비중첩. 검증 미래에서 실제 Euler generation도 실행.
- 6,352 flow 호출, 1,072 action generation 호출. generation당 원래 4 denoising steps 유지.
- 작은 스모크 2개를 저장한 뒤 동일 provenance로 14개 이어 실행하여 resume도 확인.
- 실측 진단+종료 감사 시간 합계 약 155초. checkpoint 로딩/사전 검사 시간 제외.
- 정책/encoder/writer 최적화 없음. simulator rollout 없음. 기존 결과/체크포인트 변경 없음.

프로토콜 fingerprint:
`451d73d8f4d1658331ede7becff2a4aaf3b1d4c1a27215ab9a1728a7295e6a7d`

원본 증거: `runs/long_memory/cvom_budget_val16_v1/`의
`protocol.json`, `context-*.json`, `verification-*.json`, `summary.json`, `contexts.csv`, `report.md`.

## 결과

양의 gain = 기준 오차 − 선택한 operation의 오차.
아래 평균/CI는 episode-macro, episode-clustered bootstrap 5,000회 기준이다.

| 검사 | 관측값 | 95% CI |
|---|---:|---:|
| A/B에서 동일한 최선 operation | 2/16 = 12.5% | [0.0%, 31.25%] |
| operation 쌍의 순서가 A/B에서 동일 | 4,268/8,448 = 50.52% | [47.19%, 53.82%] |
| 선택 A의 held-out flow-prefix gain vs FIFO | +0.000001804 | [-0.000006531, +0.000012234] |
| 선택 A의 held-out generated-prefix MSE gain vs FIFO | +0.000001213 | [-0.000001965, +0.000005602] |
| 기존 manager의 held-out generated-prefix MSE gain vs FIFO | -0.000002347 | [-0.000005112, -0.000000185] |

완전 동률 operation pair는 없었다. 비교 operation은 33개이므로 최선 선택 일치율의
우연 기준이 50%라는 뜻은 아니다. 위 50.52%는 **두 operation 사이의 순서 일치율**이다.
현재 표본에서 순위가 안정적이라고 보기 어렵다는 근거다.

Generated-prefix MSE:

| 선택 방식 | 검증 미래의 평균 MSE |
|---|---:|
| FIFO | 0.003275769 |
| A에서 선택한 operation | 0.003274556 |
| KEEP | 0.003278604 |
| 기존 frozen manager | 0.003278116 |

선택 A의 FIFO 대비 상대 MSE 감소는 약 **0.037%**에 불과하며 CI도 0을 포함한다.
이 비율은 **로봇 성공률 변화가 아니다**.
Task-macro에서도 FIFO 대비 generated gain `+0.000001140`, CI
`[-0.000001608,+0.000004727]`로 결론이 달라지지 않는다.

Joint MSE gain은 `+0.000001443`, gripper MSE gain은 `-0.000000396`이며 모두 CI가 0을 포함한다.
총 오차 한 가지만 보고 성공/실패 유형을 단정하지 않는다.

기존 manager는 full FIFO context에서 KEEP 9개, FIFO 1개, 다른 slot 교체 6개를 선택했다.
이 조건에서 FIFO보다 오차가 높았지만, **실제 learned-prefix 분포의 rollout**을 재현한 것은
아니다. 앞선 learned 36/160 vs FIFO 33/160 결과와 직접 동일시하지 않는다.

검증 미래를 미리 보고 가장 낮은 오차의 operation을 고르면 FIFO 대비 MSE gain은
`+0.000014232`지만, 이는 **같은 데이터에서 선택하고 평가한 낙관적 oracle**이다.
배포 가능한 개선이나 기억의 의미 보존 증거로 사용하지 않는다.

## 검증 범위와 한계

- 신규 CPU 테스트 20개 통과. 관련 기존 ECHO 테스트를 포함하면 **48개 통과**.
- 16개 context 모두 same-input flow/generated metric 반복 오차 0.
- global Torch RNG 변화 없음, actor 전체 파라미터/버퍼 hash 전후 일치.
- checkpoint payload, cache file-stat signature, 실행 source 전후 일치.
- cache 통계 signature는 cryptographic content hash가 아니다.
- GPU 0 사용 전에 다른 compute process 없음을 확인했고, 종료 후 GPU 작업도 남기지 않았다.
- full bank와 분리된 미래가 가능한 task는 6개뿐이다. 16-task 전체 성능 검사가 아니다.
- 각 task 3 episodes, RouteStick만 1 episode. task별 불확실성은 매우 크다.
- 현재 입력/GT는 기록된 teacher 경로이며 closed-loop 상태 분포를 평가하지 않았다.
- bank는 개입 후 고정되어 미래 WRITE/eviction은 전개하지 않는다.
- generation bridge와 기존 cache 정밀도 사용. native 온라인 observation feature 재검사 아님.
- 여러 탐색적 대비의 CI이며 다중 비교 보정·최종 일반화 주장 없음.

## 다음에 바꿀 변수 하나 — 아직 실행하지 않음

**같은 episode/context/미래 query에서 flow noise 반복만 2 → 8로 늘리는 검사**가 우선이다.
Actor, capacity, selection target, query, generation, seed 규칙은 유지한다.
순위 불안정이 Monte Carlo 표본 부족 때문인지 먼저 분리하는 목적이다.

판정 기준은 단순 top1 일치율 상승만이 아니다:

1. A/B pair 순위 재현성이 개선되는가?
2. A에서 고른 선택의 별도 미래 flow 및 generated-prefix 이득이 FIFO 대비 유지되는가?
3. task별 상쇄/소수 episode 지배가 아닌가?

noise를 늘려도 별도 미래에서 이득이 없으면 writer fitting을 늘리지 않는다.
Teacher의 조건부 예측 가치와 실제 저장 목적/행동 사용 경로를 다시 구분해야 한다.
이번에는 추가 noise 검사나 ranker 학습을 자동 실행하지 않았다.
