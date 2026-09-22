# CVoM 정답 신뢰도 검사 — 학습 전, 고정 용량 저장 결정

## 목적과 변경 범위

2026-09-22. ECHO 결과는 FIFO 33/160, learned 36/160, 원본 HAMLET 38/160이었다.
Learned−FIFO +1.875pp의 paired 95% CI는 [-1.25,+5.0]pp여서 개선 확정은 아니다.
기존 additive teacher는 다양한 크기의 `S`에서 `J(S)-J(S+event)`를 평균했다.
이번에는 **동일 full bank에서 어떤 사건 하나를 버리는 것이 좋은지**를 측정한다.
새 writer, loss, 학습 파이프라인을 추가하지 않는다. 결과를 보고 자동 학습하지 않는다.

기존 semantic 코드도 operation 비교를 시도했다. 따라서 operation 비교 자체를 새 해결책으로
주장하지 않는다. 이번 차이는 경쟁 선택의 common noise, 독립 noise 재검사,
선택하지 않은 미래 시점에서의 이득 검증을 **학습 전에** 수행하는 것이다.

모든 HAMLET/AE LoRA/encoder/reader/fusion/manager를 동결한다.
기존 checkpoint·캐시·rollout 결과를 덮어쓰지 않는다. 기존 추론 precision도 바꾸지 않는다.
계산은 원래 cache 특징과 기존 flow/생성 bridge를 사용한다. native 온라인 특징의 재검사는 아니다.

## 고정 실험

기본 checkpoint: `runs/long_memory/echo_cvom_full_v1/refresh/checkpoint-001000`.
기본 cache: `runs/long_memory/cache_full1600_v1`.

1. cache-VAL에서 task-balanced **16 episodes, episode당 한 context**를 결과 확인 전에 선택한다.
2. FIFO로 관측 `[0,t-1]`을 저장한다. bank가 32개 찬 시점만 선택한다.
   학습된 admission이 채우기 전에 버리는 영향은 이 검사에서 분리한다.
3. 관측한 새 사건 t를 더해 33개 pool을 만든다. 각각 하나씩 버려 **모든 선택이 32개**를 남긴다.
   - KEEP: 새 사건을 버림.
   - FIFO: 가장 오래된 사건을 버리고 새 사건 추가.
   - 나머지 31개: 해당 과거 사건 하나를 버리고 새 사건 추가.
   - frozen learned manager가 같은 FIFO bank에서 고른 선택도 함께 보고한다.
4. 후보보다 HAMLET short window 바깥의 미래 query를 사용한다 (`q > t+4`, endpoint index 기준).
   선택용 미래 2개(A), 같은 미래+다른 noise 2개(B), 더 나중의 검증 미래 2개(H)를 고정한다.
   A와 H 사이에는 **50-step GT chunk 전체가 겹치지 않는 frame 간격**을 둔다.
5. A/B/H에서 query당 flow noise 2회. 같은 query·noise에서 33개 선택 모두
   현재 short, state, GT, mask, noise/timestep이 같다. victim별 seed를 만들지 않는다.
6. H에서는 GT 없는 pure-noise Euler action generation도 query당 1회, 33개 선택 모두 실행한다.
   기존 action 생성 timestep 수를 유지한다. GT는 생성 후 오차 계산에만 사용한다.
7. **A의 executed-prefix flow 오차만으로 선택**한다. B는 noise 재현성 검사,
   H는 선택의 미래 이득 검사다. H에서 가장 좋은 선택은 낙관적인 in-sample oracle로만 표시한다.

과거 bank는 개입 후 고정한다. 중간/미래 event를 추가하지 않는다.
이는 snapshot의 조건부 예측 가치이며, 미래 FIFO 전개·환경 성공·terminal reward가 아니다.
현재 query와 과거 HAMLET source window 간 중복 정보는 존재할 수 있다.
토큰 하나가 프레임/물체 하나라는 가정도 하지 않는다.

강화된 미래 분리 조건에서 접근 가능한 데이터는 TRAIN 246 / cache-VAL 69 episodes이며,
양쪽 모두 6 tasks만 지원한다. 기본 실행은 이 중 VAL 16개다:

| Task | eligible TRAIN | eligible cache-VAL |
|---|---:|---:|
| BinFill | 25 | 8 |
| PickXtimes | 18 | 6 |
| RouteStick | 6 | 1 |
| VideoPlaceButton | 80 | 20 |
| VideoPlaceOrder | 78 | 22 |
| VideoRepick | 39 | 12 |

다른 task는 capacity가 차고 충분한 미래가 남는 조건을 만족하지 않는다.
이 진단을 16-task 전체의 결론으로 확대하지 않는다.
기존 **VAL160 simulator 평가와 다른 offline cache-VAL 진단**이다.

## 실행

```bash
cd /home/sjkim/HAMLET-Isaac-GR00T

# 읽기 전용: 경로·checkpoint·분할·고정 패널·계산량 확인
bash run_scripts/robomme/run_cvom_budget_probe.sh preflight

# 새 경로에서 16개 모두. GPU 사용 중이면 시작하지 않음
bash run_scripts/robomme/run_cvom_budget_probe.sh run
```

스모크를 먼저 하고 싶다면 `run` 대신 `smoke`를 사용한다. 동일한 16개 계획에서
처음 2개만 완료하고 멈춘다. 이미 스모크가 끝났거나 중단되었다면:

```bash
bash run_scripts/robomme/run_cvom_budget_probe.sh resume

# GPU 없이 저장된 결과만 재집계
bash run_scripts/robomme/run_cvom_budget_probe.sh report
```

새 실험 경로는 `CVOM_PROBE_OUTPUT=runs/long_memory/cvom_budget_val16_v2`처럼 지정한다.
`run`은 기존 경로를 거부한다. `resume`은 checkpoint, 패널, source, 환경, cache 파일 정보가
완전히 같아야 한다. 소스 수정 후 옛 provenance를 고쳐 억지로 이어가지 않는다.
이 도구의 결과는 context 단위로 저장하므로 중단 시 최대 한 context만 재계산한다.
모델/캐시/소스 불변성의 최종 감사가 완료되어야 보고서를 작성할 수 있다.

직접 CLI에서 `--generation-repeats 0`을 쓸 수 있지만 이는 flow-only screening이다.
그 결과를 실제 action generation 검사라고 하지 않는다. 설정이 달라지므로 새 출력 경로가 필요하다.
`--split train`도 가능하지만 writer는 학습하지 않으며 TEST split은 허용하지 않는다.

## 출력과 해석

`runs/long_memory/cvom_budget_val16_v1/`:

- `protocol.json`: 성과와 무관하게 확정한 episodes/시점/미래 panel, checkpoint/cache/source/environment.
- `context-*.json`: 선택별 모든 paired loss, seed, bank event 출처, frozen manager 선택.
- `verification-*.json`: parameter/checkpoint/cache/source 불변성, 해당 결과 파일에 묶인 감사.
- `summary.json`, `contexts.csv`, `report.md`: episode별·task별 결과와 episode-clustered bootstrap.
- `launch-*.json`: 실행 인자와 GPU 점유 사전 검사.

주요 해석:

- `stability`: A/B에서 순위와 최선 선택이 재현되는가? 동률은 별도로 표시한다.
- `selected_a_vs_fifo`, `selected_a_vs_keep`: A에서 고른 선택을 H에 적용했을 때의 이득.
  **기준 오차 − 선택 오차**, 따라서 양수가 개선이다.
- `generated_mse`: 실제 denoising을 거친 생성 action의 관측된 실행 prefix 오차. 로봇 성공률 아님.
- joint/gripper 오차도 별도로 저장·집계한다. 한 차원이 총 오차를 지배하는지 함께 확인한다.
- `manager`: 기존 writer가 이 full FIFO bank에서 고른 선택. 실제 learned-prefix 분포와는 다르다.
- `heldout_oracle_*_in_sample`: H를 보고 고른 최상의 선택이므로 낙관적 참고값일 뿐이다.

context/프레임을 독립 표본으로 취급하지 않는다. episode-macro와 task-macro를 구분한다.
Task당 1 episode인 경우 bootstrap으로 그 task의 실제 변동을 추정할 수 없다.
특히 RouteStick은 eligible cache-VAL 자체가 1 episode뿐이다.
16개는 qualification의 작은 시작이며 CI가 넓거나 0을 포함하면 불확실이다.

다음 단계는 자동 실행하지 않는다:

- 순위가 반복되고, 고른 선택이 H의 flow와 generated-prefix 오차에서 FIFO/KEEP보다 유리하면
  그때 TRAIN context로 **고정 actor + within-bank ranking writer** 한 가지만 검토한다.
- noise를 바꾸면 선택이 뒤집히거나, H에서 이득이 사라지면 teacher 정답 문제를 우선 조사한다.
- H oracle 자체도 거의 차이가 없다면 이 snapshot/actor에서 저장 결정을 바꿀 여지가 작은 것일 수 있다.
  장기기억이 불필요하다는 증명은 아니다. read→decision→action 연결은 별도 검사가 필요하다.

코드 위치: `cvom_budget_probe.py`(고정 패널/teacher/CLI),
`cvom_budget_statistics.py`(통계), `tests/test_cvom_budget_*.py`(CPU 회귀 검사).
