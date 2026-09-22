# ECHO-CVoM 구현 검증 — 2026-09-21

이는 **실행·인과성·저장 계약 검사**다. Full training이나 성공률 개선 결과가 아니다. 기존 V19/원본 HAMLET checkpoint·cache·평가 결과를 덮어쓰지 않았다. 새 산출물 root는 `runs/long_memory/echo_cvom_smoke_20260921`이다.

## 실제 실행 결과

| 검사 | 결과 |
| --- | --- |
| 신규 ECHO + 이전 admission 회귀 pytest | **96 passed**, matplotlib deprecation warnings 14개 |
| bash syntax / full warmup preflight | 통과; 24,286 queries / batch 4 / 6,072 updates 계획 |
| 실제 GPU stage 1 | query batch 1, 2 updates 정상 종료; 전체 24,286-step smoke 계획에서는 의도적인 `paused` |
| FIFO-bank teacher | TRAIN 8 + cache-VAL 4 contexts, 37 targets, 실제 paired AE forward 1,184회, frozen state 유지 |
| CPU writer | 20 updates 완료, actor/expert 불변 확인 |
| Learned-bank teacher refresh | TRAIN 4 + cache-VAL 2 contexts, 18 targets, paired AE forward 576회, frozen state 유지 |
| CPU writer refresh | 20 updates 완료, actor/expert 불변 확인 |
| 실제 native VLM/AE replay | VAL episode 1355의 72 endpoints와 626의 6 endpoints, READ ON/OFF 각각 검사; 총 156 endpoints / 22 AE 생성 |
| Evaluation preflight | 두 Python 환경, 원본 HAMLET baseline reference, 고정 VAL160 계약 검증 통과; simulator/출력 생성 없음 |

Stage 1 initial→2 업데이트를 실제 tensor로 비교했다. `memory.*` 43개 중 26개, `effect_adapter.*` 10개 전부, AE LoRA tensor 256개 전부가 변했다. `manager.*` 10개는 변하지 않았다. 이 비교는 업데이트와 동결 계약의 증거이지, 모든 변경이 유용하다는 증거가 아니다.

작은 stage 1 offline validation은 action loss `0.004507 → 0.004492`, generated-prefix MSE `0.004145 → 0.004139`였다. **2-step smoke의 미세한 수치 차이를 수렴·성능 향상이라고 해석하지 않는다.** 작은 writer 검증도 final refresh utility correlation `-0.649`, confident write accuracy `0.25`로 좋지 않았다. 이 20-step/극소 라벨 smoke는 배관 검사용이며 이 checkpoint를 성능 평가용 모델로 승격하지 않는다.

Native replay에서는 다음을 검사했다.

- 실제 실행 control의 온라인 정규화 및 mask가 기존 cache와 정확히 일치.
- 첫 관측과 demo에는 완료 action-effect를 만들지 않음.
- READ-before-WRITE, session reset, READ-off의 WRITE/관측 상태/RNG 소비 유지.
- 같은 native 관측을 offline prefix와 online step으로 인코딩했을 때 query/stored 최대 절대 오차: episode 1355 `2.8610e-6`, episode 626 `1.4305e-6`.
- READ-off fused output은 원래 short와 동일. 실제 AE를 생략하지 않았음.
- 검사 전후 core/expert weights 불변.

이는 동일 입력의 batch-shape 수치 차이 범위 내 일치다. Native/cached VLM feature 전체가 bitwise 동일하다는 검사가 아니며, teacher 관측을 재생하므로 closed-loop task success도 아니다.

## 산출물과 식별 정보

- 시작 parent: `runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072`
- 기존 cache: `runs/long_memory/cache_full1600_v1`
- Cache fingerprint: `f4ee3fabe5d0e41840a72b8345d2688a94e94a4296689dea98adc284935ccb46`
- Stage 1 smoke: `stage1/checkpoint-000002`
- Writer / refresh smoke: `stage2/checkpoint-000020`, `refresh/checkpoint-000020`
- Teacher protocols/packet hashes/frozen-state checks: `labels0/{protocol,manifest}.json`, `labels1/{protocol,manifest}.json`
- 실제 native 검사: [plan](runs/long_memory/echo_cvom_smoke_20260921/runtime_verification/plan.json), [completed](runs/long_memory/echo_cvom_smoke_20260921/runtime_verification/completed.json), [events](runs/long_memory/echo_cvom_smoke_20260921/runtime_verification/events.json)
- 각 checkpoint의 `checkpoint.json`에 base/parent/payload/source/config/plan 식별 정보를 보존했다.
- 초기 smoke 후 label 검증·미완료 checkpoint guard 등을 강화했다. 최초 stage 1 source hash를 사후 수정하지 않았으며 각 실제 실행의 source hash는 해당 산출물에 그대로 기록되어 있다. 변경 후 full workflow는 새 run에서 시작한다.

Stage 2/refresh는 자체 20 updates를 끝냈지만 `inherited_actor_training_complete=false`, `smoke_only=true`다. 일반 평가 CLI가 이를 완료 학습 모델로 받지 않는 guard도 검사했다. 아래 evaluation preflight의 opt-in은 이 기능 검사에만 사용했다.

## 실행한 주요 명령

아래는 실제 실행 기록이다. 다시 검사하려면 root를 `echo_cvom_smoke_20260921_repeat1` 등 **새 경로**로 바꾸고 모든 참조도 함께 바꾼다. 완료된 과거 run을 덮어쓰지 않는다.

```bash
cd /home/sjkim/HAMLET-Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NO_ALBUMENTATIONS_UPDATE=1

.venv/bin/python run_scripts/robomme/train_echo_cvom.py warmup \
  --output-dir runs/long_memory/echo_cvom_smoke_20260921/stage1 \
  --query-batch-size 1 --val-per-task 1 --val-noise-samples 1 \
  --stop-after-steps 2 --eval-steps 2 --save-steps 2 --plot-steps 2 --log-steps 1 \
  --activation-checkpointing

.venv/bin/python run_scripts/robomme/train_echo_cvom.py labels \
  --checkpoint runs/long_memory/echo_cvom_smoke_20260921/stage1/checkpoint-000002 \
  --output-dir runs/long_memory/echo_cvom_smoke_20260921/labels0 \
  --train-context-limit 8 --val-context-limit 4 --contexts-per-episode 2 \
  --targets-per-context 4 --coalitions 4 --future-samples 2 --noise-samples 2

.venv/bin/python run_scripts/robomme/train_echo_cvom.py writer \
  --checkpoint runs/long_memory/echo_cvom_smoke_20260921/stage1/checkpoint-000002 \
  --labels-dir runs/long_memory/echo_cvom_smoke_20260921/labels0 \
  --output-dir runs/long_memory/echo_cvom_smoke_20260921/stage2 --device cpu \
  --writer-updates 20 --writer-batch-size 8 --eval-steps 10 --save-steps 10 --plot-steps 10 --log-steps 5

.venv/bin/python run_scripts/robomme/train_echo_cvom.py labels \
  --checkpoint runs/long_memory/echo_cvom_smoke_20260921/stage2/checkpoint-000020 \
  --output-dir runs/long_memory/echo_cvom_smoke_20260921/labels1 \
  --train-context-limit 4 --val-context-limit 2 --contexts-per-episode 2 \
  --targets-per-context 4 --coalitions 4 --future-samples 2 --noise-samples 2

.venv/bin/python run_scripts/robomme/train_echo_cvom.py writer \
  --checkpoint runs/long_memory/echo_cvom_smoke_20260921/stage2/checkpoint-000020 \
  --labels-dir runs/long_memory/echo_cvom_smoke_20260921/labels1 \
  --output-dir runs/long_memory/echo_cvom_smoke_20260921/refresh --device cpu \
  --writer-updates 20 --writer-batch-size 8 --eval-steps 10 --save-steps 10 --plot-steps 10 --log-steps 5

.venv/bin/python run_scripts/robomme/verify_echo_cvom.py \
  --checkpoint runs/long_memory/echo_cvom_smoke_20260921/refresh/checkpoint-000020 \
  --output-dir runs/long_memory/echo_cvom_smoke_20260921/runtime_verification --device cuda:0

.venv/bin/python run_scripts/robomme/eval_echo_cvom.py \
  --checkpoint runs/long_memory/echo_cvom_smoke_20260921/refresh/checkpoint-000020 \
  --models fifo memory --output-dir runs/eval/robomme/echo_cvom_smoke_preflight_20260921 \
  --allow-initialization-checkpoints --preflight-only
```

Full training 및 새 success-rate 결과는 **아직 없음**. 정식 실행은 [설계 문서](ECHO_CVOM_DESIGN.md)의 `run_echo_cvom.sh preflight` / `all`을 사용한다. 배포 전에 자동 native verification을 다시 실행하고, 같은 ECHO actor의 FIFO와 learned memory를 동일 VAL160으로 비교한다.
