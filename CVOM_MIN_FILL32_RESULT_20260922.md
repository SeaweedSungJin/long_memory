# CVoM min_fill=4 → 32: fixed VAL160 rollout 결과

## 결론

동일 ECHO 체크포인트에서 **포화 전 거부를 없애도 성공률이 높아지지 않았다.**
기존 `min_fill=4`는 36/160 (22.500%), 새 `min_fill=32`는 31/160 (19.375%)였다.
이번 개발용 VAL160에서는 -5개, **-3.125%p**다. 설정의 기본값은 변경하지 않는다.

이는 현재 가중치의 추론 시 저장 규칙을 바꾼 ablation이다. 처음부터 min_fill32 분포로
학습한 모델의 결과가 아니며, 더 많은 기억을 잘 활용하는 학습 자체가 불가능하다는 결론도 아니다.

## 실제 실행과 완료 확인

- 시작 HEAD: `9a63ad64a8a4e4772e755cf423c8529b10165e30`.
- Checkpoint: `runs/long_memory/echo_cvom_full_v1/refresh/checkpoint-001000`.
- Checkpoint runtime SHA256: `2e4cc7f141906eb0817f9a5e37f5a0b4d7e123c8acda9e8adeed80eea29f766a`.
- Control: `runs/eval/robomme/echo_cvom_full_v1_val160_seed6/memory` (기존 완료 결과 검증 후 재사용).
- New run: `runs/eval/robomme/echo_min_fill32_val160_seed6`.
- Evaluation ID: `d776174408a6e88f1ea513cc1449ff25de77e775bf572b666f1f96111ced714f`.
- 16 tasks × 10 scenarios, VAL, inference seed 6, native feature precision,
  action interval 16, max episode steps 1300. 기존 AE LoRA, denoising, READ/fusion 유지.
- 바뀐 변수: bank occupancy <32일 때 강제 APPEND. Full 이후 기존 learned admission/교체 유지.
- 실행 전 GPU0/1 사용 프로세스 없음 확인. GPU0에서 새 arm만 실행.
- 프로세스 exit code 0, `complete=true`, `failures=[]`, `interrupted=false`.
- 실제 완료 160 sessions, missing sessions/identity 0. 종료 후 GPU compute process 없음.
- 모델/캐시/학습 코드/기존 결과 수정 및 새 학습 없음. 영상 재실행 없음.

실제로 실행한 명령:

```bash
cd /home/sjkim/HAMLET-Isaac-GR00T

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash run_scripts/robomme/run_cvom_decision_study.sh minfill-preflight

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash run_scripts/robomme/run_cvom_decision_study.sh minfill-eval

bash run_scripts/robomme/run_cvom_decision_study.sh minfill-report
bash run_scripts/robomme/run_cvom_decision_study.sh storage-compare
```

## 성공률과 paired 결과

| 항목 | min_fill4 | min_fill32 |
|---|---:|---:|
| 성공 | 36/160 | 31/160 |
| 성공률 | 22.500% | 19.375% |
| 포화 전 KEEP / 기회 | 1751/5289 (33.106%) | 0/4042 (0%) |
| 한 번이라도 capacity32 도달 | 53/160 episodes | 77/160 episodes |
| episode별 최종 bank 항목 수 평균 | 22.1125 | 25.2625 |
| policy-call별 WRITE 전 bank 항목 수 평균 | 17.5895 | 20.3331 |

실제 새 runtime의 6,476회 호출 모두 override32를 확인했다. APPEND=4,042,
KEEP=1,209, REPLACE=1,225, MERGE=0이며 모든 KEEP은 포화 후다.
따라서 단순히 옵션이 적용되지 않아 나타난 결과는 아니다.
Bank 항목은 RGB 프레임/객체 수가 아니다. 점유 수 비교에는 서로 달라진 rollout 경로·길이 영향도 포함된다.

- 기존 실패 → 새 성공: **0**.
- 기존 성공 → 새 실패: **5**.
- 양쪽 성공: **31**, 양쪽 실패: **124**.
- Paired delta: **-3.125%p**.
- 95% within-task episode-paired percentile bootstrap CI (5,000 draws, seed17): **[-5.625, -1.250]%p**.
- Exact two-sided McNemar p: **0.0625**.

Bootstrap CI는 음수지만 discordant pair가 5개뿐이고, 관측된 개선 반전은 0개다.
이 작은 표본에서 percentile bootstrap의 경계/이산성 한계와 exact test의 차이를 무시하여
"통계적으로 확정된 보편적 악화"라고 표현하지 않는다. 이번 패널에서 개선은 관측되지 않았고
점추정은 악화였다는 것이 명확한 결과다. 반복 개발에 사용한 VAL160/단일 seed이며 최종 TEST 주장이 아니다.

## Task별 성공 수 (각 10개)

| Task | min_fill4 | min_fill32 | 변화 |
|---|---:|---:|---:|
| BinFill | 1 | 1 | 0 |
| PickXtimes | 1 | 1 | 0 |
| SwingXtimes | 4 | 4 | 0 |
| StopCube | 1 | 1 | 0 |
| VideoUnmask | 5 | 5 | 0 |
| VideoUnmaskSwap | 1 | 1 | 0 |
| ButtonUnmask | 1 | 1 | 0 |
| ButtonUnmaskSwap | 1 | 0 | -1 |
| PickHighlight | 4 | 4 | 0 |
| VideoRepick | 1 | 1 | 0 |
| VideoPlaceButton | 4 | 3 | -1 |
| VideoPlaceOrder | 5 | 5 | 0 |
| MoveCube | 5 | 2 | -3 |
| InsertPeg | 0 | 0 | 0 |
| PatternLock | 0 | 0 | 0 |
| RouteStick | 2 | 2 | 0 |

총점이 같은 나머지 task에는 성공/실패 상쇄 반전도 없었다.

## 반전된 scenario (episode index는 0부터)

| Task | Episode | Scenario seed | Inference seed | 기존 steps | 새 steps/status |
|---|---:|---:|---:|---:|---|
| ButtonUnmaskSwap | 4 | 1070400 | 1615499610 | 1227 | 590 / fail |
| VideoPlaceButton | 8 | 1100800 | 1551451791 | 458 | 279 / fail |
| MoveCube | 0 | 1140000 | 1258496019 | 1062 | 1300 / step_limit |
| MoveCube | 2 | 1140200 | 1851718550 | 312 | 269 / fail |
| MoveCube | 8 | 1140800 | 452633870 | 199 | 1300 / step_limit |

`fail`은 simulator의 종료 상태이며 구체적인 물리적 실패 원인을 설명하지 않는다.
영상 또는 추가 trajectory 분석을 하지 않았으므로 잘못된 검색/충돌/순서 오류 등의 원인을 단정하지 않는다.

## 해석과 종료

확정 사실: 더 일찍 더 많이 저장하도록 바뀌었고, 이번 성공 수는 감소했다.
따라서 **현재 actor에서 포화 전 강제 저장만 적용하는 변경은 채택하지 않는다**.
가능한 가설은 불필요한 기억 간섭, 달라진 bank 분포에 대한 reader/fusion 부적응 등이나,
이번 저장 규칙 하나의 ablation만으로 어느 모듈이 원인인지 특정할 수 없다.
특히 기존 min_fill4 또는 teacher/writer의 학습이 잘 되어 있다는 증명도 아니다.

Teacher 별도 미래 유용성이 아직 확인되지 않았다는 이전 결론은 그대로다.
이번 결과를 이유로 전체 재학습, writer 재학습, reader 적응을 자동 실행하지 않았다.

## 원본 산출물

- `runs/eval/robomme/echo_min_fill32_val160_seed6/comparison_summary.json` / `.txt`
- 같은 폴더의 `comparison_manifest.json`, `driver_status.json`
- `memory/<task>/simulation_results.csv`, `memory_diagnostics.jsonl`, `rollout.log`
- `runs/long_memory/echo_storage_minfill4_vs32_v1/report.md`, `summary.json`,
  `episodes.csv`, `tasks.csv`, `audit_manifest.json`

Storage audit report의 "No new rollout"은 **그 감사 명령 자체가 로그 분석만 한다**는 뜻이다.
이번 작업에서는 그 전에 위 새 rollout 160개를 실제로 완료했다.
