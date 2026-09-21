# V19 feature precision A/B — 2026-09-20

## 사전 고정한 실험

질문: 학습 캐시와 추론 특징의 정밀도 불일치를 제거하면 **동일 V19 정책의 실제 성공률**이 달라지는가?

- 후보 checkpoint: `runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072`.
- 비교 변수 하나: `--feature-precision native|cache-aligned`.
- native: 기존 특징 추출 동작 유지. 기본값도 native.
- aligned: VLM→VLLN→HAMLET short 추출만 BF16 autocast, 전체 postprocessed backbone features를 BF16으로 양자화, 외부 메모리용 normalized moment는 BF16→FP32 경계 적용.
- 내부 HAMLET recurrent cache에는 추가 양자화를 하지 않는다. 기존 cache.py와 같은 내부 계산 상태를 유지한다.
- 외부 메모리 계산 FP32, AE state encoder/denoising은 feature autocast 밖에서 그대로 실행한다.
- 가중치/학습 코드/기존 캐시/메모리 용량/encoder/reader/fusion/writer/loss/denoising step/seed/실행 action 길이/종료 조건을 바꾸지 않는다.
- baseline은 원래 native HAMLET이다. aligned baseline은 이번 실험에 포함하지 않는다.

## 기존 결과와 재사용 조건

원본 결과: `runs/eval/robomme/v19_fullcoverage_v1/prefix_full`.

- native V19: 37/160; 원본 HAMLET native: 38/160.
- 16 tasks × 첫 10 VAL scenarios, seed 6, action interval 16, max episode steps 1300, save_videos=False, device=cuda:0.
- 작업 시작 시 160개 episode CSV, task policy manifests, seed 유도 규칙, driver 정상 종료를 확인했다. native terminal status: success 37 / fail 104 / step_limit 19.
- 수정 전 native 평가의 소스 hash 152개가 작업 시작 시 현재 소스와 일치했다.
- 수정 시작 HEAD: `009ef0cda240901c8d252c89cb96ac312b9e98e6`.
- 기존 진단 작업의 untracked 코드/문서와 `.gitignore` 변경은 보존한다.

native 재사용은 실제 회귀 검사와 환경·checkpoint·scenario/seed·완료 기록의 호환성 검증을 통과할 때만 허용한다. 이번 옵션 연결에 필요한 policy/server/evaluator 3개 변경과 precision helper 1개 추가만 좁게 인정한다. 기존 manifest 내용/hash/evaluation ID는 변경하지 않는다. 다른 차이가 있으면 기존 37/160을 억지로 재사용하지 않는다.

## 실행 게이트

1. 성과를 보지 않고 구조 조건으로 고정한 소규모 cache-VAL panel을 기록한다.
2. 수정 전 native 코드와 수정 후 기본/명시적 native, cache-aligned를 같은 관측/seed로 비교한다.
3. feature-only 검사와 실제 AE action generation을 구분하고, reset/read-before-write/cadence/mask를 확인한다.
4. 위 회귀 및 재사용 provenance 검증 통과 후, aligned VAL160을 끝까지 실행한다.
5. 중간 성공률을 근거로 task/subset/settings/stopping rule을 변경하지 않는다.

## 결과 판정 원칙

- 특징 정렬 성공과 rollout 성공률 개선은 서로 다른 판정이다.
- 주지표는 paired 실제 task success이며, offline loss를 대신 사용하지 않는다.
- episode별 native 실패→aligned 성공, 역방향 변화와 task별 성공 수를 보고한다.
- task 내 episode를 paired bootstrap하여 95% CI를 계산한다. 고정된 task/scenario/seed에 조건부이며 학습 seed·전체 TEST 일반화 불확실성까지 포함하지 않는다.
- CI가 0을 포함하면 불확실하다고 보고한다. “주원인이 아니다”로 단정하지 않는다. CI 상한이 허용하는 개선 크기도 함께 제시한다.
- 반복 개발용 VAL160이므로 최종 일반화 성능 확증으로 취급하지 않는다.
- 영상이 없는 기존 native 결과는 terminal status/step 수 이상의 물리적 실패 원인을 임의로 추정하지 않는다.

이번 A/B 완료 후에만 다음 변수 하나를 제안하며, 후속 학습/ablation을 자동 시작하지 않는다.

## 실행 결과

### 1. 고정 패널 및 실제 회귀 검사: PASS

성과/라벨을 보지 않고 cache-VAL의 시간 구조만으로 먼저 `regression/plan.json`에 고정했다.

| episode | 선택 이유 | 총 event | demo event |
|---|---|---:|---:|
|1355|기존 진단 기준 episode|81|67|
|1363|1355를 제외한 긴 demo 사례|85|71|
|626|demo 없는 짧은 사례|9|0|

- 총 175개 endpoint. 1355/1363은 FIFO32를 넘는다. event/token을 RGB 한 장이나 물체 하나로 해석하지 않는다.
- 과거 평가 manifest가 가리키는 원본 policy 소스를 해당 git HEAD에서 읽어 동일한 동결 checkpoint에 연결했다. 기본 native/명시적 native는 이 수정 전 경로를 재현했다.
- 같은 계산 경로의 feature 비교 10,500건은 값이 정확히 일치했다. native 비교의 기준은 수정 전 native이며, aligned 비교의 기준은 실제 캐시/캐시 재생성 경로이다. **native와 aligned가 서로 같다는 의미가 아니다.**
- moment, 전체 backbone features, short, encoded event, bank-before/after, fused, state, attention/image mask를 확인했다. 내부 reset/read-before-write/demo·execution cadence도 통과했다.
- 실제 AE는 각 episode의 첫/마지막 유효 행동 시점 6곳에서 실행했다. 원래 denoiser를 총 78회 호출했고, 동일 conditioning/seed의 action 비교 66건이 정확히 일치했다. 나머지 관측에서는 denoising을 생략한 feature 검사만 수행했다.
- 실제 AE state encoder/denoiser 및 외부 메모리 계산에 feature autocast가 침범하지 않는지 hook으로 확인했다. native/aligned의 실제 AE conditioning dtype은 모두 BF16이었다.
- batch-shape 차이는 별도 비교했다. 350건 중 211건에서 비트 단위 차이가 있었으나 최대 절댓값 차이는 `2.861e-6`, 최대 상대 L2 오차는 `1.552e-7`이었다. 순차/배치 계산을 비트 단위 동일하다고 주장하지 않는다.
- 실제 action 검사에는 query별 고정 generator를 사용했다. 중간 denoising을 생략했으므로 전체 episode의 RNG 진행을 GPU에서 모두 재연한 검사는 아니다. RNG 코드 무변경과 CPU session 회귀 검사로 이를 보완한다.
- 관련 단위/기존 회귀 테스트: **65개 PASS**. GPU 패널 실행 시간 약 61초. 정책 학습 파라미터 0개이며 기존 checkpoint/cache는 변경하지 않았다.

원시 증거: `runs/diagnostics/v19_precision_ab_20260920/regression/`의 `completed.json`, `plan.json`, feature/action/batch-shape CSV, session/cadence 기록.

### 2. 기존 native 재사용 감사: PASS

`runs/diagnostics/v19_precision_ab_20260920/reuse_audit/audit.json`에 원본 manifest, 160개 episode 기록, task manifests, driver 완료 기록의 SHA256을 보관했다. 체크포인트, scenario/episode/seed, simulator source/scenario metadata, 패키지 버전, 관측·action 규칙을 비교했다. 기존 native diagnostics도 160 sessions / missing 0 / complete evidence 조건을 충족한다.

한계: 과거 evaluator는 simulator Python/scenario 파일과 패키지는 식별하지만, 대형 3-D asset 파일 전체와 당시 하드웨어/드라이버 전체를 hash하지 않았다. 확인하지 못한 항목까지 완전 동일하다고 주장하지 않는다. 기존 provenance를 수정하지 않았다.

### 3. 실제 VAL160 rollout

**완료: 정밀도 정렬에는 성공했으나, 이번 VAL160에서 성공률 개선은 확인되지 않았다. 점추정은 악화했다.**

회귀와 감사 통과 후 aligned 후보만 새로 160개 실행했다. 16 tasks 모두 scenario 0–9를 완료했고 exit code 0, `interrupted=false`, `failures=[]`였다. manifest 생성부터 driver 완료까지 약 17분 54초(9월 20일 23:53:26 → 9월 21일 00:11:20 KST)였다. 중간 결과에 따른 설정/종료 변경은 없었다.

| 조건 | 성공 수 | 성공률 | 이번 새 rollout |
|---|---:|---:|---:|
|원본 HAMLET / native|38/160|23.750%|0, 기존 참조 재사용|
|동일 V19 prefix checkpoint / native|37/160|23.125%|0, 감사 통과 후 재사용|
|동일 V19 prefix checkpoint / cache-aligned|32/160|20.000%|160|

주 A/B는 **두 V19 조건의 비교**이다. evaluator의 일반 `baseline -> memory` 요약(원본 HAMLET 대비 −3.75%p)과 혼동하지 않는다.

- aligned − native: **−3.125%p**.
- native 실패 → aligned 성공: **3개**.
- native 성공 → aligned 실패: **8개**.
- 성공 여부 동일: **149개**(둘 다 성공 29개, 둘 다 실패 120개). 행동 궤적까지 같다는 의미는 아니다.
- task 내 episode paired bootstrap 10,000회, seed 190020: **95% CI [−6.875, +0.625]%p**.
- McNemar exact 양측 p = **0.2265625**.

| task | native 성공 /10 | aligned 성공 /10 | 개선 / 악화 pair |
|---|---:|---:|---:|
|BinFill|2|1|0 / 1|
|PickXtimes|1|1|0 / 0|
|SwingXtimes|4|4|0 / 0|
|StopCube|1|1|0 / 0|
|VideoUnmask|4|5|1 / 0|
|VideoUnmaskSwap|1|1|0 / 0|
|ButtonUnmask|0|1|1 / 0|
|ButtonUnmaskSwap|1|0|0 / 1|
|PickHighlight|4|3|0 / 1|
|VideoRepick|1|1|1 / 1|
|VideoPlaceButton|4|3|0 / 1|
|VideoPlaceOrder|5|5|0 / 0|
|MoveCube|6|5|0 / 1|
|InsertPeg|0|0|0 / 0|
|PatternLock|0|0|0 / 0|
|RouteStick|3|1|0 / 2|

각 task는 모두 10개를 실제 평가했다. `0 / 0`은 성공 여부가 뒤집힌 pair가 없다는 뜻이지, 평가 생략이 아니다. task당 10개이므로 task별 차이를 큰 모집단의 강점/약점으로 일반화하지 않는다.

### 4. 실제 경로 및 기존 결과 보존 최종 확인

- aligned의 runtime diagnostics를 원본 JSONL에서 다시 계산하여 `comparison_summary.json`과 일치함을 확인했다.
- 160 completed sessions, 6,384 policy calls, read-enabled 4,454 calls, missing sessions 0, missing identity 0, `complete_evidence=true`.
- 모든 완료 session의 checkpoint/정밀도 값/정밀도 규칙 및 passive demo read-off 조건을 검증했다. 호출 수 차이는 rollout 궤적과 종료 시점이 달라지므로 가능하며, 호출 수나 gate 값 자체를 유용한 기억의 증거로 해석하지 않는다.
- 작업 전 감사의 기존 native 파일 34개 SHA256이 최종 감사와 모두 동일했다. 원본 manifest, 16개 task CSV와 policy manifests, driver status를 수정하지 않았다.
- checkpoint/base/source/시나리오/패키지 호환성 최종 재검증도 통과했다. CPU 테스트를 종료 후 다시 실행해 65개 PASS, `git diff --check` PASS. 종료 후 남은 GPU compute process도 없었다.
- paired 비교 CLI 자체는 runtime evidence boolean을 강제하지 않으므로, 이번 보고는 위 별도 재계산/검증까지 수행한 결과이다.

### 5. 뒤집힌 사례의 확인 가능한 실패 유형

성공→실패 8개 중 `step_limit`이 2개, 환경의 일반 `fail` 종료가 6개였다.

- BinFill episode 7 (scenario 1040700): 성공 725 steps → step_limit 1,300 steps.
- MoveCube episode 0 (scenario 1140000): 성공 1,122 steps → step_limit 1,300 steps.
- 나머지 새 실패: ButtonUnmaskSwap 4, PickHighlight 5, VideoRepick 4, VideoPlaceButton 8, RouteStick 4/8.
- 반대로 개선된 3개: VideoUnmask 6, ButtonUnmask 2, VideoRepick 2. 모두 `fail → success`.
- 전체 terminal status는 native `success 37 / fail 104 / step_limit 19`, aligned `success 32 / fail 105 / step_limit 23`.

이번에는 영상 저장 조건을 변경하지 않았다. 따라서 여기까지는 **시뮬레이터 종료 유형**이며, 잘못된 과거 검색/물체 선택/접촉 제어 실패 중 무엇인지는 확정할 수 없다. 전체 11개 flip의 scenario seed, inference seed, 지시문, 종료 상태와 steps는 `paired_report/flipped_episodes.csv`에 보관했다.

### 6. 판단 및 다음 행동 하나 — 제안만, 미실행

**현재는 native를 유지한다.** “캐시 정밀도 정렬만 하면 이 checkpoint의 성공률이 오른다”는 개선 가설은 이번 조건에서 지지되지 않았다. 다만 CI가 0을 포함하므로 악화가 확정되었다거나, 정밀도가 무관하다거나, 정밀도 불일치가 주원인이 아니라고 단정하지 않는다.

이 표본/seed/추정법의 95% 구간에는 약 **+0.625%p의 작은 개선**도 포함된다. 이는 일반화 성능의 절대 상한이 아니다. 반복 개발용 VAL160에서 한 checkpoint, 한 inference seed로 수행한 결과이며, 구조적 정보 손실·retrieval·fusion·AE 활용 중 어디가 병목인지는 이번 실험만으로 특정하지 못한다. P2/P3 정확도 차이를 eviction 근거로 재사용하지 않았다.

다음 검사 하나만 권한다면, **뒤집힌 11개 scenario 전부를 native/aligned 동일 seed로 영상 포함 재실행하는 22-episode 재현성·실패 분석**이다. 가중치/메모리/학습을 바꾸지 않고 조건별 재현성과 궤적 분기를 먼저 확인한다. 새 VAL160 두 조건(320개)이나 정책 재학습보다 계산량이 적다.

- 조건별 성공 여부가 재현되면: 해당 사례에서 정밀도 변경에 따른 궤적 분기가 반복되는지와 종료 실패 유형을 확인한다. 재현만으로 잘못된 검색/AE 활용을 단정하지 않는다.
- 동일 조건 결과부터 바뀌면: 정밀도 효과의 해석을 보류하고 실행 변동성/과거 native 재현성 문제로 분류한다.
- 성과를 보고 선택한 11개 사례이므로 **새 성공률 추정이나 우월성 검정에 사용하지 않는다**. 현재 VAL160 수치를 대체/덮어쓰지 않는다.
- 후속 검사를 시작하거나 새 학습/ablation을 자동 실행하지 않았다.

## 산출물

- 요약/실행 명령/해석: 이 문서.
- 회귀 증거: `runs/diagnostics/v19_precision_ab_20260920/regression/`.
- 사전 native 재사용 감사: `runs/diagnostics/v19_precision_ab_20260920/reuse_audit/`.
- 주 native/aligned 비교: `runs/diagnostics/v19_precision_ab_20260920/paired_report/report.md`, `tasks.csv`, `flipped_episodes.csv`, `audit.json`.
- 실제 aligned rollout/episode 로그/runtime manifest: `runs/eval/robomme/v19_precision_val160_seed6/cache_aligned/`.
- 전체 driver stdout: `/tmp/v19-feature-precision-aligned-val160-20260920.log`.

## 실제 실행 명령

모든 명령은 `/home/sjkim/HAMLET-Isaac-GR00T`에서 실행한다. 기존 산출물을 보존하도록 진단 output은 새 디렉터리여야 한다. 아래는 이번 실행에 사용한 정확한 경로이며, 재실행 시에는 새 run 이름을 사용한다.

처음 회귀 패널을 만드는 경우에는 `verify_feature_precision_v19.py --output-dir <새 경로> --prepare-only`로 CPU에서 패널을 먼저 고정한 다음, 같은 경로에 아래 `--execute-plan`을 사용한다. 이미 결과가 들어 있는 이번 regression 디렉터리는 덮어쓸 수 없다.

```bash
# 앞서 사전 등록한 regression/plan.json 실행
CUDA_VISIBLE_DEVICES=0 GR00T_INFERENCE_SEED=6 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/python run_scripts/robomme/verify_feature_precision_v19.py \
  --output-dir runs/diagnostics/v19_precision_ab_20260920/regression \
  --execute-plan

# 기존 native 재사용 허용 여부 감사
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python run_scripts/robomme/compare_feature_precision_v19.py audit \
  --aligned-run runs/eval/robomme/v19_precision_val160_seed6/cache_aligned \
  --regression-report runs/diagnostics/v19_precision_ab_20260920/regression/completed.json \
  --output-dir runs/diagnostics/v19_precision_ab_20260920/reuse_audit

# 아래와 같은 명령에 --preflight-only를 붙여 먼저 환경/입력 검증도 통과했다.
# 실제 aligned VAL160 (baseline은 기존 native 참조 재사용, 새 rollout 0개)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python run_scripts/robomme/eval_representation_v18.py \
  --checkpoint runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072 \
  --feature-precision cache-aligned --models memory \
  --baseline-reference runs/eval/robomme/archive_read_best1250_val_n10_seed6 \
  --tasks all --dataset val --n-episodes 10 --seed 6 \
  --output-dir runs/eval/robomme/v19_precision_val160_seed6/cache_aligned \
  > /tmp/v19-feature-precision-aligned-val160-20260920.log 2>&1

# 실제 완료 후 수행한 native/aligned paired 보고
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python run_scripts/robomme/compare_feature_precision_v19.py report \
  --aligned-run runs/eval/robomme/v19_precision_val160_seed6/cache_aligned \
  --regression-report runs/diagnostics/v19_precision_ab_20260920/regression/completed.json \
  --output-dir runs/diagnostics/v19_precision_ab_20260920/paired_report
```

실행한 CPU 단위/기존 회귀 검사:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/python -m unittest \
  tests.test_feature_precision_v19_policy \
  tests.test_compare_feature_precision_v19 \
  tests.test_verify_feature_precision_v19 \
  tests.test_representation_eval_v18 \
  tests.test_representation_comparison_v18 \
  tests.test_representation_core_v18 \
  tests.test_full_training_workflow_v19 -q
```

## 식별 정보

- native evaluation ID: `c4ccf4e6f27d635a8e9207bc498fde857738caf6562dd826c96c545c40c2518e`.
- aligned evaluation ID: `20f004535dad26ff0dfb1529e2f340481e8430cea48da829914bff7cd355b69d`.
- 원본 HAMLET baseline evaluation ID: `ec9b3756764d0c04e75e841055a95e74de21d936f4b70bb94ef70df97148ac25`.
- checkpoint memory payload SHA256: `7624d3e21ec65425020d4c84e810ab7c8870f38387a1a99326c0eed56445e556`.
- checkpoint adapted expert payload SHA256: `2824c8d917e7a2ca6d2bc78222283b7b3ae8346212f6961271aba18ac06bf5b4`.
- policy 환경: `.venv/bin/python`, torch `2.7.1+cu128`, transformers `4.51.3`, safetensors `0.8.0`, numpy `1.26.4`.
- simulator 환경: `/home/sjkim/robomme_benchmark/.venv/bin/python`, RoboMME `0.1.0`, ManiSkill `3.0.0b21`, SAPIEN `3.0.2`, torch `2.9.1`.
- GPU 사용 직전 compute process가 없음을 확인하고 GPU0 RTX5080만 사용했다. GPU UUID: `GPU-750d8a65-9668-71a1-3204-976941a2812d`.
- 전체 source/weight/scenario hash 및 precision contract는 aligned `comparison_manifest.json`과 감사 JSON에 보관한다. 9월 20일 시작한 작업/경로 이름을 유지하며 rollout은 9월 21일 KST까지 이어졌다.

## 수정 범위

- `run_scripts/robomme/feature_precision_v19.py`: 특징 추출 정밀도 규칙과 좁은 autocast helper (신규).
- `policy_representation_v18.py`, `serve_representation_v18.py`, `eval_representation_v18.py`: 옵션 전달 및 manifest/runtime diagnostics 기록. 기본 native 유지.
- `compare_feature_precision_v19.py`: 이번 A/B에만 허용되는 차이를 검사하는 sidecar와 paired 보고서 (신규). 전역 manifest validator를 느슨하게 하지 않음.
- `verify_feature_precision_v19.py`: 사전 고정 패널, 캐시/원본 native/실제 AE 재현 검사 (신규).
- 위 기능의 신규 테스트 3개 파일 및 기존 `tests/test_representation_eval_v18.py`의 constructor를 우회하는 fixture에 명시적 native 설정 추가.
- 이 문서. 기존 진단 코드/기존 사용자 변경은 보존.
