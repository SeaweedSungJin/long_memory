# V19 native — 외부 memory READ ON/OFF rollout 비교

## 고정 질문과 범위

동일한 `runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072`에서 추가 외부 READ/fusion 경로가 실제 RoboMME 성공률에 기여하는지 검사한다.

- 주 비교: **ON 성공률 − OFF 성공률**, 양수가 ON에 유리.
- 원본 HAMLET native 38/160, 기존 V19 native ON 37/160을 검증 후 재사용한다.
- native precision, 16 tasks × VAL scenario 0–9, inference seed 6, 실행 action interval 16, max steps 1300, 영상 저장 없음.
- OFF는 외부 READ/fusion만 우회한다. 원래 HAMLET short/session, checkpoint의 short 변환, 공동학습한 AE LoRA, FIFO WRITE와 용량, denoising은 유지한다.
- precision flip 영상 재실행, 추가 학습/loss/writer/캐시 변경은 하지 않는다.
- **OFF는 별도로 학습한 AE-only 모델이 아니다.** ON>OFF가 원본 HAMLET 대비 개선을 뜻하지 않으며, OFF>ON도 reader 단독 원인이라는 증거가 아니다.

## 기존 결과 탐색

실행 전 `runs/eval/robomme` 및 `runs/`의 comparison manifests와 실제 episode CSV 경로를 검색했다. 대상 checkpoint의 기존 rollout은 native ON과 cache-aligned ON뿐이며, 동일 checkpoint의 OFF는 없었다. 경로 이름 외 checkpoint 파일 SHA256 세 개로도 교차 검색했다. 새 실행 manifest를 포함한 52개 중 같은 가중치의 OFF는 이번 신규 run뿐이었다. 다른 버전의 `memory-off`/`archive-off` 및 offline validation은 재사용하지 않았다.

새 출력 경로가 존재하지 않음을 확인하고 `test ! -e ...` guard를 걸어 실행했다. 기존 결과는 덮어쓰지 않는다.

## 실행 전 실제 policy 검사 — PASS

검증 코드: `run_scripts/robomme/verify_read_ablation_v19.py`.

- cache-VAL episode **1355**(긴 demo, FIFO32 초과)와 **626**(demo 없음)을 성과와 무관하게 고정했다.
- 동일 관측열을 ON/OFF 각각 90개 endpoint에서 재생했다.
- demo가 아닌 모든 endpoint에서 **실제 AE를 총 46회** 실행했다. denoising stub, query별 재시딩을 사용하지 않았다.
- 호출별 지속 episode generator의 AE 직전/직후/호출 종료 상태가 두 조건에서 일치했다. 외부 memory step 자체는 CUDA 전역 RNG도 소비하지 않았다.
- 동일 관측에서 HAMLET cache, short, encoded event, FIFO bank가 일치했다. FIFO는 OFF에서도 매 호출 WRITE했고 32 events/128 tokens 한계를 유지했다.
- OFF fused 값이 현재 short와 정확히 같고, `ae_conditioning_delta_norm=0`임을 확인했다.
- AE LoRA **256개 tensor가 디스크 checkpoint와 정확히 일치**했고 비영인 학습된 `lora_B`를 확인했다. ON/OFF 이후에도 tensor hash가 유지됐다.
- 모든 과정은 frozen inference이다. teacher 관측 경로/RNG 검사이지 rollout 성공률 결과는 아니다. 생성된 행동은 달라도 정상이며, 실제 closed-loop에서는 이후 관측 및 종료 시점도 달라질 수 있다.

증거: `runs/diagnostics/v19_read_ablation_20260921/policy_check/{plan.json,events.json,completed.json,gpu_before.json}`.

## 기존 native ON의 source 차이 처리 — PASS

일반 comparator의 source 일치 검사를 완화하지 않았다. 새 sidecar `compare_read_ablation_v19.py`에서만 다음을 확인했다.

1. 기존 ON과 신규 OFF의 checkpoint/AE/short/storage 설정이 같고, 허용된 모델 차이는 `memory_off`, 설명 문자열, 이전에 없던 명시적 native precision 필드뿐이다.
2. 소스 차이는 지난 precision 작업의 policy/server/evaluator 3개와 precision helper 추가만 허용한다. 같은 이전 native 회귀 보고서와 소스 hash를 요구한다.
3. 이전 감사에 저장된 ON manifest/CSV/task manifests/완료 기록의 hash와 현재 값이 일치한다.
4. simulator 소스/scenario metadata/패키지 버전, seed, 관측·action 규칙, 정상 완료 및 episode별 기록을 검증한다.
5. 기존 ON의 실제 160 sessions/6,225 calls에서도 AE-adapted identity와 FIFO WRITE를 검증했다. READ-enabled 호출은 4,295회였다.

소스 비교 helper의 내부 hash 라벨 `aligned`는 이전 precision 코드에서 쓰던 “수정 후 소스” 필드명이다. **이번 OFF의 precision 설정은 native이며**, 두 조건 모두 native 계약을 별도로 강제한다.

기존 manifest를 수정하지 않았다. ON 37/160만 보고 pairing한 것이 아니라 episode ID, scenario seed, inference seed, instruction, task 처리 규칙이 일치하는 행들을 비교한다. 과거 환경에서 대형 3-D asset bytes와 전체 GPU driver는 기록하지 않았다는 제한은 남는다.

감사 산출물: `runs/diagnostics/v19_read_ablation_20260921/reuse_audit/`.

## 실행 명령

작업 위치: `/home/sjkim/HAMLET-Isaac-GR00T`. GPU 사용 상태를 확인했으며, preflight 및 실제 검증 전에 GPU compute process는 없었다. GPU0만 사용했다.

```bash
# 실제 LoRA / FIFO / episode RNG 검사 (새 diagnostic output 필요)
CUDA_VISIBLE_DEVICES=0 GR00T_INFERENCE_SEED=6 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/python run_scripts/robomme/verify_read_ablation_v19.py \
  --output-dir runs/diagnostics/v19_read_ablation_20260921/policy_check \
  > /tmp/v19-read-ablation-policy-check-20260921.log 2>&1

# 이전 native 회귀 증거를 사용하는 제한된 재사용 감사
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python run_scripts/robomme/compare_read_ablation_v19.py audit \
  --policy-check runs/diagnostics/v19_read_ablation_20260921/policy_check/completed.json \
  --output-dir runs/diagnostics/v19_read_ablation_20260921/reuse_audit

# 실제 아래 평가와 동일한 명령 + --preflight-only로 먼저 PASS 확인.
# 새 native OFF VAL160; baseline은 참조 재사용, 새 baseline rollout 0개.
test ! -e runs/eval/robomme/v19_native_read_off_val160_seed6 && \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python run_scripts/robomme/eval_representation_v18.py \
  --checkpoint runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072 \
  --feature-precision native --models memory-off \
  --baseline-reference runs/eval/robomme/archive_read_best1250_val_n10_seed6 \
  --tasks all --dataset val --n-episodes 10 --seed 6 \
  --output-dir runs/eval/robomme/v19_native_read_off_val160_seed6 \
  > /tmp/v19-native-read-off-val160-20260921.log 2>&1
```

코드 회귀 검사: 신규 비교 테스트와 기존 policy/core/precision 테스트 **45개 PASS**, `py_compile`, `git diff --check` PASS. 최초 새 감사 wrapper에서 `Path` 대신 문자열을 넘기는 오류를 발견·수정한 후 감사를 통과했다. policy/rollout 경로 변경이나 실패한 rollout은 없었다.

## 식별 정보

- 시작 HEAD: `009ef0cda240901c8d252c89cb96ac312b9e98e6`.
- 기존 ON evaluation ID: `c4ccf4e6f27d635a8e9207bc498fde857738caf6562dd826c96c545c40c2518e`.
- 신규 OFF evaluation ID: `4689c99a9dec4887716454535971527069791aa08467f496b2776b652b75ff44`.
- 원본 HAMLET baseline ID: `ec9b3756764d0c04e75e841055a95e74de21d936f4b70bb94ef70df97148ac25`.
- `model.safetensors` SHA256: `7624d3e21ec65425020d4c84e810ab7c8870f38387a1a99326c0eed56445e556`.
- `expert.safetensors` SHA256: `2824c8d917e7a2ca6d2bc78222283b7b3ae8346212f6961271aba18ac06bf5b4`.
- 실제 설치된 LoRA tensor 내용 hash: `a50732db0f29e7bec8ef41e76f026d47fbbc28c9a6612369fea5bc4af8a0f59e` (파일 hash와는 다른 계산).
- policy: `.venv/bin/python`, torch 2.7.1+cu128, transformers 4.51.3. Simulator: `/home/sjkim/robomme_benchmark/.venv/bin/python`, RoboMME 0.1.0, ManiSkill 3.0.0b21, SAPIEN 3.0.2, torch 2.9.1.
- 전체 설정 및 source/checkpoint/environment hash는 `reuse_audit/audit.json`, 신규 run의 `comparison_manifest.json`에 기록했다.

## 최종 3-way 결과

**native OFF VAL160 전체 완료. 외부 READ를 켜면 이 표본에서 순 3개 성공이 늘었지만, 평균 개선의 불확실성은 크고 원본 HAMLET보다 좋아진 결과는 아니다.**

신규 OFF rollout은 2026-09-21 10:00:02–10:17:54 KST, 약 17분 52초에 완료됐다. 모든 task 10/10, exit code 0, `interrupted=false`, `failures=[]`. 중간 점수에 따라 subset/설정/종료 조건을 바꾸지 않았다.

| 조건 | 성공 수 | 성공률 | 이번 새 rollout |
|---|---:|---:|---:|
|원본 HAMLET native|38/160|23.750%|0 — 기존 참조|
|동일 V19 native 외부 READ OFF|34/160|21.250%|160|
|동일 V19 native 외부 READ ON|37/160|23.125%|0 — 검증 후 재사용|

주 비교 **ON − OFF**:

- 차이 **+1.875%p**.
- OFF 실패 → ON 성공: **7개**.
- OFF 성공 → ON 실패: **4개**.
- 양쪽 성공: **30개**; 양쪽 실패: **119개**.
- 95% task 내 episode-paired bootstrap CI: **[−1.875, +5.625]%p**, 10,000회, bootstrap seed 190021.
- McNemar exact 양측 p = **0.548828125**.

CI가 0을 포함하므로 신뢰할 만한 평균 개선을 확정할 수 없다. 그렇다고 무효과가 입증된 것도 아니다. 이 조건부 구간은 −1.875%p의 악화부터 +5.625%p의 개선까지 포함한다. 반복 개발용 VAL160, 고정 16 tasks와 inference seed 6에 조건부이며 최종 일반화 성능 검증은 아니다.

### Task별 성공 수

모두 분모 10이다. 개선/악화는 OFF→ON의 성공 여부 반전 수다. `0/0`도 task를 수행하지 않았다는 뜻이 아니다.

| task | HAMLET | OFF | ON | 개선/악화 | 양쪽 성공/양쪽 실패 |
|---|---:|---:|---:|---:|---:|
|BinFill|2|2|2|0/0|2/8|
|PickXtimes|1|1|1|0/0|1/9|
|SwingXtimes|3|4|4|0/0|4/6|
|StopCube|2|2|1|0/1|1/8|
|VideoUnmask|4|4|4|0/0|4/6|
|VideoUnmaskSwap|1|1|1|0/0|1/9|
|ButtonUnmask|1|0|0|0/0|0/10|
|ButtonUnmaskSwap|1|1|1|1/1|0/8|
|PickHighlight|4|3|4|1/0|3/6|
|VideoRepick|1|2|1|1/2|0/7|
|VideoPlaceButton|4|3|4|1/0|3/6|
|VideoPlaceOrder|7|5|5|0/0|5/5|
|MoveCube|5|4|6|2/0|4/4|
|InsertPeg|0|0|0|0/0|0/10|
|PatternLock|0|0|0|0/0|0/10|
|RouteStick|2|2|3|1/0|2/7|

관측된 상쇄:

- MoveCube +2, PickHighlight/VideoPlaceButton/RouteStick 각 +1 → task 순증 합 +5.
- StopCube −1, VideoRepick −1 → 순감 합 −2.
- ButtonUnmaskSwap은 총점이 같아도 **1개 개선과 1개 악화가 상쇄**됐다.
- VideoRepick도 단순히 1개 악화가 아니라 **1개 개선과 2개 악화**다.
- 따라서 “거의 같은 점수라 READ가 아무 일도 안 한다”는 해석은 부정확하다. 일부 결과를 바꾸지만 유익한 변화만 일으키지는 않는다. 각 task 10개뿐이므로 task의 일반적 강점/약점 확정도 피한다.

### 반전 scenario 11개

아래 episode는 task 내부 0-based index이며, scenario seed와 inference seed를 모두 보관한 원본 CSV는 `report/flipped_scenarios.csv`이다. 이전 precision flip 목록과는 별개의 READ 실험 목록이며, 영상 재실행은 하지 않았다.

| task | episode | scenario seed | OFF → ON |
|---|---:|---:|---|
|StopCube|8|1020800|성공 → 실패|
|ButtonUnmaskSwap|4|1070400|실패 → 성공|
|ButtonUnmaskSwap|8|1070800|성공 → 실패|
|PickHighlight|5|1120500|실패 → 성공|
|VideoRepick|1|1090100|성공 → 실패|
|VideoRepick|2|1090200|성공 → 실패|
|VideoRepick|4|1090400|실패 → 성공|
|VideoPlaceButton|8|1100800|실패 → 성공|
|MoveCube|0|1140000|시간 제한 → 성공|
|MoveCube|2|1140200|시간 제한 → 성공|
|RouteStick|4|1160400|실패 → 성공|

종료 기록만으로 잘못된 검색, fusion, 제어 실패를 구분하지 않는다. ON의 일반 fail 104 / step_limit 19 / success 37, OFF의 fail 103 / step_limit 23 / success 34이다.

### 실제 runtime 증거와 보존 확인

| 확인 항목 | ON | OFF |
|---|---:|---:|
|완료 session|160|160|
|검증한 policy calls|6,225|6,423|
|READ-enabled calls|4,295|**0**|
|최대 READ conditioning delta norm|172.6823|**0**|
|매 호출 adapted AE identity 확인|PASS|PASS|
|매 호출 FIFO WRITE/갱신 수 확인|PASS|PASS|
|최대 bank tokens / capacity|128 / 128|128 / 128|
|missing session / identity|0 / 0|0 / 0|
|complete evidence|true|true|

- OFF의 `expert_adapted=true`, checkpoint payload hash, native 정밀도 규칙을 모든 완료 호출에서 검사했다. 별도의 실제 동일-code policy 검사에서는 LoRA 256개 tensor가 checkpoint와 일치하고 유지되는 것도 확인했다. runtime 선언만을 근거로 판단한 것이 아니다.
- passive demo 호출은 두 조건 모두 1,860회다. 실제 행동 호출은 ON 4,365회, OFF 4,563회로 다르다. 이는 행동에 따라 episode 종료 시점이 달라지기 때문이며, 동일 관측·호출 순서의 RNG 소비는 별도 실제 AE 검사에서 일치했다.
- ON/OFF 각각 manifest/driver/CSV/task manifest/runtime JSONL **50개 파일씩 hash를 검증**했다. ON은 실행 전 감사와도 동일하다. 두 조건 모두 원본 HAMLET 참조를 검증했고, 원본 결과 파일을 수정하지 않았다.
- 일반 strict comparator도 실제 실행해 `Bound inference file changed: .../eval_representation_v18.py`로 거부됨을 확인했다. 검증을 삭제하지 않고, 앞서 설명한 native 회귀 증거 기반 제한 경로에서만 비교를 통과시켰다. 이는 rollout 실패가 아니다.
- 최종 evaluator runtime summary와 독립 sidecar의 재계산이 일치한다. 종료 후 GPU compute process도 남지 않았다. CPU 회귀 45개를 다시 실행하여 PASS했다.

### 해석과 다음에 바꿀 변수 하나

확정한 사실은 **공동학습된 동일 정책에서 외부 READ 유무가 11개 scenario의 성공 여부를 바꿨고, 이번 표본의 순차이는 ON에 유리한 3개**라는 것이다. 외부 기억을 올바르게 검색한다는 의미론적 증명이나, 신뢰할 만한 평균 이득/원본 HAMLET 초과의 증명은 아니다. OFF와 원본 HAMLET의 34→38 차이를 AE LoRA 단독 효과라고 해석하지도 않는다.

다음 변수 하나만 제안한다면 **메모리 residual의 적용 배율 `alpha`**다. 현재 ON의 배율 1과 OFF의 배율 0 사이에서, **0.5 한 조건만** 사전에 정해 비교한다:

`candidate_fused = short + 0.5 * (current_fused - short)`

의도는 일부 이득을 유지하면서 손해를 줄일 여지가 있는지 보는 것이며, fusion이 원인이라고 미리 단정하는 것이 아니다. native/가중치/reader/encoder/FIFO/AE/seed를 유지하고 추가 학습 없이 새 160개만 필요하다. 성공 기준은 기존 ON 대비 paired 성공률 개선이며, CI가 0을 포함하면 여전히 불확실로 판정한다. 여러 배율을 사후 탐색하여 좋은 값만 보고하지 않는다.

**이번에는 이 변경을 구현하거나 실행하지 않았다.** 3-way 비교 보고로 작업을 종료하며 후속 학습·영상 분석도 시작하지 않았다.

### 저장된 보고서와 재생성

- 전체 분석/명령: 이 문서.
- 자동 집계 및 원시 수치: `runs/diagnostics/v19_read_ablation_20260921/report/{report.md,audit.json,tasks.csv,flipped_scenarios.csv}`.
- 실제 OFF 결과/로그: `runs/eval/robomme/v19_native_read_off_val160_seed6/`.
- driver stdout: `/tmp/v19-native-read-off-val160-20260921.log`.
- strict 거부 기록: `/tmp/v19-read-ablation-strict-comparator-20260921.log`.

완료 후 실제 실행한 paired 보고 명령은 다음과 같다. 재실행 시에는 `--output-dir`을 새 경로로 지정한다. 이 명령은 모델/시뮬레이터를 새로 돌리지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python run_scripts/robomme/compare_read_ablation_v19.py report \
  --policy-check runs/diagnostics/v19_read_ablation_20260921/policy_check/completed.json \
  --output-dir runs/diagnostics/v19_read_ablation_20260921/report
```

## 이번에 추가한 파일

- `run_scripts/robomme/verify_read_ablation_v19.py`: frozen actual-policy 경로/RNG/LoRA 진단.
- `run_scripts/robomme/compare_read_ablation_v19.py`: 제한된 이전 native 소스 bridge, 3-way 결과 및 실제 READ-off/WRITE/AE 증거 검사.
- `tests/test_compare_read_ablation_v19.py`: 비교 부호/pairing/동일 총점의 상쇄/불허 설정 변화 회귀 테스트.
- 이 문서. 기존 policy/server/evaluator/학습/가중치/캐시와 이전 작업 파일은 수정하지 않았다.
