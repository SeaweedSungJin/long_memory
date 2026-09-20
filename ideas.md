# ECHO-HAMLET: 실행 가능한 장기 사건 기억 연구 프로토타입

사용자의 목표는 **현재 상황에 필요한 과거 정보를 검색해서 매 action 생성의 condition으로 제공하는 것**이다. 이 패키지는 이를 위한 PyTorch 구현, causal 학습 루프, CVoM teacher, 온라인 메모리 관리, HAMLET 연결 helper를 제공한다.

**검증 범위:** PyTorch 단위 테스트와 합성 지연 단서 학습 실험. 실제 HAMLET checkpoint·RoboMME/RMBench·실물 로봇을 학습하거나 평가한 결과는 아니다. 기존 HAMLET repository를 자동으로 수정하지 않았다. 합성 실험의 action head는 작은 MLP이며 실제 flow matching은 별도 helper/bridge를 사용한다.

## 먼저 실행하기

Python 3.10 이상, PyTorch 2.4 이상에서 프로젝트 폴더를 현재 디렉터리로 실행한다.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python examples/delayed_cue.py --seed 7 --warmup-steps 250 --joint-steps 80 --output runs/my_experiment
```

이미 torch와 pytest가 설치되어 있으면 설치 명령 없이 나머지 명령을 실행할 수 있다. 검증 환경은 Windows, Python 3.12, torch 2.4.0, FP32 CPU다. GPU smoke는 `--device cuda`로 선택할 수 있지만 제공된 결과는 CPU 실행이다.

- `echo_memory/core.py`: event encoder, query/key/value, 검색, utility, write gate, bounded bank, 온라인 read/write.
- `echo_memory/training.py`: 과거에서 미래 순서의 replay, loss 계산, EMA teacher, sampled CVoM.
- `echo_memory/hamlet_adapter.py`: 기존 token 수를 보존하는 fusion, 실제 GR00T 구조의 flow matching loss helper.
- `echo_memory/cached_hamlet.py`: 동결한 HAMLET 특징과 실제 action head를 연결하는 bridge.
- `examples/delayed_cue.py`: 외부 데이터 없이 가능한 지연 기억 학습 실험.
- `examples/train_cached_hamlet.py`: 실제 HAMLET feature cache와 frozen action head로 새 memory 모듈 학습.
- `docs/HAMLET_INTEGRATION.md`: 실제 저장소의 train/inference 삽입 지점, checkpoint·데이터 변환 안내.
- `docs/CACHE_FORMAT.md`: 실제 feature 추출·tensor 파일 구성·cached training 명령.
- `docs/RESULTS.md`: 실행 결과와 해석의 한계.

## 여러 모듈과 loss를 붙이면 joint training이 되는가?

`sum(losses).backward()`는 **각 loss와 계산 그래프로 연결된 파라미터**에만 gradient를 전달한다. optimizer에 포함되어 있고 `requires_grad=True`여야 업데이트된다. 같은 optimizer로 학습한다는 것과 모든 loss가 모든 모듈을 학습한다는 것은 다르다.

이 구현은 다음 경로를 의도적으로 만든다.

```text
미래 action loss
  -> 기존 action expert (가중치 동결 가능, 입력 gradient는 유지)
  -> residual fusion -> retrieval attention
  -> 과거 key/value -> 과거 event encoder

detached CVoM target
  -> utility regression -> utility predictor
  -> write classification -> write gate

event reconstruction / variance / covariance
  -> event encoder (그리고 reconstruction decoder)
```

| Loss | 직접 학습하는 대상 | 이 구현의 중요한 선택 |
|---|---|---|
| action | query, key, value, event encoder, fusion; 선택적으로 action head | 과거 event를 현재 파라미터로 재계산한다 |
| reconstruction | event encoder, decoder | detached raw moment/state delta를 복원한다 |
| variance / covariance | event encoder와 그 앞의 projection | `e_t`에 직접 적용, invalid event 제외 |
| utility | utility predictor | 입력 `e, short, read`를 detach하여 auxiliary target이 representation을 왜곡하지 않게 한다 |
| write | 2-input write MLP | utility와 novelty 입력 detach; 의미가 바뀌지 않도록 분리 |
| budget | write MLP | 선택적 평균 write-rate 상한 penalty |
| hard write / eviction | 일반 backprop 대상 아님 | 명시적 비교·선택 알고리즘 |

`write_mode='soft'`이면 write 확률을 attention prior로 넣어 미래 action gradient가 write gate까지 갈 수 있다. 이것은 **실제 hard 저장 선택의 정확한 gradient가 아닌 surrogate**다. 기본 실험은 `all`로 reader/representation을 학습하고 write는 CVoM으로 지도학습한다. inference에서 `hard`로 실제 저장 여부를 결정한다. 이 차이를 숨기지 않고 hard-write 성능을 별도로 측정한다.

기본 residual output은 zero-init이다. 따라서 첫 backward에서는 output projection만 memory-path action gradient를 받고, 한 번 업데이트되어 경로가 열린 뒤 Q/K/V/event encoder도 받는다. 테스트는 이 순서를 확인한다. immediate gradient audit에만 `residual_init=1e-3` 등을 사용할 수 있다.

## 모듈 종류를 어떻게 정했는가?

| 부분 | 기본 구현 | 이유 / 변경이 필요한 경우 |
|---|---|---|
| 현재 query | 2-layer MLP | pooled short context + state로 검색 벡터 생성 |
| 시각·state 변환 | Linear projection | 서로 다른 입력 차원을 작은 공통 공간으로 변환 |
| 실행 action chunk | 작은 1-layer GRU | 실제 실행한 여러 control의 순서를 보존; single action이면 Linear로 단순화 가능 |
| event encoder | 2-layer MLP + LayerNorm | 상황·행동·전후 관측의 결합; 이미 HAMLET이 시각/단기 시간 인코딩을 담당 |
| key / value | 각각 Linear | 검색 주소와 읽을 내용의 역할 분리 |
| retrieval | normalized dot-product attention 1회 | 작은 bank에서는 추가 transformer stack 불필요 |
| fusion | token별 sigmoid gate + residual MLP | 기존 `[N,Q,D]` 조건 형상을 보존 |
| utility | 2-layer MLP | 미래 손실 감소를 예측하는 scalar |
| write | 입력 2개, hidden 32 MLP | 사용자의 `[utility, novelty]` 설계 |

추가 shallow transformer는 첫 버전의 필수 요소가 아니다. 검색된 여러 사건의 순서/관계를 함께 추론해야 하고 단일 pooled read가 병목이라는 근거가 생겼을 때 multi-query cross-attention 또는 1-layer event transformer를 추가한다. 이 프로토타입은 사건의 **내용 기반 회상**을 검증한다. bank 배열 순서를 유지하지만 explicit time/order embedding은 없으므로 반복 횟수·순서 추론은 별도의 확장과 평가가 필요하다.

권장 출발 크기는 `hidden_dim=128, key_dim=64, value_dim=128, max_slots=32~64`이며 최적값이 아니라 작은 실험을 위한 제안이다. 실제 moment/state/action 차원은 checkpoint와 dataset에서 읽는다. 단순히 module 종류를 늘리기보다 module별 gradient와 retrieval 효과를 먼저 확인한다.

## 수식과 구현 선택

원안과의 차이를 명시한다. 이 구현은 최초 실험을 위한 v1이다.

| 원안 요소 | v1 선택 |
|---|---|
| 검색 recency 항 | 사용자 수정대로 제거 |
| write gate의 e 직접 입력 | 사용자 수정대로 제거 |
| 검색 utility bias | 옵션 제공, 기본 0으로 content retrieval부터 확인 |
| fusion 바깥 LayerNorm | 기존 conditioning을 보존하는 residual branch 내부 normalization으로 변경 |
| effect prediction | event에 직접 연결되는 reconstruction으로 변경 |
| merge | 기본 제외, 의미 보존 문제를 별도 실험으로 분리 |
| CVoM 전체 미래 평균 | valid near/far future decision 샘플링 |
| write-budget 목표치 맞추기 | 선택적 상한 penalty; 필요 없는 저장을 강요하지 않음 |


관측 시점 t에서 완료된 과거 사건만 읽는다.

```text
q_t = normalize(MLP([mean(short_t), state_t]))
score_tj = dot(q_t,k_j)/temperature + beta*clip(utility_j,0,1)
read_t = Attention(score, values, mask)
z_t = short_t + token_gate(short_t,read_t)*memory_projection(read_t)
```

query/key 모두 normalize했으므로 `/sqrt(d)` 대신 temperature를 사용한다. 기본 `beta=0`: global utility가 현재 query 관련성을 압도하지 않도록 read와 storage 역할을 먼저 분리한다. 사용자의 utility-aware retrieval을 재현하려면 `utility_bias>0`로 비교한다. recency penalty는 없고 오래된 event라는 이유로 검색을 감점하지 않는다.

Null slot은 값이 0이며 empty-bank softmax를 안전하게 처리한다. soft-write 모드의 `log(weight)`도 사용한다. 완전히 빈 bank는 fusion에서 원래 tokens를 정확히 반환한다. Null slot을 선택하는 비율은 유용한 진단 지표지만 내부 LayerNorm/MLP가 있기 때문에 임의의 작은 read가 항상 작은 residual을 보장하는 것은 아니다.

```text
event_t = Encoder(short_t, pre_moment, post_moment, delta_moment,
                  state_t, delta_state, executed_action_sequence, action_present)
key_t = normalize(W_k event_t)
value_t = W_v event_t
utility_t = softplus(MLP([event_t, mean(short_t), read_t]))
novelty_t = 1-max_j cos(key_t,key_j)  # [0,2], empty bank에서는 1
write_logit_t = MLP([utility_t, novelty_t])
```

원안의 effect prediction은 별도 입력에서 결과를 예측하므로 event encoder에 gradient가 연결되지 않을 수 있었다. 여기서는 `Decoder(event_t)`가 raw frozen moment/state delta를 복원하도록 명시적으로 연결했다. **결과가 이미 encoder 입력이므로 이것은 미래 예측이 아니라 reconstruction이다.** Action의 인과적 효과를 식별했다고 주장할 수 없다.

Merge는 기본 구현에서 제외했다. 같은 행동 반복이나 모순된 상태를 평균내면 의미가 사라질 수 있기 때문이다. Eviction은 FIFO baseline과 utility+diversity 방식이 있다. 후자는 새 후보까지 B+1개를 함께 비교하므로 낮은 점수의 새 사건이 기존 기억을 무조건 밀어내지 않는다. Hard replacement 정책은 CVoM addition utility와 같은 목표는 아니며 개선 실험 대상으로 남는다.

## 학습 순서

1. **원본 HAMLET baseline 확인.** 동일 평가에서 성공률과 delay별 성능을 기록한다. checkpoint token/window/conditioning config를 일치시킨다.
2. **동결한 특징 추출.** 각 episode의 moment·short tokens·상태·실제 실행 action·endpoint를 저장한다. VLM뿐 아니라 HAMLET short memory와 moment tokens도 동결되어야 캐시가 유효하다.
3. **Reader warm-up.** all-write/FIFO로 과거 사건을 제공한다. action loss + 작은 reconstruction/variance/covariance로 representation과 reader/fusion을 학습한다. real bridge의 action expert는 기본 동결이다.
4. **EMA CVoM target 생성.** warm reader로 candidate 전후 미래 action loss를 비교한다. 같은 teacher로 key/value를 다시 만들고 candidate와 bank의 표현 버전을 맞춘다.
5. **Utility/write 학습.** action 학습을 계속하며 detached utility target과 binary write target을 추가한다. 한 번에 학습하되 loss별 gradient 경로는 위 표처럼 분리한다.
6. **Hard-write 평가.** 같은 bank budget에서 all/FIFO, novelty-only, learned write를 비교한다. 정확도뿐 아니라 중요한 사건 보존·retrieval·write rate를 측정한다. hard-write 검증 없이 soft training loss만으로 성공을 주장하지 않는다.

합성 기본값은 warm-up 250 updates, CVoM 80 updates이며 실제 VLA의 최적 schedule을 뜻하지 않는다. 모듈 수보다 데이터가 정말 과거를 필요로 하는지가 더 중요하다.

## CVoM의 정확한 의미

candidate e_i에 대해 j<i의 과거 사건만 coalition S에 넣는다. `|S|<=budget-1`이므로 S+e_i도 budget 안이다. e_i가 완료된 뒤의 valid future decisions만 near/far 구간에서 샘플링한다. 두 조건에서는 동일 S를 유지하고, i 이후의 intervening events는 **양쪽 모두** 넣지 않는다. 이 조건부 기여도 정의는 이후 사건을 계속 쓰는 rollout 평가와 다르다.

```text
signed_gain = mean_{sampled S,tau} [L_teacher(tau | S) - L_teacher(tau | S+e_i)]
utility_target = max(signed_gain,0) / fixed_utility_scale
write_target = 1[signed_gain > write_delta]
```

Signed gain은 보존하고 utility regression에만 positive clipping을 적용한다. 현재 값은 exact Shapley value가 아니며 memory coalition sampled marginal이다. Per-batch min/max normalization은 하지 않는다. 기본 loss의 absolute scale에 맞춰 `write_delta`, `utility_scale`을 validation data로 정한다.

Teacher는 eval/no_grad이고 학생의 gradient와 분리된다. 실제 flow matching에서는 noise·time을 두 조건에 동일하게 넘긴다. fallback RNG pairing은 PyTorch 기본 CPU/CUDA generator만 복원한다. Python/NumPy RNG나 별도 custom generator까지 동일하게 만드는 기능은 없으므로 외부 objective는 난수 tensor를 미리 준비한다.

MSE/flow matching surrogate가 줄었다는 것이 환경 성공률의 인과적 향상을 뜻하지 않는다. closed-loop 평가는 별도로 필요하다.

## 데이터 계약과 시간 정렬

`FeatureEpisode`는 **하나의 배치 행이 하나의 episode**다. T개의 decision과 T+1개의 observation endpoint를 가진다. 서로 다른 episode를 한 행에 이어 붙이면 안 된다. `transition_valid=False`는 padded/missing event를 제외할 뿐 episode reset을 수행하지 않는다.

| tensor | shape |
|---|---|
| short, moment | `[N,T+1,Q,D]` |
| state | `[N,T+1,S]` |
| actions | `[N,T,C,A]` 실제 실행 control들 |
| action_mask | `[N,T,C]` 연속된 유효 prefix, passive는 모두 False |
| targets | 합성 `[N,T,O]`, 실제 GR00T `[N,T,H,A]` |
| decision_mask | `[N,T]` 행동 loss가 존재하는 시점 |
| transition_valid | 선택적 `[N,T]`, event/auxiliary loss 유효 여부 |

여기서 t+1은 다음 policy call이다. 그 사이 C controls를 실행했다면 pre/post observation도 그만큼 떨어져 있어야 한다. 예측한 H-step chunk 전체와 실제 실행한 C-step prefix를 혼동하지 않는다. Demo action의 결과에 모델 예측 action을 붙이지 않는다. Visual short context는 causal이어야 하며 미래를 본 feature cache는 이 코드가 검출할 수 없다.

장기 구간 전체를 autograd로 연결하면 메모리가 커진다. 제공 replay는 작은 complete episode를 위한 참조 구현이다. 큰 데이터에서는 동결 특징을 저장하고 minibatch마다 선택한 과거 event encoder/key/value만 재계산한다. 학습된 k/v cache를 detach해서 영구 재사용하면 action loss의 과거 event gradient가 끊기고 parameter drift가 생긴다.

## 온라인 호출

```python
runtime = OnlineMemory(memory, batch_size=num_envs)
read = runtime.read(short_t, state_t)
actions = action_expert(h_t, read.z, state_t)
# 실제 실행 후 endpoint 관측과 실제 action prefix를 준비한다.
runtime.observe(moment_t, moment_next, state_next, executed_actions, action_mask)
# 다음 policy call에서 방금 완료된 사건부터 이용 가능하다.
runtime.reset(done_env_mask)  # read/observe가 완료된 경계에서 수행
```

여기서 `action_expert`는 개념적 호출이다. 실제 GR00T는 `docs/HAMLET_INTEGRATION.md`의 conditioning 경로를 사용한다. Auto-reset 환경에서는 terminal observation과 reset observation을 구별해야 한다. terminal endpoint가 없으면 해당 transition을 invalid 처리하고 메모리를 reset한다. rollout 중 model parameter를 바꾸면 stored key/value가 stale해지므로 초기화하거나 raw history에서 재구성한다.

## 초기 실험의 판단 기준

- no-memory와 shuffled-memory가 성능을 떨어뜨리는가?
- original HAMLET window 밖의 cue가 필요한 경우에 개선되는가?
- hard write에서도 action loss와 성공률이 유지되는가?
- 동일 B에서 FIFO·uniform·novelty-only보다 중요한 사건을 더 보존하는가?
- oracle relevant event를 넣으면 잘되는가? 안 되면 writer보다 reader/fusion/representation을 먼저 수정한다.
- 여러 seed, 더 긴 delay, distractor, 반복 순서, 작은 budget을 따로 검증했는가?

## 참고 소스

- HAMLET source: https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T
- HAMLET paper: https://arxiv.org/abs/2510.00695
- PyTorch autograd: https://docs.pytorch.org/docs/stable/notes/autograd.html

이 패키지는 사용자 제안에서 출발한 연구용 참조 구현이다. 재현 가능한 engineering feasibility 검증과 실제 로봇 성능 검증을 구분한다.
