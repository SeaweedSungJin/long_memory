# ECHO-CVoM v1: 구현 범위와 재현성 계약

작성 기준: 2026-09-21. 이 문서는 [알고리즘 제안.pdf](<알고리즘 제안.pdf>)의 ECHO-HAMLET 제안과 현재 코드를 대조한 **설계·실행 설명**이다. 학습 완료나 closed-loop 성능 향상을 선언하는 결과 보고서가 아니다. 실제 실행 경로, 완료 상태, 측정 결과는 후속 실행 기록에 별도로 추가한다.

## 1. 무엇을 바꾸는가

ECHO-CVoM은 기존 HAMLET의 short memory와 V19 외부 reader를 유지하면서, **이미 실행한 행동의 관측된 결과를 저장 event에 더하고, 모든 retained slot과 새 후보의 미래 행동 예측 기여도를 학습해 저장·퇴출을 결정**한다.

기준 초기화는 `runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072`이다. 원본 HAMLET은 `checkpoints/author_hamlet_robomme/checkpoint-60000`, 캐시는 `runs/long_memory/cache_full1600_v1`이다. 기존 checkpoint, cache, 결과 파일은 수정하지 않는다. ECHO checkpoint variant는 `echo_cvom_v1`이며, 이전 `cvom_admission_v1` writer와 호환되는 이름 바꾸기가 아니다.

| 항목 | 이전 oldest-admission 실험 | ECHO-CVoM v1 |
| --- | --- | --- |
| actor | V19 representation/AE LoRA 고정 | stage 1에서 effect·reader/fusion·AE LoRA 추가 학습 후 고정 |
| 저장 event | 관측 short 기반 event | 관측 event + 완료된 action-effect residual |
| 대상 | 현재 candidate의 admission | 기존 모든 slot + candidate에 공통 utility predictor 적용 |
| full-bank 결정 | KEEP 또는 고정 oldest 교체 | candidate gate 후 전체 slot retention 경쟁, 최저 점수 slot 교체 |
| counterfactual | oldest replacement 중심 비교 | `J(S) - J(S ∪ {e})`의 정확한 한-slot 추가 비교 |
| context 범위 | 이전 실행에서는 overflow 중심 TRAIN 280개 | 기본적으로 모든 eligible TRAIN episode, episode당 최대 2 contexts |
| supervision | confident target 수가 매우 적었던 실험 | signed continuous target 전부 회귀, confidence는 BCE에만 적용 |
| 병합 | 없음 | 보호 조건이 있는 선택 기능, 기본 OFF |

이전 실험의 confident TRAIN target은 single 45/280, coalitional 61/280이었다. 이 작은 유효 표본과 제한적인 context 분포는 강한 일반화 주장을 뒷받침하지 못한다. 해당 수치는 [기존 실행 기록](CVOM_ADMISSION_STUDY.md)에 남기며 ECHO 성능으로 인용하지 않는다.

## 2. 모듈과 tensor 계약

현재 parent의 설정은 short window `K=4`, short/moment token 수 `Q=4`, HAMLET feature 폭 `F=2048`, memory 폭 `D=256`, attention head 수 4, event budget `B=32`다. 아래 shape의 첫 축은 online에서 한 session, 즉 batch 1이다.

| 모듈/데이터 | 실제 형태·차원 | 역할 |
| --- | --- | --- |
| HAMLET short/moment | `[1,4,2048]` | 기존 visual-language/단기 문맥; backbone 및 HAMLET short Transformer 고정 |
| 관측 state / 정규화 action | 각각 128차원 | 기존 processor 공간; 실제 로봇 control은 8차원이며 정규화·padding 후 사용 |
| parent event encoder | short LN/linear, state projection, 시간 encoding | query와 base event `[1,4,256]` 생성 |
| `CompletedEffectEncoder` | 입력 4352 → LN → 128 → SiLU → 128 → SiLU | 관측 pre/post moment 평균, state 차이, 실행 control 평균으로 hidden 구성 |
| effect output / auxiliary decoder | 128 → 1024, 128 → 2304 | output을 `[1,4,256]` residual로 reshape; 별도 완료-transition 재구성 |
| `EchoBank` | tokens `[1,N×4,256]`, `0≤N≤32` | 최대 128개 memory token; timestamp/demo/count/event lineage는 sidecar metadata |
| 기존 reader/fusion | 256폭 multi-head cross-attention, gate, 256→2048 projection | 전체 bank를 읽어 원래 short에 residual을 더함; 출력 `[1,4,2048]` |
| `SlotUtilityMLP` | 입력 `[N+1,776]` → LN → 128 → SiLU → 128 → SiLU | 모든 slot 및 새 후보에 같은 predictor 사용 |
| predictor heads | 각각 128→1 | signed utility, write logit 및 sigmoid probability |
| AE LoRA | rank 8, alpha 16 | 기존 AE attention projection에 설치된 adapter만 학습 가능 |

숫자는 현재 parent 설정의 실제 값이다. `776 = 3D+8`, effect 입력은 `2F+state_dim+action_dim`, decoder target은 `F+state_dim+action_dim`이다. effect output과 utility/write heads는 zero initialization이다. Effect zero initialization은 parent 저장 표현을 시작점으로 보존하지만, stage 1 이후의 actor가 원본 V19와 동일하다는 뜻은 아니다.

Utility 입력의 세 vector는 해당 slot 평균, **현재** query 평균, 현재 실제 bank 평균이다. 추가 8개 scalar는 last/first age의 `log1p`, recency, novelty, query cosine, demo 여부, merge count, candidate 여부다. Task 이름, episode ID, 정답 action, 성공 여부는 predictor feature가 아니다. `event_ids`는 순서·출처 검증용이며 숫자 자체를 predictor에 넣지 않는다.

Reader는 parent 그대로 **전체 token attention**을 한다. 별도 utility-biased Top-K retrieval이나 utility를 attention logit에 더하는 기능은 없다. Utility는 저장·퇴출 결정에만 사용한다. Fusion도 PDF의 수식을 새로 구현한 것이 아니라 기존 zero-read-residual 계약을 가진 gated residual reader다.

구현: [core](run_scripts/robomme/echo_cvom_core.py), [parent representation](run_scripts/robomme/representation_core_v18.py), [reader](gr00t/long_memory/recurrent_v7.py).

## 3. 인과적 action-effect와 READ-before-WRITE

여기서 `t`는 원본 영상의 매 raw frame이 아니라 **관측 endpoint 인덱스**다. 실제 frame 번호는 별도로 보존한다. 실행 시 action chunk의 앞 16 control을 실행한 뒤 다음 endpoint 관측이 들어온다.

```text
이전 endpoint → 실제 control 실행 → 현재 endpoint t 관측
                                       │
                         [t-1,t] 완료 transition으로 e_t 구성
                                       │
                    기존 bank(<t) READ → 현재 행동 조건 feature 확정
                                       │
                                 e_t WRITE 결정
                                       │
                         다음 endpoint t+1부터 e_t를 READ 가능
```

따라서 현재 관측에서 완성된 effect event는 **한 endpoint 늦게** action query에 읽힌다. 저장값은 `base_event_t + effect(pre_moment, post_moment, state_delta, executed_controls)`지만 **현재 query에는 effect residual을 더하지 않는다**. 새 저장 event를 현재 query가 즉시 읽는 self-read 경로도 없다. Online의 WRITE 호출이 AE 실행 호출보다 앞서더라도 AE에 전달할 fused feature는 이미 이전 bank로 확정되어 있다.

Offline에서는 `actions[:t]`, action mask, transition validity와 관측 prefix만으로 완료 effect를 만든다. 현재 시점의 미래 target action chunk를 event encoder에 넣지 않는다. Online에서는 모델의 미실행 예측 chunk가 아니라 환경이 보고한 **실제 실행 control**을 이전 raw state와 원본 processor로 정규화해 사용한다. Control은 유효 mask로 평균되므로 이 adapter는 chunk 내부 행동 순서를 완전히 보존하는 sequence encoder가 아니다.

Demo는 관측-only event다. Demo 구간 및 demo→execution 경계에는 실제 완료 control이 있다고 가정하지 않으며 effect residual을 0으로 둔다. Demo short/state/time 표현은 기존 방식대로 저장할 수 있다. 처음 관측도 존재하지 않는 이전 transition을 만들지 않는다.

`memory-off`는 READ만 끈다. WRITE, completed-control 처리, short cache, episode RNG 및 reset 경로는 유지한다. 출력은 원래 short feature와 같아야 한다. 이 계약은 [runtime policy](run_scripts/robomme/policy_echo_cvom.py), [native replay 검사](run_scripts/robomme/verify_echo_cvom.py)에서 확인한다. Native replay는 teacher 관측을 사용하는 integration test이며 simulator 성공률 평가가 아니다.

## 4. 저장·퇴출·선택적 병합

기본 설정은 capacity 32, `min_fill=4`, candidate probability threshold 0.5, recency/diversity 가중치 각각 0.05다.

1. FIFO 모드는 utility와 관계없이 빈 공간에 append하고, 가득 차면 oldest를 교체한다.
2. Learned 모드는 최초 4개 event까지 강제 append한다. 이후 candidate의 write probability가 threshold 미만이면 KEEP이다.
3. 공간이 있으면 gate를 통과한 candidate를 append한다.
4. 가득 차면 현재 query를 사용해 기존 모든 slot과 candidate를 다시 평가한다. `R = max(0, predicted_utility) + 0.05×recency + 0.05×novelty`다.
5. Candidate retention이 기존 최소 retention보다 **엄격하게 클 때만** 해당 slot을 교체한다. 동점은 KEEP이다. Utility의 단위는 stage 2의 TRAIN-only 정규화 단위이며 raw future-loss 감소량과 같지 않다.

Utility는 PDF의 slot별 EMA 누적값이 아니라 현재 context에서 다시 계산하는 값이다. novelty와 recency도 현재 실제 bank에 대해 계산한다. Hard 선택은 미분하지 않지만, 남는 event tensor 자체를 detach하지 않으므로 stage 1의 이후 action loss가 과거 event encoder/effect adapter까지 전파될 수 있다.

선택적 merge는 구현되어 있으나 `merge_threshold=None`으로 **기본 OFF**다. 켰을 때도 full-bank admission/retention 조건을 통과한 뒤, 시간순으로 인접한 두 event 중 cosine 조건을 만족하는 쌍만 병합한다. 두 event는 같은 demo/execution phase여야 하고, frame gap은 기본 32 이하, 합산 count는 기본 2 이하이어야 한다. Token은 count-weighted 평균, first/last frame·count·정렬된 source event ID는 함께 갱신한다. Demo와 execution을 섞거나 시간순서를 바꾸지 않는다.

이 설정을 바꾸는 `--merge-threshold`, `--min-fill`, `--recency-weight`, `--diversity-weight`는 `warmup`에서만 지정한다. 이후 labels/writer/runtime은 checkpoint의 architecture 설정을 상속하며 별도 override로 서로 다른 storage 의미를 섞지 않는다.

이것은 PDF의 arbitrary nearest-key merge, probability-weighted fractional count, 누적 utility EMA와 동일하지 않다. 출처/count/시간순서 보호가 표현의 의미 보존을 증명하지도 않는다. 시간 encoding까지 평균되므로 반복 횟수·순서 정보가 약해질 수 있다. Merge ON은 별도 ablation으로 검증해야 하며 현 기본 실행의 성능 요소로 주장하지 않는다.

## 5. 학습 1단계: causal actor warm-up

[trainer](run_scripts/robomme/train_echo_cvom.py)의 `warmup`은 기존 V19 full-coverage query planner와 flow objective를 재사용한다.

- 학습: event/effect encoder, 외부 reader/fusion, AE LoRA. Utility manager는 고정한다.
- 고정: 원본 VLM/HAMLET backbone·short Transformer·AE base weights.
- 메모리 수집: FIFO. 각 sampled query마다 그 query 이전 전체 관측 prefix를 현재 trainable weights로 다시 인코딩한다.
- 목적함수: `L_flow(prefix weight=1, tail weight=0.25) + 0.01×L_completed_reconstruction`.
- 기본 optimizer: AdamW, memory learning rate `3e-5`, AE LoRA `3e-6`, weight decay 0.01, gradient clipping 1.0.
- 원본 AE action horizon은 50이며 배포 실행 prefix는 16이다. Tail weight 0.25는 나머지 valid target을 완전히 버리지 않는 V19 목적함수다.

현재 고정 cache/plan에는 TRAIN 1,276 episodes, eligible action queries 24,286개가 있다. `epochs=1`, query batch 4이면 **6,072 optimizer updates**이며 마지막 batch는 2 queries다. 여기서 full coverage란 **cache의 모든 eligible TRAIN action query를 한 번 사용하는 것**이지, 원본 영상의 모든 raw frame을 action-loss query로 쓰는 뜻이 아니다. Passive/demo endpoint는 action query에서 제외되지만 causal memory prefix에는 들어간다.

Cache-VAL 324 episodes는 학습에서 제외한다. 기존 캐시의 VideoUnmask 계열 두 블록은 metadata만으로 원래 task 이름을 확정할 수 없어 `VideoUnmaskFamily_block400/1500`으로 구분하는 기존 resolver를 유지한다. 이 group 이름은 sampling/report용이고 모델 입력이 아니며, simulator task 이름을 임의로 덮어쓰지 않는다.

Reconstruction은 hidden에서 이미 관측된 moment 변화, state 변화, 평균 control을 복원하는 block-normalized MSE다. 입력에 post observation이 있으므로 **미래 transition을 예측하는 world model이 아니다**. PDF의 pre-action effect prediction, variance/covariance regularization을 구현했다고 주장하지 않는다. Effect output residual은 zero-init이며 action loss를 통한 학습 경로도 가진다.

Warmup validation은 cache-VAL offline action 오류다. 기본 `val_per_task=2`, noise 2이며 simulator VAL160과 구분한다. Best checkpoint를 VAL로 고르는 대신 고정 마지막 epoch checkpoint를 사용한다. Smoke의 `--stop-after-steps`는 full-epoch 완료가 아니다.

## 6. Frozen teacher: 모든 slot에 대한 additive CVoM

[teacher](run_scripts/robomme/echo_cvom_teacher.py)는 stage 1 또는 stage 2의 **불변 checkpoint snapshot**을 고정해 label을 만든다. Teacher는 EMA가 아니라 명시적으로 저장·hash된 한 snapshot이다.

### Context 및 target 표집

기본 `--train-context-limit 0`은 eligible TRAIN episode 전체를 포함한다. 각 episode에서 기본 2개의 서로 다른 context를 고른다. Demo와 execution이 모두 eligible이면 각각 하나를 보장하고, 가능한 경우 pre-capacity demo와 post-capacity execution을 고른다. 한 phase만 있으면 capacity 전후 또는 temporal halves를 사용한다. `--contexts-per-episode 1`은 작은 smoke 용도로 지원하며 eligible demo를 우선하는 편향이 있다.

각 context 이후 HAMLET short window를 벗어난 유효 future action query가 최소 2개 있어야 eligible이다. 정확한 조건은 future endpoint `q > t + K`다. Action 정답 값·성공 여부로 context를 선별하지 않는다. Task는 limit 적용 시 균형 표집 및 보고에만 사용한다. 기본 cache-VAL budget은 **64 contexts**이며, 이는 64 episodes나 simulator VAL160이라는 뜻이 아니다.

실제 캐시의 기본 preflight 결과는 TRAIN **1,267 episodes / 2,534 contexts**, cache-VAL **60 episodes / 64 contexts**다. TRAIN 중 demo context는 748개, endpoint index 32 이상은 280개다. 미래 query 조건이 안 되는 9개 TRAIN episode는 CVoM label에서만 제외되고 stage 1 action 학습에서는 제외되지 않는다. Actor 호출 상한은 한 label round당 332,544회다.

시점 `t`의 actual bank는 관측 `[0,t)`만 replay하여 얻는다. 여기에 현재 candidate `e_t`를 더한 known pool에서 **현재 candidate를 항상 포함**하고 기존 slot 중 균등 표집하여 최대 4개 targets를 고른다. 즉, 실제 수는 `min(4,N+1)`이다. Oldest 하나에 supervision을 몰지 않는다. Predictor feature는 각 target에 대해 **현재 실제 bank와 현재 query에서의 `manager.score(...)` 결과를 그대로 저장**한다. Coalition bank나 future query로 predictor feature를 대체하지 않는다.

### 정확히 무엇을 비교하는가

Target `e`를 제외한 known pool에서 coalition `S`를 고르고 다음을 비교한다.

```text
signed_gain(S, future_query, noise) = J(S) - J(S ∪ {e})
raw_signed_mean = mean(all signed_gain draws)
positive_utility = max(0, raw_signed_mean)
```

`|S|≤B-1`이므로 추가한 branch도 budget B를 넘지 않는다. 두 branch의 차이는 **정확히 한 event의 추가**이며 token 수가 같지 않다. Fixed oldest replacement가 아니다. Old target의 coalition에는 현재 candidate가 들어갈 수 있다. 그것은 이미 시점 t에 관측된 known item이기 때문이다. 두 branch 모두 원래 시간순서와 동일한 token 내용을 유지한다.

기본 target당 coalition 4개를 사용한다. 첫 draw는 가능한 최대 cardinality의 subset이고, 나머지는 `0…min(B-1,other_count)`에서 cardinality를 균등하게 고른 뒤 해당 크기의 subset을 균등 표집한다. Empty coalition과 반복 subset을 허용한다. 이 largest-subset-first 혼합 표집은 **정확한 Shapley 값이나 unbiased Shapley estimator가 아니다**.

Future query 2개와 noise 2개를 사용하며, 각 pair는 동일 future observation/query, GT action target, noise 및 timestep을 공유한다. **시점 t 이후의 memory write는 어느 branch에서도 하지 않는다.** 따라서 label은 frozen actor가 주어진 미래 관측에서 고정된 memory content를 읽을 때의 conditional predictive utility다. 이후 policy/action/환경을 분기해 얻은 causal rollout return이 아니며 terminal success label도 아니다.

현재 budget의 actor 호출 상한은 context당 `4 targets × 4 coalitions × 2 future queries × 2 noises × 2 branches = 128`회다. 초기 bank의 target 수가 적으면 실제 호출은 줄어든다. 이 비용에는 prefix encoding/read 비용도 추가되므로 wall-clock을 호출 수만으로 단정하지 않는다. 실제 actor call count를 결과에 기록한다.

Signed mean, positive-clipped utility, 개별 gains/branch losses, noise seeds, coalition index, feature, draw/noise spread를 모두 보존한다. Positive clipping은 **각 draw 전이 아니라 signed mean을 구한 뒤** 한다. Noise spread는 조건부 변동성의 heuristic이지 episode-level 신뢰구간이 아니다.

## 7. 학습 2단계와 learned-bank refresh

`writer`는 저장된 TRAIN features/labels만으로 CPU에서 utility manager를 학습한다. Effect encoder·reader/fusion·AE LoRA를 포함한 actor는 고정한다. 저장 시 모든 non-manager tensor를 source checkpoint와 비교하고 AE payload도 source에서 가져온다. 따라서 **같은 stage 2 snapshot의 FIFO와 learned storage 비교에서는 actor가 같다**.

Regression target은 `asinh(raw_signed_mean / scale)`이며 scale은 TRAIN의 median absolute gain과 작은 floor로 정한다. Cache-VAL로 scale/threshold를 맞추지 않는다. Smooth-L1 regression에는 음수·0 근처·불확실한 label을 포함한 **모든 continuous labels**가 들어간다. Noise에 따른 soft weight는 최소 0.1로 제한한다.

Write BCE에만 `abs(raw_gain) > max(1.96×noise_mean_std, 1e-6)`인 confident subset을 사용하고, target은 `raw_gain > 1e-6`이다. 기본 총 loss는 weighted regression + `0.25×BCE`다. 이 confidence 규칙으로 regression label의 80% 이상을 버리는 구조가 아니다. Confident BCE 표본이 0이면 그 batch의 BCE를 0으로 하고 regression은 계속한다.

기본 writer는 1,000 updates, batch 64, learning rate `3e-4`, AdamW weight decay 0.01, gradient clipping 1.0이다. 고정 마지막 update를 사용한다. `write_accuracy`, utility correlation, predicted admission rate는 **offline storage prediction 지표**이며 로봇 성공률이 아니다.

Stage 1 snapshot에서 `labels`를 만들면 FIFO로 prefix bank를 수집한다. Stage 2 snapshot을 다시 `labels --checkpoint ...`에 주면 **그 frozen learned writer로 같은 observed prefix를 replay**해 자신의 bank-state 분포에서 label을 갱신한다. 새로운 labels로 다시 `writer`를 실행할 수 있다. 이때 actor는 그대로 고정하고 snapshot/source/protocol이 다른 label을 섞지 않는다. 이것은 observed-trajectory bank-state refresh이며 새로운 simulator rollout이나 joint policy fine-tuning은 아니다. 자동 EMA 갱신도 아니다.

## 8. PDF와 일치하는 부분, 아직 다른 부분

| PDF의 제안 | 현재 v1 구현 |
| --- | --- |
| short + action-effect episodic memory | 완료 transition residual을 저장 event에 추가 |
| coalition 추가에 따른 미래 action-loss 기여도 | additive teacher, all-slot targets, matched query/noise |
| 경량 online utility/write policy | slot-wise MLP로 distillation, teacher는 online에서 호출하지 않음 |
| utility/recency/diversity eviction | 현재 context에서 모든 slot 재평가; fixed oldest eviction 아님 |
| utility-biased Top-K retrieval | **미구현**; 기존 전체-bank attention 유지 |
| key/value/utility EMA 및 probability-weighted merge | 별도 K/V/utility EMA 없음; optional count-weighted adjacent token merge만 구현 |
| scheduled fusion gate floor | **미구현**; 기존 parent gate를 학습 |
| pre-action effect prediction + variance/covariance loss | 완료-transition reconstruction만 구현; 해당 regularizer 없음 |
| write-budget target/loss | **미구현**; hard capacity, min-fill, probability gate만 사용 |
| EMA teacher + 마지막 joint fine-tuning | **미구현**; frozen checkpoint teacher + actor-fixed writer stage |
| 다중 seed와 광범위 ablation | 별도 수행 필요; 현재 고정 VAL160은 연구 개발용 패널 |

따라서 이 코드는 PDF 전체의 동일 재현이나 완성된 논문 실험이 아니라, 핵심 action-effect/additive-CVoM 저장 가설을 기존 substrate 위에서 검사하는 **명시적으로 축소된 v1**이다.

## 9. 평가와 재현성

[평가 wrapper](run_scripts/robomme/eval_echo_cvom.py)는 고정 development **VAL160**만 사용한다: 16 tasks × task당 10 episodes, dataset `val`, seed 6, action prefix 16, max episode steps 1300, feature precision `native`.

- `memory`: stage 2의 learned storage.
- `fifo`: 같은 checkpoint의 effect/reader/AE/manager를 로드하고 저장 결정만 FIFO로 바꾼다.
- `memory-off`: 같은 storage update를 계속 수행하면서 READ만 끄는 선택적 ablation.
- `baseline`: 기존 완료 결과에서 원본 HAMLET을 strict reference contract로 재사용한다. Adapter가 있는 memory-off를 original baseline이라고 부르지 않는다.

ECHO warmup은 actor를 바꾸므로 과거 V19 FIFO 결과를 **same-actor FIFO control로 재사용할 수 없다**. Original HAMLET 대비 차이는 effect·reader·AE warmup과 storage를 함께 포함한다. Learned storage 자체의 기여는 같은 ECHO actor의 FIFO와 비교해야 한다. Source/benchmark/base identity가 맞지 않으면 baseline provenance 검증을 완화하지 않는다.

평가는 parent/checkpoint payload, 실행 source, base files, benchmark/settings를 manifest에 묶고 runtime WRITE/READ decision counters를 수집한다. 결과 보고에는 성공률뿐 아니라 paired 변화, uncertainty, write rate/KEEP/REPLACE/MERGE, latency 및 peak memory를 함께 확인해야 한다. VAL160을 final TEST 결과라고 부르거나 한 seed의 우세를 일반적 향상으로 단정하지 않는다.

Warmup resume는 동일 stage/plan/source/settings 및 optimizer/RNG를 요구하고 **새 output directory**로 이어간다. Label preparation은 동일 frozen snapshot/source/plan/protocol에서 완료 context를 재사용한다. Inner context의 episode/event/future, snapshot, slot 및 finite feature/label도 검사한다. Writer는 현재 interrupted training의 exact resume를 지원하지 않으며 새 output이 필요하다. Intermediate writer checkpoint는 완료 checkpoint가 아니고, 원래 warmup이 불완전한 checkpoint도 본 실행용 완료 actor로 승격하지 않는다.

Checkpoint/label payload는 SHA256으로 검사한다. Selected cache episode는 대형 payload 전체를 다시 hash하지 않고 **path/size/mtime_ns/ctime_ns 파일 signature**를 protocol에 기록하여 변경 여부를 확인한다. 이것은 **cryptographic episode-content hash가 아니다**. 따라서 문서상 cache fingerprint나 file-stat 검증을 전체 데이터의 암호학적 불변성 증명으로 확대 해석하지 않는다. Checkpoint/label hash와 state-before/after를 보존하고, 장시간 실행 도중 source/cache를 수정하지 않는다. 실행 전에 preflight와 현재 구현의 검증 범위를 확인한다.

기존 native/cache backbone feature 불일치는 이 작업에서 새로 해결하지 않았다. Effect 입력은 양쪽에서 moment의 BF16 경계를 맞추지만 원래 reader/backbone precision 정책을 전역 변경하지 않는다. **같은 native 입력을 넣었을 때** online/offline effect·query parity를 검증하는 것과, cached/native VLM feature가 항상 bitwise 동일하다는 주장은 다르다. 이 차이는 남아 있는 실험 한계다.

수동 keyframe, task-specific oracle memory, 성공 여부, object/subtask 정답 metadata를 writer label로 사용하지 않는다. Demo mask·관측 시간·실행 control은 online에서 인과적으로 이용 가능한 입력이다. 미래 GT/query는 frozen teacher label 계산에만 들어간다. 관측된 action effect 역시 통제된 인과효과 추정과 동일하지 않다.

## 10. 실행 명령

아래 경로는 재현 예시이며 실행 완료를 의미하지 않는다. Repo root에서 실행한다. 기존 run이 있으면 덮어쓰지 말고 새 output 경로를 선택한다. GPU 실행 전 사용 현황을 확인하며 다른 작업을 종료하지 않는다.

### 10.1 단계별/일괄 wrapper

[run_echo_cvom.sh](run_scripts/robomme/run_echo_cvom.sh)의 기본 training root는 `runs/long_memory/echo_cvom_full_v1`, 평가 root는 `runs/eval/robomme/echo_cvom_full_v1_val160_seed6`다. 명령은 다음과 같다.

| 명령 | 동작 |
| --- | --- |
| `preflight` | stage 1 full-epoch 계획만 확인 |
| `stage1` | FIFO actor warmup |
| `labels0` | 완료된 stage 1 frozen teacher의 FIFO-bank labels |
| `stage2` | labels0로 첫 writer 학습 |
| `labels1` | 완료된 stage 2 frozen teacher의 learned-bank labels |
| `refresh` | labels1로 actor-fixed writer 재학습 |
| `verify` | 최종 checkpoint의 native teacher-observation integration 진단 |
| `eval-preflight` / `eval` | fixed VAL160 평가 준비 확인 / 실제 simulator 평가 |
| `all` | 위 학습 2 rounds와 refresh, 진단, 평가까지 순서대로 실행 |
| `monitor` | training root를 읽는 독립 TensorBoard monitor |

```bash
bash run_scripts/robomme/run_echo_cvom.sh preflight

# 실제 full workflow: warmup + labels0/stage2 + labels1/refresh + verify + eval
CUDA_VISIBLE_DEVICES=0 bash run_scripts/robomme/run_echo_cvom.sh all

# 다른 terminal/session에서 모니터링
bash run_scripts/robomme/run_echo_cvom.sh monitor
```

기본 final phase는 `refresh`다. 첫 writer만 별도로 진단/평가하려면 `ECHO_FINAL_PHASE=stage2`를 `verify`, `eval-preflight`, `eval` 명령에 준다. `all`은 이 변수와 관계없이 labels1/refresh 단계까지 수행한다. Wrapper의 기본 `ECHO_MODELS="fifo memory"`는 **새 rollout 320개**와 strict-reused 원본 baseline이다. 추가 READ-off는 `ECHO_MODELS="fifo memory memory-off"`로 명시한다.

`ECHO_RUN_DIR`, `ECHO_EVAL_DIR`, `ECHO_CACHE_DIR`, `ECHO_PARENT`, `ECHO_DEVICE`, `ECHO_SEED`, `ECHO_WRITER_UPDATES` 등으로 실행 경로·설정을 지정할 수 있다. 동일 root 중복 실행은 lock으로 거부한다. 완료된 각 단계는 source/cache/parent/seed 및 고정 마지막 checkpoint 검증 후에만 재사용하며, partial output을 무조건 건너뛰지 않는다. Source가 바뀐 실험은 새 root로 시작한다. Wrapper monitor의 기본 포트는 **6008**이고 `ECHO_TB_PORT`로 바꿀 수 있다.

### 10.2 개별 Python CLI

다음은 wrapper와 독립적인 예시 root `echo_cvom_v1`을 사용한다. Wrapper가 사용하는 `stage1/labels0/stage2/labels1/refresh` 경로와 아래 예시 경로를 섞지 않는다.

```bash
# 1) CPU/read-only 계획 확인
.venv/bin/python run_scripts/robomme/train_echo_cvom.py warmup \
  --output-dir runs/long_memory/echo_cvom_v1/warmup \
  --epochs 1 --query-batch-size 4 --activation-checkpointing \
  --preflight-only

# 2) Full TRAIN warm-up: 기본 parent/cache, 1 epoch
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python run_scripts/robomme/train_echo_cvom.py warmup \
  --output-dir runs/long_memory/echo_cvom_v1/warmup \
  --epochs 1 --query-batch-size 4 --activation-checkpointing

# 3) Frozen teacher 계획 확인 후 label 준비
.venv/bin/python run_scripts/robomme/train_echo_cvom.py labels \
  --checkpoint runs/long_memory/echo_cvom_v1/warmup/checkpoint-006072 \
  --output-dir runs/long_memory/echo_cvom_v1/labels_fifo \
  --contexts-per-episode 2 --train-context-limit 0 --val-context-limit 64 \
  --targets-per-context 4 --coalitions 4 --future-samples 2 --noise-samples 2 \
  --preflight-only

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python run_scripts/robomme/train_echo_cvom.py labels \
  --checkpoint runs/long_memory/echo_cvom_v1/warmup/checkpoint-006072 \
  --output-dir runs/long_memory/echo_cvom_v1/labels_fifo \
  --contexts-per-episode 2 --train-context-limit 0 --val-context-limit 64 \
  --targets-per-context 4 --coalitions 4 --future-samples 2 --noise-samples 2

# 4) CPU writer learning: 모든 continuous TRAIN labels 사용
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python run_scripts/robomme/train_echo_cvom.py writer \
  --checkpoint runs/long_memory/echo_cvom_v1/warmup/checkpoint-006072 \
  --labels-dir runs/long_memory/echo_cvom_v1/labels_fifo \
  --output-dir runs/long_memory/echo_cvom_v1/writer \
  --writer-updates 1000 --device cpu

# 5) Native teacher-observation/실제 AE integration 진단; 성공률 평가 아님
CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_scripts/robomme/verify_echo_cvom.py \
  --checkpoint runs/long_memory/echo_cvom_v1/writer/checkpoint-001000 \
  --output-dir runs/diagnostics/echo_cvom_v1/native_replay

# 6) 동일 actor의 learned/FIFO VAL160. 원본 baseline은 strict reference 재사용
.venv/bin/python run_scripts/robomme/eval_echo_cvom.py \
  --checkpoint runs/long_memory/echo_cvom_v1/writer/checkpoint-001000 \
  --models memory fifo --output-dir runs/eval/robomme/echo_cvom_v1_val160 \
  --preflight-only

CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_scripts/robomme/eval_echo_cvom.py \
  --checkpoint runs/long_memory/echo_cvom_v1/writer/checkpoint-001000 \
  --models memory fifo --output-dir runs/eval/robomme/echo_cvom_v1_val160
```

READ-off가 필요하면 같은 평가 계획에 `--models memory fifo memory-off`를 명시한다. Default evaluator는 세 역할을 모두 포함하므로 두 fresh roles만 원하면 위처럼 `--models memory fifo`를 지정한다. `--report-only --output-dir ...`는 완료된 평가 결과를 집계한다. Init/smoke checkpoint 예외 옵션을 최종 학습 성능 주장에 사용하지 않는다.

Learned-bank refresh는 위 label/writer 명령에서 source checkpoint를 `writer/checkpoint-001000`으로 바꾸고 각각 `labels_learned`, `writer_refresh`처럼 **새 디렉터리**를 지정한다. Teacher가 stage 2임을 보고 learned replay mode를 선택한다. 이전 FIFO labels를 덮어쓰지 않는다.

## 11. Live monitor

현재 제공되는 monitor는 training JSONL을 읽어 별도 TensorBoard event session으로 미러링한다. 모델이나 GPU를 로드하지 않으며 monitor를 종료해도 학습은 계속된다.

```bash
.venv/bin/python run_scripts/robomme/monitor_long_memory.py \
  --logdir runs/long_memory/echo_cvom_v1 \
  --host 127.0.0.1 --port 6006 --poll-seconds 2
```

원격 접속은 VS Code port forwarding 또는 SSH의 `-L 6006:127.0.0.1:6006`을 사용한다. 브라우저는 `http://127.0.0.1:6006`에서 Scalars/Time Series를 열고 Step 축, Reload data를 사용한다. TensorBoard 의존성이 없으면 [monitor requirements](run_scripts/robomme/long_memory_monitor_requirements.txt)를 해당 Python 환경에 설치한다.

`metrics.jsonl`, `status.json`, `*.launch.log`, label preparation의 `protocol.json`/`manifest.json`, 평가의 `comparison_manifest.json`과 runtime diagnostics를 함께 확인한다. Label preparation은 일반 training scalar 곡선이 아니라 status/launch log로 진행을 본다. Writer accuracy 곡선을 로봇 task accuracy로 해석하지 않는다.

## 12. 코드와 향후 실행 기록

핵심 파일은 [core](run_scripts/robomme/echo_cvom_core.py), [teacher](run_scripts/robomme/echo_cvom_teacher.py), [trainer](run_scripts/robomme/train_echo_cvom.py), [checkpoint](run_scripts/robomme/echo_cvom_checkpoint.py), [policy](run_scripts/robomme/policy_echo_cvom.py), [server](run_scripts/robomme/serve_echo_cvom.py), [integration verification](run_scripts/robomme/verify_echo_cvom.py), [evaluation wrapper](run_scripts/robomme/eval_echo_cvom.py), [실행 wrapper](run_scripts/robomme/run_echo_cvom.sh)다.

실제 학습·label·native 진단·VAL160 결과는 완료 증거와 정확한 경로/hash를 확보한 뒤 이 절 아래에 추가한다. 이 문서의 설계 설명만으로 정확도·성공률 향상을 보장하지 않는다.

2026-09-21 구현 검증 기록은 [ECHO_CVOM_SMOKE_20260921.md](ECHO_CVOM_SMOKE_20260921.md)에 있다. 실제 작은 학습/teacher/writer/온라인 AE 검사는 완료했지만 full training 및 simulator VAL160은 아직 실행하지 않았다.
