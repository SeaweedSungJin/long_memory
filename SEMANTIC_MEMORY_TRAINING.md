# Metadata-answer + bounded operation-CVoM memory

This file lives at repository root because the existing `/docs` ignore rule
would otherwise omit the new experiment instructions from version control.

Implementation: 2026-09-21. This is a new, separately saved experiment, not a
rewrite of V19 checkpoints, caches, training code, or previous evaluations.
Offline losses, semantic answer accuracy and writer accuracy are NOT task
success rates. No 30%/40% performance improvement is assumed.

## Fixed starting point and comparable endpoint

- Initialize the existing V19 prefix actor:
  `runs/long_memory/v19_fullcoverage_v1/prefix_full/checkpoint-006072`.
- Keep its original HAMLET frozen; update external encoder/reader/fusion and
  the existing AE LoRA. Fusion remains Linear+sigmoid; no backbone unfreezing.
- Cache: `runs/long_memory/cache_full1600_v1`; TRAIN 1,276 episodes / 24,286
  eligible queries; cache-VAL 324 episodes. Cache-VAL is not simulator VAL160.
- Full coverage uses the existing V19 epoch planner. One epoch / query batch 4
  is 6,072 optimizer updates, not an arbitrary best step or 6,072 episodes.
- Each query causally replays its episode prefix, re-encoding retained history
  with current parameters. READ precedes WRITE. GT actions/labels never enter
  event encoding, query, controller, bank, or online policy.
- Existing cache precision is retained, and runtime precision stays native.
  This experiment does not retest precision alignment or change denoising.
- Fixed development evaluation: all 16 RoboMME tasks, VAL scenario 0–9 each,
  seed 6, action interval 16, max 1,300 steps, no video by default.
- Reuse the original 38/160 HAMLET reference only through the unchanged strict
  baseline reference checker. No old CSV/manifest edits or copied results.

## File map

| File | Responsibility |
|---|---|
| `semantic_memory_targets.py` | Read-only cache/Parquet alignment; labels-only target artifact |
| `semantic_memory_answers.py` | Small pooled actual-read MLP prediction heads |
| `semantic_memory_core.py` | Observe actual attention output; independent current/clock controls |
| `semantic_memory_storage.py` | Causal bounded KEEP / arbitrary REPLACE; guarded merger primitive |
| `semantic_memory_teacher.py` | Matched future/noise operation CVoM labels and controller loss |
| `semantic_memory_checkpoint.py` | Atomic actor + hashed semantic sidecars |
| `train_semantic_memory.py` | Stage1/Stage2, full coverage, logs, validation, resume |
| `eval_semantic_memory.py` | Fixed VAL160 comparison wrapper |
| `verify_semantic_memory.py` | Actual native policy/AE ON-OFF invariants on fixed teacher observations |
| `run_semantic_memory.sh` | Ordered workflow / TensorBoard commands |

All Python files above are under `run_scripts/robomme/`. Original `gr00t/`
runtime sources remain unchanged to preserve the reviewed baseline closure.

## Labels: present metadata, not hypothetical HDF5 fields

Four simple/grounded planner/online strings exist in the local Parquets.
`choice_action`, `is_subgoal_boundary`, depth, and `is_completed` do not.
The preparation records source hashes, schemas, observation cadence, labels,
masked reasons, source intervals, and the unchanged TRAIN/cache-VAL split.
Vocabularies/regression-head eligibility are fitted on TRAIN only.

Supported compositional targets: direction, occurrence ordinal, grasp end,
insertion side. Generic action names/static/no-record are not memory labels.
Online/planner disagreement and a two-frame boundary margin are excluded.
These labels refer to the query endpoint, not every action in its 16-step chunk.

Substantive past-target targets: VideoPlaceOrder and VideoPlaceButton. Parse
the instruction's requested ordinal or immediately-before/after button relation;
select the appropriate past demo placement; require a canonically observed
source frame, stable role-specific grounding and planner/online agreement.
Require its anchor to agree with the eventual execution target within 12px.
The regression target is the **past demo subgoal-onset y,x anchor**, not a
current object tracking location or arbitrarily selected first coordinate.
Future annotations are used only to validate targets, never as model input.

Initial prepared artifact: `runs/long_memory/semantic_targets_full1600_v1`.
TRAIN 6,864 / VAL 1,782 queries have at least one target. Past-anchor targets:
658 / 172 queries, across 130 / 34 episodes. Unsupported queries still receive
ordinary action loss. Missing targets are masked, never assigned invented GT.

## Stage 1: differentiable policy + content-answer supervision

FIFO32; each event is four 256-D encoded tokens, not four objects or RGB frames.
The answer head consumes the reader's actual attention-value output before
query/FFN residuals. An observing hook does not modify deployed computation.

`L_actor = existing prefix-weighted flow + 0.01 * memory_answer_loss`.
Classification uses CE; normalized past-anchor regression uses MSE. The weight
is a configurable starting value, not an optimized claim. Original tail weight
0.25 remains; generated-action validation also uses common unweighted flow.

Two separate equal-capacity MLP controls learn the same targets: current-only
(current encoded short/state), and clock-only (causal time basis + demo flag).
Their inputs are detached: they cannot create an actor shortcut. Their losses
train only their own heads. Clock has several channels so LayerNorm cannot
erase a lone time magnitude. No future-normalized episode progress is used.

These controls and zero-memory answer intervention are diagnostic, not proof
of unique correct retrieval. Grounding/current visibility and time shortcuts
remain possible. Answer heads are absent from online inference.

## Stage 2: actual bounded bank operations

Start from a completed Stage1 actor. Until 32 events: always append. At capacity:
KEEP the existing bank or REPLACE any one victim and append the new event.
All-zero initial scores exactly reproduce FIFO (deterministic tie fallback).
A shared two-layer MLP scores candidate/victim/bank context and operation type.

CVoM teacher uses a fixed actor/head/controller snapshot during each refresh:

`advantage(operation) = future_loss(KEEP) - future_loss(operation)`.

Future loss combines the same flow and weighted answer losses; near and
beyond-short-window future queries have identical observations/noise in every
branch. Only one write differs, followed by common FIFO continuation. Therefore
this is conditional one-operation utility, not optimal memory policy or robot
success. Signed gains are not clipped. Rotating candidate victims supplement
mandatory KEEP/FIFO controls to limit counterfactual cost.

Record the precise teacher write trace so controller training does not replay
a different bank after the controller changes. Re-encode its members with
current features; periodic teacher refresh records and bounds representation
drift. Actual actor queries use the current controller's hard choices.

Controller bootstrap updates only its MLP; subsequent actor/answer and writer
updates share a trainer. Action gradients pass through selected bank contents;
they **do not differentiate argmax eviction choices**. Those choices learn
from signed teacher advantages. No-informative-label refresh stops with an audit
instead of manufacturing write positives. Confidence uses paired-noise standard
errors as a heuristic, not a statistical significance claim.

Default: 128 TRAIN contexts, 16 held-out contexts, two paired noise samples,
five operation candidates, 100 bootstrap steps, refresh every 1,500 updates.
Teacher labels are selected solely from the proper split, never simulator TEST.

`--storage-policy fifo` produces an equal-budget continued-training control.
Same-actor `fifo` evaluation is only an inference ablation, not that trained
control. OFF preserves learned AE and WRITE, bypassing only external READ.

### Merge status (important)

An ordered cross-attention + FFN adjacent-event merger is implemented and
gradient/budget tested. It is **disabled in this trainer and every default run**.
Unit tests are not evidence that merging preserves counts/order/identity.
No semantic-preservation validation exists yet, so the trainer deliberately
does not expose a flag that would enable an unvalidated merger. Do not describe
the initial experiment as learned merging. First assess KEEP/REPLACE; merge
training/semantic validation is a subsequent controlled change, not automatic.

## Run in order

```bash
cd /home/sjkim/HAMLET-Isaac-GR00T
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# Metadata is already prepared; validate/reuse it, do not overwrite it.
bash run_scripts/robomme/run_semantic_memory.sh preflight
bash run_scripts/robomme/run_semantic_memory.sh stage1

# Recommended first rollout: assess the semantic FIFO actor before writer work.
bash run_scripts/robomme/run_semantic_memory.sh eval-stage1

bash run_scripts/robomme/run_semantic_memory.sh stage2-preflight
bash run_scripts/robomme/run_semantic_memory.sh stage2
bash run_scripts/robomme/run_semantic_memory.sh eval-preflight
bash run_scripts/robomme/run_semantic_memory.sh eval
```

Default training directory: `runs/long_memory/semantic_full_v1/{stage1,stage2}`.
Default evaluation: `runs/eval/robomme/semantic_full_v1_val160_seed6/{stage1,stage2}`.
Override `SEMANTIC_RUN_DIR` and `SEMANTIC_EVAL_DIR` for a **new experiment**.
`all` runs preparation → both stages → final 160 rollouts with preflights; it
does not inspect interim success to change settings. It is intentionally not
started by code implementation. Existing output paths are never overwritten.

To compare learned storage / FIFO / no READ of the SAME final actor:

```bash
# Different model set needs a new eval identity/output directory.
SEMANTIC_EVAL_DIR=runs/eval/robomme/semantic_full_v1_ablations_val160_seed6 \
bash run_scripts/robomme/run_semantic_memory.sh eval-ablation
```

This runs 3×160 new candidate episodes; baseline is reused. Main `eval` runs
only 160 new episodes. Paired scenarios, task success and uncertainty—not
offline loss—are the final performance criterion. Repeated-use VAL160 is a
development benchmark, not confirmatory generalization evidence.

## Monitoring, checkpointing, resuming

```bash
bash run_scripts/robomme/run_semantic_memory.sh monitor
# Browser / VS Code forward port6007: http://127.0.0.1:6007
```

Uses existing live TensorBoard monitor; train/val journals continually append.
Plots are also saved every `--plot-steps` and at completion. Watch action_loss,
generated_prefix_mse, memory_gain; separately compare answer heads and writer
metrics. `bank_fill` reaching1 is expected, not evidence of useful memory.
Read `training.log`, `metrics.jsonl`, `metrics.csv`, `validation-*.json`,
`answers-validation-*.json`, `teacher-*.json`, `status.json` and `last_checkpoint.json`.
Head/task-specific counts and original records are retained for later analysis.

Actor/AE/heads/manager/optimizer/RNG are saved in atomic new checkpoints. Never
resume by overwriting an old run. Example (replace saved path):

```bash
.venv/bin/python run_scripts/robomme/train_semantic_memory.py \
  --stage 1 --targets-dir runs/long_memory/semantic_targets_full1600_v1 \
  --resume runs/long_memory/semantic_full_v1/stage1/checkpoint-001000 \
  --output-dir runs/long_memory/semantic_full_v1/stage1_resumed \
  --activation-checkpointing
```

Exact resume enforces unchanged code/runtime/plan/targets/settings and keeps
the original full-epoch schedule. A fresh branch initializes weights only via
`--init-checkpoint`. `--smoke-only` is a two-update diagnostic, marked incomplete;
it is not a full trained model and cannot silently become a Stage2 parent.

## Acceptance and remaining limitations

- Preserve cache chronology, complete-episode reset, no label inputs and live
  gradient paths; regression tests cover these invariants.
- Compare answer heads to trained current/clock controls. Direction/count
  classification alone is not sufficient evidence of long-memory reasoning.
- Inspect CVoM sign balance, uncertainty, actual replacements and no-op rate.
- Compare candidate to original HAMLET and its own OFF, keeping these distinct.
- Early goal is evidence of reliable additional memory use and paired rollout
  improvement; it is not guaranteed by implementing the losses.
- Short features may have lost required information; this experiment cannot
  recover information absent from them. Merge, new visual encoders and backbone
  fine-tuning are not silently included.

## Executed verification (2026-09-21)

No full policy training and no new simulator-success evaluation was run during
implementation. Old models, caches, data and evaluation records remain intact.

1. Prepared all1,600 episodes' metadata targets, preserving1,276/324 split.
2. Real GPU Stage1 smoke: two updates / eight TRAIN queries, actual adapted AE
   flow/backward, offline generated-action validation, finite atomic checkpoint.
   Output `runs/long_memory/semantic_stage1_smoke_20260921`.
3. Initial Stage2 two-context smoke correctly stopped because paired gains
   did not exceed noise. Audit retained at
   `runs/long_memory/semantic_stage2_smoke_20260921`; no writer was fitted.
4. Structural smoke panel expanded to16 TRAIN contexts to cover overflowing
   task groups, retaining the same uncertainty rule (not selecting by success).
   Nineteen informative operation labels: eight positive, eleven negative.
   Two bootstrap steps and two real joint actor/controller updates completed:
   `runs/long_memory/semantic_stage2_smoke_panel16_20260921`.
5. Actual saved Stage2 policy checked on fixed cache-VAL1355 first72 endpoints
   and626 first6 endpoints, ON/OFF:78 endpoints/condition,156 WRITE attempts,
   **22 genuine AE generations** with the configured4 denoising steps.
   Exact matching bank/short/HAMLET-cache/persistent RNG, read-before-write,
   mask/cadence; OFF conditioning delta0. All256 AE adapter tensors and storage
   weights match the saved checkpoint and stay unchanged. Both KEEP and REPLACE
   actually executed (54 KEEP,26 REPLACE;76 underfull appends). No merge.
   Evidence: `runs/diagnostics/semantic_memory_smoke_20260921/completed.json`.
   These are teacher observations, not closed-loop success results. Online
   action_input_mask is absent in both passes; actual image/attention masks and
   cached executed-prefix masks were compared explicitly.
6. Stage1/Stage2 fixed VAL160 preflight and strict reuse of original HAMLET
   reference passed; no simulator/model was loaded by preflight.
7. 140 CPU tests passed across new modules and existing V18/V19/precision/READ
   comparison regressions. Tiny actual core/LoRA/optimizer checkpoint tests
   prove bit-exact pause/resume for Stage1 and Stage2 (toy action objective in
   those CPU tests, not a replacement for the real GPU smoke above).

Smoke reproduction (NEW output names required):

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python run_scripts/robomme/train_semantic_memory.py \
  --stage 1 --targets-dir runs/long_memory/semantic_targets_full1600_v1 \
  --output-dir runs/long_memory/semantic_stage1_smoke_new \
  --smoke-only --activation-checkpointing

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
.venv/bin/python run_scripts/robomme/train_semantic_memory.py \
  --stage 2 --targets-dir runs/long_memory/semantic_targets_full1600_v1 \
  --init-checkpoint runs/long_memory/semantic_stage1_smoke_new/checkpoint-000002 \
  --output-dir runs/long_memory/semantic_stage2_smoke_new \
  --smoke-only --activation-checkpointing

CUDA_VISIBLE_DEVICES=0 GR00T_INFERENCE_SEED=6 \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1 \
.venv/bin/python run_scripts/robomme/verify_semantic_memory.py \
  --checkpoint runs/long_memory/semantic_stage2_smoke_new/checkpoint-000002 \
  --output-dir runs/diagnostics/semantic_memory_smoke_new
```

Do not compare two-update smoke checkpoints as trained models. The full160
wrapper rejects them unless explicitly opted into initialization diagnostics.
