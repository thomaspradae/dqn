# Released DeepMind DQN Code Biopsy

Source inspected: `google-deepmind/dqn` commit
`9d9b1d13a2b491d6ebd4d046740c511c662bbe0f` on `ofi1`.

Scope: the released `run_cpu` path and the active Nature-style PyTorch trainer.
This document separates training-dynamics mismatches from intentionally different
monitoring/reporting choices.

## Confirmed training-dynamics differences

| Item | Released DeepMind code | Current v3 baseline | Status |
| --- | --- | --- | --- |
| RMSProp gradient mean | `g = .95*g + .05*grad` | absent (`centered=False`) | mismatch |
| RMSProp variance | `g2 = .95*g2 + .05*grad²` | present | match |
| RMSProp denominator | `sqrt(g2 - g² + .01)` | `sqrt(g2) + .01` | mismatch |
| RMSProp momentum | none; update buffer reset every step | `momentum=.95` | mismatch |
| Resize interpolation | `image.scale(..., 'bilinear')` | `cv2.INTER_AREA` | mismatch |

The released optimizer is in `NeuralQLearner.lua:qLearnMinibatch`. It is
centered RMSProp, with epsilon **inside** the square root and no momentum
accumulator. `deepmind_rmsprop.py` implements that exact update rule and its
deterministic manual-formula test passes locally and on `ofi2`.

The active v3 baseline was deliberately not modified. Its metadata and
checkpointed optimizer remain the PyTorch `RMSprop(alpha=.95, eps=.01,
momentum=.95)` configuration, so it remains interpretable as the
reward-clipping/decision-schedule baseline.

## Confirmed matches

| Item | Evidence |
| --- | --- |
| Architecture | `convnet_atari3.lua`: 32x8/4, 64x4/2, 64x3/1, 512-unit hidden layer; matches `network.py`. |
| Input scale | Official replay supplies states divided by 255 in `TransitionTable.lua`; `network.py` divides byte-valued stacks by 255. |
| Action repeat / max-pooling | `run_cpu`: `actrep=4`, two-frame max pool; current environment path is configured equivalently. |
| Reward storage | Official clips `reward` before `transitions:add`; corrected trainer aggregates repeat rewards then clips once before replay insertion. |
| Replay size / batch / gamma | `run_cpu`: 1M, 32, .99; current defaults match. |
| Decision schedules | learning start 50k, update every four decisions, epsilon decay 1M decisions after start, target copy post-increment at `1 mod 10k`; audited separately in `COUNTER_AUDIT.md`. |

## Intentionally different monitoring

| Item | Released launcher | Current project protocol |
| --- | --- | --- |
| Milestone evaluation | every 250k decisions; 125k decision rollout | every 1.25M decisions; 10 episodes |
| Checkpoint save | every 125k decisions | every 100 episodes |
| Completion report | launcher history | 30 episodes each for final and best checkpoint |

These choices do not affect replay, optimization, or action selection during
training. The decision-unit milestone trigger is in the registry and status
snapshot, not in the report text.

## Items requiring a separate decision or microprobe

- **Resize interpolation:** switch future experimental runs to bilinear only
  after deciding whether it belongs with the optimizer correction. Do not alter
  v3.
- **Loss reduction convention:** released Lua passes clipped TD residuals
  directly to `network:backward`; PyTorch Huber loss uses a batch mean. RMSProp
  is largely gradient-scale invariant away from epsilon, but Torch7 reduction
  semantics should be microprobed before calling this a confirmed mismatch.
- **Initial parameter distribution:** architecture is aligned; exact Torch7 vs
  PyTorch initializer equivalence has not yet been source-proven.
- **Life-loss terminal signal:** current behavior was video-checked; the
  released ALE wrapper implementation is outside this repository checkout and
  has not been source-compared line-by-line.

## Next optimizer experiment

Use `--optimizer deepmind_rmsprop` for an explicitly named future run. It has
no momentum, centers gradients, and places epsilon inside the square root. The
default remains `pytorch_rmsprop` so existing and active runs remain unchanged.
