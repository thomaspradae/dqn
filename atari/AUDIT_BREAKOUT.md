# Breakout Nature DQN Audit

Audit date: 2026-06-21

Scope: the two seed-controlled 50M-frame Breakout runs, their evaluator, and the
automatic-evaluation path. `old1` / SB3 are intentionally out of scope.

Reference protocol: [Mnih et al., *Human-level control through deep
reinforcement learning* (2015)](https://doi.org/10.1038/nature14236). The paper
describes a 50M-frame training budget, 4-frame action repeat, 30 evaluations of
at most five minutes with epsilon 0.05, and random no-op initial conditions.
The released DeepMind launcher instead uses `steps=50M` with `actrep=4` and logs
raw frames as `steps * actrep` (200M repeated emulator frames). This audit keeps
agent decisions / perceived states separate from repeated raw emulator frames.

## Recovered completed runs

| Run | Node | Final training state | Best 10-episode milestone eval | Final 30-episode eval | Best-checkpoint 30-episode eval |
| --- | --- | --- | --- | --- | --- |
| `breakout_v2_seed20260609` | `ofi1` | 50,000,000 frames; 18,230 episodes; 3,115,588 optimizer steps | ep8200: 210.20 | final: **3.07** | ep8200: **177.90** |
| `breakout_v2_seed20260610` | `ofi2` | 50,000,000 frames; 19,102 episodes; 3,115,300 optimizer steps | ep7300: 127.10 | final: **19.37** | ep7300: **90.77** |

Both runs have `q_net_final.pt`, `checkpoint_final.pt`, `checkpoints.csv`, and
`RUN_METADATA.json`. Neither node had an active training or evaluation process.

The complete milestone trajectories show real learning followed by severe
regression, not a one-off noisy score:

- Seed 20260609: 36.70, 172.60, **210.20**, 5.40, 6.40, 8.80, 9.60, 17.20.
- Seed 20260610: 46.10, **127.10**, 20.70, 70.80, 38.90, 11.60, 23.60, 5.20.

The prior final-milestone evaluations were invalid: each attempted a nonexistent
`q_net_ep<final_episode>.pt` instead of `q_net_final.pt`. They provide no model
performance evidence. The repaired final evaluations above supersede those rows.

## Evaluation protocol

Confirmed aligned with the paper:

- `ALE/Breakout-v5`, emulator `frameskip=1`, `repeat_action_probability=0.0`.
- Four-frame action repeat, max-pool of the last two raw frames, luminance
  conversion, 84x84 resize, and a four-frame stack.
- Evaluation rewards are raw/unclipped.
- Epsilon is 0.05.
- The five-minute cap is implemented as 4,500 decisions x 4 raw frames = 18,000
  emulator frames.
- Evaluation ends only on actual game termination, not life loss.

Items to keep explicit:

- Milestone evaluations were 10 episodes, so they are diagnostic only. Completed
  runs now receive report-grade 30-episode final and best-checkpoint evaluations.
- Resets use 1–30 no-ops rather than allowing zero no-ops. This is a small
  protocol difference worth normalizing once the original implementation choice
  is pinned down.
- Breakout exposes `NOOP`, `FIRE`, `RIGHT`, and `LEFT`. Thirty no-ops leave the
  game dormant, and a `FIRE` action starts play. Neither training nor evaluation
  injects `FIRE`; the policy must select it. This is compatible with the paper's
  minimal-action-knowledge framing, but should be video-checked after a life
  loss to make sure a restart is not being mishandled.

## Training audit

### 1. Update frequency — direct official-code comparison complete

The official `google-deepmind/dqn` checkout was cloned on `ofi1` at commit
`9d9b1d13a2b491d6ebd4d046740c511c662bbe0f`. Its `run_cpu` sets `actrep=4` and
`update_freq=4`. `train_agent.lua` increments `step` and calls `agent:perceive`
once per selected action; `NeuralQLearner.lua` calls Q-learning when
`numSteps % update_freq == 0`, then increments `numSteps` once. Its own progress
line reports raw frames as `step * actrep`.

Our `action_step` increments once per selected action after the four raw-frame
repeat and trains when `action_step % train_freq == 0`. Therefore
`train_freq=4` matches the released DeepMind **decision-step cadence**: one
optimizer update every four selected actions. It is not the next parameter to
change.

The direct comparison exposes a separate run-budget mismatch: DeepMind's
launcher sets `steps=50M`, which its log labels as 200M raw frames at `actrep=4`.
Our v2 jobs stop at 50M raw emulator frames, or roughly 12.5M selected actions.
The completed v2 runs therefore performed roughly one quarter of the official
launcher's 50M-decision loop and one quarter of its optimizer updates. The full
schedule comparison is in
[`audit_deepmind_compare/COUNTER_AUDIT.md`](audit_deepmind_compare/COUNTER_AUDIT.md).

The counter audit also found and corrected three v2 decision-unit mismatches:
learning started at 50k raw frames rather than 50k decisions, epsilon decayed
from the first raw frame rather than after learning start in decision units, and
target copies occurred every 10k optimizer updates rather than every 10k
decisions. A live 1,000-decision smoke test now stops exactly at the decision
budget and matches the scheduled update and target-copy counts.

### 2. Reward clipping — corrected and tested

The original training loop clipped each of the four repeated-frame rewards and
then summed them. A replay transition could therefore hold a reward from -4 to
+4. The official `NeuralQLearner.lua` clips the reward before
`transitions:add(...)`.

`train_nature.py` now sums the raw reward across the action repeat, clips that
transition value to `[-1, 1]`, and passes only the clipped value to
`ReplayMemory.append(...)`. Episode-return logging remains raw/unclipped. The
unit test in `tests/test_transition_reward_clipping.py` verifies the stored
transition-reward bound and the post-aggregation clipping behavior.

### 3. Life-loss terminals / reset behavior — partly confirmed

Training marks a loss of life terminal in replay and begins a fresh stacked-frame
history without resetting the full game. Evaluation uses true game-over only.
That is the intended high-level distinction. What remains to test is the exact
post-life-loss action sequence: the current code does not inject a no-op or
`FIRE` after the replay boundary.

### 4. Optimizer / targets / replay — decision-schedule corrections applied

- Current optimizer: PyTorch RMSprop with `lr=2.5e-4`, `alpha=0.95`,
  `eps=0.01`, and `momentum=0.95`.
- Batch size 32, replay capacity 1M, learning starts at 50k decisions, epsilon
  decays over 1M decisions after learning starts, target copy is scheduled every
  10k decisions, and gamma is 0.99.
- TD targets are detached and the loss is the unit-threshold Huber form.
- Life-loss and true-terminal transitions are stored with terminal flags.

The parameters match the reported recipe, but PyTorch RMSprop is not a proof of
bit-for-bit equivalence to the original DeepMind variant. This remains a lower
priority fidelity question after the counter schedule, reward aggregation, and
life-reset behavior have been isolated.

## Auto-evaluation repair

The old final-milestone queue retried the missing-checkpoint error indefinitely.
The repaired workflow now:

1. Resolves `latest` to `q_net_final.pt` when it exists.
2. Treats missing-artifact queue failures as terminal failures, not retry loops.
3. Claims a completed run with `AUTOEVAL_RUNNING` and an atomic lock.
4. Evaluates the final checkpoint for 30 episodes.
5. Evaluates the best pre-final evaluator candidate (or `best_train` if no prior
   evaluator result exists) for 30 episodes.
6. Writes `AUTOEVAL_SUMMARY.txt`, sends it through the existing Telegram config,
   and replaces the running marker with `AUTOEVAL_DONE`.

The watcher already runs every 15 minutes on `ofi1`; the final-run claim makes
the process idempotent across invocations. Both recovered v2 runs completed this
flow successfully on 2026-06-21 and their reports were sent through Telegram.

## Next experiment sequence

1. Preserve the completed v2 final-vs-best evaluations as the pre-correction
   evidence.
2. Video-check Breakout resets and life-loss restarts.
3. Select and name an explicit decision or raw-frame budget. The trainer now
   rejects the old ambiguous budget flag and provides a zero-environment counter
   smoke test.
4. Run one explicitly named corrected baseline once the budget is selected. Keep
   `train_freq=4`, learning rate, and evaluation protocol unchanged.
5. Reserve train-every-action for a later, separate ablation.
