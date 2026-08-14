# Counter Audit

Question: does each active training schedule use the same unit as the released
DeepMind DQN code?

Official source: `google-deepmind/dqn`, commit
`9d9b1d13a2b491d6ebd4d046740c511c662bbe0f`, cloned at
`/home/uace/dqn/deepmind-dqn-official` on `ofi1`.

## Master units

| Implementation | Decision counter | Raw-frame counter | Optimizer counter |
| --- | --- | --- | --- |
| DeepMind | `step` / `numSteps` | displayed as `step * actrep` | implicit minibatch calls |
| v2 | `action_step` | `env_step` | `step` / `train_step` |
| corrected trainer | `action_step` | `env_step` | `step` / `train_step` |

DeepMind increments `numSteps` once per `perceive` call. `train_agent.lua`
asserts `step == agent.numSteps`; each loop calls `game_env:step` once with
`actrep=4` configured by `run_cpu`.

## Schedule comparison

| Item | DeepMind value | DeepMind unit | v2 unit | Corrected unit | Verdict |
| --- | ---: | --- | --- | --- | --- |
| Total budget | 50M | decisions | 50M raw frames | explicit choice | v2 mismatch |
| Action repeat | 4 | raw frames/action | 4 raw frames/action | 4 raw frames/action | match |
| Update frequency | 4 | decisions | 4 decisions | 4 decisions | match |
| Learning start | 50k | decisions | 50k raw frames | 50k decisions | fixed |
| Epsilon decay | 1M after learning start | decisions | 1M raw frames from start | 1M decisions after learning start | fixed |
| Target copy | 10k | decisions | 10k optimizer updates | 10k decisions | fixed |
| Main evaluation frequency | 250k | decisions | 5M raw-frame milestone | unchanged | intentionally separate monitoring protocol |
| Main evaluation length | 125k | decisions | 10 episodes, 4,500 decisions each | unchanged | not a launcher-equivalent eval |
| Checkpoint save | 125k | decisions | 100 episodes | unchanged | monitoring mismatch, not training dynamics |
| Replay reward | clip before replay | transition reward | could be [-4, 4] | clip aggregate to [-1, 1] | fixed |

## Corrected trainer contract

`train_nature.py` now exposes mutually exclusive budget flags:

```text
--max-agent-steps-total N       # selected actions / perceived states
--max-raw-env-frames-total N   # raw emulator frames, including reset no-ops
```

There is no ambiguous `--max-env-steps-total` flag. A decision budget records a
nominal raw-action-frame budget of `N * frame_skip` in run metadata; `env_step`
continues to record actual raw emulator interactions, including reset no-ops and
short terminal repeats.

## Deterministic 1,000-decision smoke result

Command:

```bash
python train_nature.py --counter-smoke-test \
  --max-agent-steps-total 1000 --learning-starts 0 \
  --train-freq 4 --target-update-freq 100 --frame-skip 4
```

Result:

```json
{"agent_decisions": 1000, "nominal_raw_action_frames": 4000,
 "optimizer_updates": 249, "target_copies": 10}
```

The smoke test creates no environment, replay buffer, checkpoint, or run
directory.

The `249` count is intentional and matches the Lua loop convention. A
1,000-decision budget executes calls with pre-increment `numSteps` values
`0..999`; eligible update counters are `4, 8, ..., 996`, which is 249 updates.
With `learning_starts=32`, the eligible counters are `36, 40, ..., 996`, which
is 241 updates. Target copies are checked after incrementing, so their trace is
separate from update counters.

Use `--trace-counters` to print both lists explicitly, for example:

```bash
python train_nature.py --counter-smoke-test --trace-counters \
  --max-agent-steps-total 20 --learning-starts 0 \
  --train-freq 4 --target-update-freq 10 --frame-skip 4
```

This reports update pre-increment counters `4, 8, 12, 16` and target-copy
post-increment counters `1, 11`, exactly mirroring `NeuralQLearner.lua`.
The first target copy at `1` is intentional: the learner increments
`numSteps` after the first perception and then tests `numSteps % target_q == 1`.
Subsequent copies are exactly `target_q` decisions apart.

A live 1,000-decision Breakout smoke test with `learning_starts=32`,
`train_freq=4`, and `target_update_freq=100` stopped at exactly 1,000 agent
decisions and recorded 241 optimizer updates and 10 target copies. It made 3,917
actual raw emulator interactions versus 4,000 nominal action frames because
life-loss terminals can end an action repeat early.

## Pre-launch requirement

The next baseline must declare its budget in its run ID and metadata. A
DeepMind-launcher-style run uses 50M decisions, a nominal 200M raw action frames,
and 12.5M scheduled optimizer updates before replay availability effects.
