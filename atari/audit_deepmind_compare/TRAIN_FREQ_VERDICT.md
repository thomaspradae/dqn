# Train Frequency Verdict: Released DeepMind Code

Verified on 2026-06-21 from the official checkout on `ofi1`:

```text
/home/uace/dqn/deepmind-dqn-official
commit 9d9b1d13a2b491d6ebd4d046740c511c662bbe0f
```

## Official code facts

`run_cpu` configures:

- `update_freq=4`
- `actrep=4`
- `steps=50000000`
- `learn_start=50000`
- `target_q=10000`
- `min_reward=-1`, `max_reward=1`

`dqn/train_agent.lua` increments `step` once per training-loop iteration, calls
`agent:perceive(...)` once, then calls `game_env:step(...)` once. Its progress
line explicitly reports raw frames as `step * opt.actrep`.

`dqn/NeuralQLearner.lua` states that `numSteps` is the “Number of perceived
states.” In `perceive`, it:

1. clips `reward` to `min_reward` / `max_reward`;
2. stores that reward with `self.transitions:add(...)`;
3. runs Q-learning when `numSteps % update_freq == 0`;
4. increments `numSteps` once at the end of the non-test call.

## Our code facts

`train_nature.py` selects one action, repeats it for `frame_skip=4` raw emulator
steps, increments `action_step` once, and trains when
`action_step % train_freq == 0`. With `train_freq=4`, it also updates once every
four selected actions / perceived states.

## Direct verdict

`train_freq=4` matches the released DeepMind code’s **decision-step cadence**:
one optimizer update every four selected actions. It should not be changed in the
next controlled baseline.

There is a separate, non-equivalent **run-budget** issue. The official launcher
sets `steps=50M` and its own log labels that as `steps * actrep = 200M` raw
frames at `actrep=4`. Our v2 runs stop at `max_env_steps_total=50M` raw emulator
frames, which is roughly 12.5M selected actions. Therefore the v2 runs executed
about one quarter of the launcher’s 50M-decision training loop and about one
quarter of its expected optimizer updates. This is a budget-definition issue, not
a `train_freq`-counter mismatch.

## Reward-clipping verdict

The official code clips the transition reward immediately before replay storage.
The corrected local implementation now sums the raw reward across the action
repeat and clips the resulting transition reward to `[-1, 1]` before
`ReplayMemory.append(...)`. Raw episode return remains un-clipped for logging.
