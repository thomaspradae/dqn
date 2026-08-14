# Breakout v3 Preflight Contract

## Scientific configuration

The first full baseline is named
`breakout_v3_seed20260621_rewardclip_dm50mdec`.

- Budget: 50,000,000 agent decisions, with 200,000,000 nominal raw action
  frames at `frame_skip=4`.
- Schedules: learning starts at 50,000 decisions; epsilon decays for 1,000,000
  decisions after learning starts; Q-learning occurs every four decisions; the
  target network is copied every 10,000 decisions.
- Replay transition reward: sum the action-repeat reward, then clip once to
  `[-1, 1]` before replay insertion.

## Monitoring (intentionally not launcher-equivalent evaluation)

- Milestone evaluation: every 1,250,000 agent decisions (equivalent to 5M
  nominal action frames), 10 episodes.
- Report-grade evaluation after completion: 30 episodes for final checkpoint
  and 30 episodes for the best pre-final milestone checkpoint.
- `training_diagnostics.csv`: every 50,000 agent decisions. It records
  counters, epsilon, reward-clipping counts, loss/TD/Q/gradient aggregates,
  life losses, target copies, and selected-action counts.

## Predeclared interpretation

Using report-grade 30-episode means:

- **Retained performance:** final score is at least 50% of the best checkpoint
  score (within a factor of two).
- **Collapse:** final score is below 20% of the best checkpoint score.
- **Degraded / inconclusive:** final score is from 20% through below 50% of the
  best checkpoint score. Do not retrospectively call this a success or a
  collapse without a follow-up replication.

These thresholds classify persistence of learned performance; they do not set
a claim about reaching the Nature score.

## Launch gate

1. **Passed 2026-06-21:** a policy recording from v2 best checkpoint `ep8200`
   started active play after reset, logged nonterminal life changes `5→4→3→2→1`,
   and ended only at `1→0`. A frame sequence around the first loss shows the
   ball active again after the `5→4` boundary. No forced reset/FIRE injection is
   present in the evaluation path; the policy/environment continued play.
2. Pass the 100k-decision Slurm preflight: exact decision stop, correct
   metadata, diagnostics rows, valid checkpoint ledger, and terminal
   auto-evaluation behavior.
3. Launch the first full seed on `ofi1` only. Inspect its first 12–24 hours of
   diagnostics for non-finite values, sharply growing Q/TD/gradient values, or
   pathological action concentration before launching a second seed on `ofi2`.

## 100k-decision Slurm preflight result

**Passed 2026-06-21** as
`breakout_v3_smoke_seed20260621_rewardclip_dmdec100k_retry2` (job 41).

- Log wall time: `20:54:16Z` to `21:05:56Z` = 700 seconds.
- Final checkpoint: 100,000 agent decisions, 401,324 actual raw emulator
  interactions, 12,499 optimizer updates, and 10 target copies.
- Actual raw interactions can differ from the 400,000 nominal action frames:
  terminal/life-loss boundaries can shorten action repeats, while reset no-ops
  add raw emulator interactions. In this smoke, reset no-ops dominated, so the
  actual count was 401,324.
- Diagnostics were emitted at every 10k decisions and remained finite through
  the final row (loss 0.0041, TD-error p95 0.0241, max absolute Q 0.1121,
  gradient norm 0.0142).
- Final and best-checkpoint 30-episode auto-evaluations completed, followed by
  `AUTOEVAL_DONE`.

This is 143 decisions/s (573 actual raw interactions/s). The v2 seed 20260609
log took about 40h58m for 50M raw interactions, or about 85 decision-equivalents/s.
The preflight therefore shows no slow-down that would invalidate the projected
multi-day full-budget run. Its 10-hour Slurm allocation was a safety ceiling,
not measured runtime.
