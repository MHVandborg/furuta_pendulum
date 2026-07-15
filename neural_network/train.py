"""
PPO training script for the Furuta pendulum.

Usage
-----
  # Train the balance controller (starts near upright, curriculum enabled)
  python train.py --mode balance

  # Train the swing-up controller (starts near hanging, full torque allowed)
  python train.py --mode swingup

  # Quick smoke-test run (few steps, no saving)
  python train.py --mode balance --timesteps 50000 --no-save

Outputs (written to models/)
-----------------------------
  models/balance_final.zip       Trained balance policy (SB3 format)
  models/swingup_final.zip       Trained swing-up policy (SB3 format)
  models/balance_best/           Best checkpoint during training
  models/swingup_best/           Best checkpoint during training

TensorBoard logs are written to logs/. View with:
  tensorboard --logdir neural_network/logs
"""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from envs import FurutaEnv


# --------------------------------------------------------------------------- #
# Curriculum schedule (balance mode)                                          #
# --------------------------------------------------------------------------- #
# Reward-based: advance when the rolling mean of recent episode rewards
# exceeds a threshold.  Uses model.ep_info_buffer — a deque populated
# directly by SB3's Monitor wrapper — rather than the logger, which has
# timing issues.  Requires at least MIN_EPISODES recent episodes AND
# MIN_STEPS_PER_STAGE elapsed steps before evaluating, so a stage can't
# be cleared before the policy has actually been trained at it.
#
# Thresholds are expressed as a fraction of the CURRENT stage's max possible
# reward (steps_for_truncation × ~1.0/step at a well-settled balance), not a
# bare absolute number — that keeps them meaningful across future reward
# changes instead of silently going stale (as the old hardcoded 150/350/700
# did after several reward-function rewrites). A stage that also has to
# survive the FAILURE_ANGLE_RAD termination in furuta_env.py can't reach its
# threshold via a lucky slow fall, since a terminated episode ends with
# whatever (small/negative) reward it had accumulated so far.
#
# Difficulty: 0.0=±5°, 0.15=±30°, ... capped below (see MAX_SAFE_DIFFICULTY)
#
# Curriculum difficulty is capped so that even the hardest stage's starting
# angle stays safely below FAILURE_ANGLE_RAD (furuta_env.py). A reset that
# starts ALREADY past that threshold ends the episode on step 1 by
# definition, before any control is even possible. This used to top out at
# difficulty=1.0 (full random, ±180°) — confirmed empirically that 51% of
# resets there start already past the 90° failure line. That's exactly what
# produced a wildly bimodal eval curve once training reached that stage: a
# mix of instant failures and full-length episodes averaged together (huge
# std, episode-length means landing on odd fractions of 200```). MARGIN_DEG
# below keeps a buffer so the hardest stage isn't itself borderline.
_MARGIN_DEG = 10.0
MAX_SAFE_DIFFICULTY = (np.degrees(FurutaEnv.FAILURE_ANGLE_RAD) - _MARGIN_DEG - 5.0) / 175.0  # ~0.43 (~80°)
assert 0.0 < MAX_SAFE_DIFFICULTY < 1.0

# Stage durations are chosen to sit past the "free ride" window: reset()
# gives the pendulum velocity aimed at reducing its own starting error
# (mimicking a realistic swing-up handoff), so a short enough episode can be
# "solved" by applying zero torque and just coasting on that momentum before
# gravity's destabilizing pull has time to build up. At the old stage-0
# duration (200 steps), literal zero-torque scored 99.8% — comfortably over
# the threshold, meaning that stage taught nothing. Checked empirically per
# stage at these durations: zero-action scores 64% / 26% / 12% respectively,
# all safely under their thresholds, so clearing a stage now requires actual
# correction, not just coasting on the reset's built-in assist.
BALANCE_CURRICULUM = [
    # (steps_for_truncation, threshold_fraction, new_difficulty, max_episode_steps)
    # steps_for_truncation — episode length while AT this stage (used to
    #                        compute the reward threshold below it).
    # max_episode_steps    — episode length switched to once this stage is
    #                        cleared; becomes the next row's steps_for_truncation.
    #
    # Stage 0: ±5°,  600 steps.  Advance at 75% of max (450)  → widen to ~±49°/1000 steps.
    # Stage 1: ~±49°, 1000 steps. Advance at 70% of max (700)  → widen to ~±80°/1500 steps.
    # Stage 2: ~±80°, 1500 steps. Advance at 70% of max (1050) → stays at ~±80°, full 2000 steps.
    (600,  0.75, 0.25,               1000),
    (1000, 0.70, MAX_SAFE_DIFFICULTY, 1500),
    (1500, 0.70, MAX_SAFE_DIFFICULTY, 2000),
]

# Number of recent episodes required before checking the threshold.
# Prevents advancing on a lucky streak of just a few episodes.
MIN_EPISODES = 50

# Minimum environment steps that must elapse within a stage before it's even
# considered for advancement, regardless of episode count or reward. This is
# NOT redundant with MIN_EPISODES: PPO's n_steps=2048 x n_envs=8 means the
# first gradient update doesn't happen until 16,384 steps have been
# collected, but a stage-0 episode is only 200 steps — MIN_EPISODES=50 could
# (and did) get satisfied within the very first rollout, before the policy
# had been trained at all, so the "easy" stage advanced on a near-random
# policy that was never actually trained at that difficulty. This floor
# guarantees several PPO updates happen per stage no matter how trivially
# the reward threshold gets cleared.
MIN_STEPS_PER_STAGE = 150_000


def _stage_info(stage: int) -> str:
    """One-line summary of a curriculum stage: duration, max score, and the
    bar to clear next. Used both when a stage starts and when it's cleared."""
    total_stages = len(BALANCE_CURRICULUM)
    if stage < total_stages:
        steps_for_truncation, threshold_fraction, _, _ = BALANCE_CURRICULUM[stage]
        difficulty = 0.0 if stage == 0 else BALANCE_CURRICULUM[stage - 1][2]
        max_angle_deg = 5.0 + difficulty * 175.0
        threshold = threshold_fraction * steps_for_truncation
        return (
            f"stage {stage}/{total_stages}  ~+/-{max_angle_deg:.0f}deg  "
            f"max_steps={steps_for_truncation} (max score {steps_for_truncation})  "
            f"advance at >={threshold:.0f} ({threshold_fraction*100:.0f}%)"
        )
    else:
        _, _, difficulty, max_episode_steps = BALANCE_CURRICULUM[-1]
        return (
            f"stage {stage}/{total_stages} (final)  full random  "
            f"max_steps={max_episode_steps} (max score {max_episode_steps})"
        )


class CurriculumCallback(BaseCallback):
    """Advance difficulty and episode length when reward crosses a threshold.

    Reads episode rewards from model.ep_info_buffer (populated by Monitor),
    which is reliable unlike logger.name_to_value.  Uses env_method() to
    set attributes through SB3's Monitor wrapper to the underlying FurutaEnv.
    """

    def __init__(self, train_envs, eval_envs, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._train_envs = train_envs
        self._eval_envs  = eval_envs
        self._stage = 0
        self._stage_start_step = 0

    def _on_training_start(self) -> None:
        if self.verbose:
            print(f"[Curriculum] {_stage_info(self._stage)}")

    def _on_step(self) -> bool:
        if self._stage >= len(BALANCE_CURRICULUM):
            return True

        if self.num_timesteps - self._stage_start_step < MIN_STEPS_PER_STAGE:
            return True

        buf = self.model.ep_info_buffer
        if len(buf) < MIN_EPISODES:
            return True

        steps_for_truncation, threshold_fraction, new_difficulty, max_episode_steps = BALANCE_CURRICULUM[self._stage]
        threshold = threshold_fraction * steps_for_truncation
        n_eps = len(buf)
        mean_reward = np.mean([ep['r'] for ep in buf])

        if mean_reward < threshold:
            return True

        self._stage += 1
        self._stage_start_step = self.num_timesteps
        self.model.ep_info_buffer.clear()  # discard old easy-stage episodes — buf is the same deque, so grab n_eps above first
        self._train_envs.env_method('set_difficulty', new_difficulty)
        self._train_envs.env_method('set_max_steps',  max_episode_steps)
        self._eval_envs.env_method('set_difficulty',  new_difficulty)
        self._eval_envs.env_method('set_max_steps',   max_episode_steps)
        if self.verbose:
            print(
                f"\n[Curriculum] step={self.num_timesteps:,}  "
                f"ep_rew_mean={mean_reward:.1f} (over {n_eps} eps) -> "
                f"{_stage_info(self._stage)}"
            )
        return True


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def make_env(mode: str, arm_penalty_coeff: float = 2.0, linear_penalty: bool = False):
    """Factory returning a thunk for make_vec_env."""
    def _init():
        # Balance starts at curriculum stage 0 (±5°, 600 steps) and is widened
        # by CurriculumCallback as ep_rew_mean clears each stage's threshold —
        # see BALANCE_CURRICULUM. Swing-up always starts near hanging and
        # uses the full episode length from the start.
        if mode == "balance":
            env = FurutaEnv(mode=mode, domain_randomisation=True, difficulty=0.0, max_steps=600)
        else:
            env = FurutaEnv(mode=mode, domain_randomisation=True, difficulty=1.0)
        env._arm_penalty_coeff = arm_penalty_coeff
        env._arm_penalty_linear = linear_penalty
        return env
    return _init


def train(
    mode: str,
    timesteps: int,
    n_envs: int,
    save: bool,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    arm_penalty_coeff: float = 2.0,
    linear_penalty: bool = False,
    load_model: str | None = None,
) -> PPO:
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Environments                                                         #
    # ------------------------------------------------------------------ #
    train_env = make_vec_env(make_env(mode, arm_penalty_coeff, linear_penalty), n_envs=n_envs)
    eval_env  = make_vec_env(make_env(mode, arm_penalty_coeff, linear_penalty), n_envs=1)

    # ------------------------------------------------------------------ #
    # PPO model — fresh or loaded from checkpoint                         #
    # ------------------------------------------------------------------ #
    # Policy: two hidden layers of 64 neurons with tanh activation.
    # This matches what CMSIS-NN will run on the MCU (Phase 5).
    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=__import__("torch").nn.Tanh,
    )

    if load_model is not None:
        # Continue training from a saved checkpoint.  The loaded model keeps
        # its weights and value function; only the environment (and therefore
        # the reward) changes.  learning_rate and ent_coef are updated so the
        # CLI args take effect even when resuming.
        print(f"Loading model from {load_model} …")
        model = PPO.load(
            load_model,
            env=train_env,
            tensorboard_log="logs",
            verbose=0,
            device="cpu",
        )
        model.learning_rate = learning_rate
        model.ent_coef = ent_coef
        model.gamma = gamma
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            n_steps=2048,           # steps per env per update
            batch_size=256,         # larger batch → more stable gradient estimates
            n_epochs=10,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs",
            verbose=0,
            device="cpu",
        )

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #
    callbacks = []

    if mode == "balance":
        callbacks.append(CurriculumCallback(train_env, eval_env))

    if save:
        best_model_path = f"models/{mode}_best"
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=best_model_path,
                log_path=f"logs/{mode}_eval",
                eval_freq=max(50_000 // n_envs, 1),
                n_eval_episodes=10,
                deterministic=True,
                verbose=1,
            )
        )
        callbacks.append(
            CheckpointCallback(
                save_freq=max(50_000 // n_envs, 1),
                save_path=f"models/{mode}_checkpoints",
                name_prefix=mode,
                verbose=1,
            )
        )

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=mode,
        progress_bar=True,
    )

    if save:
        final_path = f"models/{mode}_final"
        model.save(final_path)
        print(f"\nSaved final model to {final_path}.zip")

    train_env.close()
    eval_env.close()

    if save:
        _print_training_summary(mode, timesteps)

    return model


def _print_training_summary(mode: str, total_timesteps: int) -> None:
    """Read the EvalCallback log and print a concise end-of-run summary.

    Indicators that training has NOT fully converged:
      - Best reward still rising at the end of the run (increase --timesteps)
      - Large gap between best and final reward (late instability / collapse)
      - High std at the end (policy is inconsistent across episodes)
      - Plateau reward well below 1800 (reward shaping problem, not a time problem)
    """
    eval_log = f"logs/{mode}_eval/evaluations.npz"
    if not os.path.exists(eval_log):
        return

    data        = np.load(eval_log)
    timesteps   = data["timesteps"]           # shape (N,)
    results     = data["results"]             # shape (N, n_eval_episodes)
    ep_lengths  = data["ep_lengths"]          # shape (N, n_eval_episodes)

    mean_rewards = results.mean(axis=1)
    std_rewards  = results.std(axis=1)
    best_idx     = int(np.argmax(mean_rewards))
    best_reward  = mean_rewards[best_idx]
    best_step    = int(timesteps[best_idx])

    # "Final" = mean over last 3 eval points (smoothed, not just the last one)
    tail         = min(3, len(mean_rewards))
    final_reward = mean_rewards[-tail:].mean()
    final_std    = std_rewards[-tail:].mean()
    final_step   = int(timesteps[-1])

    # Trend: slope of mean reward over last 20% of training
    window = max(int(len(mean_rewards) * 0.2), 2)
    slope  = float(np.polyfit(timesteps[-window:], mean_rewards[-window:], 1)[0])
    slope_per_100k = slope * 100_000

    # Convergence verdict.
    # With the gated arm penalty (coeff=2.0, quadratic, gated by cos(θ_err)),
    # 200°/s steady spin costs ~309 units/episode, well above training noise (~150 std).
    # At 2000 steps/episode: max = 2000, well-controlled still arm ≈ 1800–1980.
    still_rising  = slope_per_100k >  5.0   # reward improving meaningfully
    collapsed     = (best_reward - final_reward) > 100
    low_plateau   = best_reward < 1200
    inconsistent  = final_std > 200

    print("\n" + "=" * 60)
    print(f"  TRAINING SUMMARY -- {mode}  ({total_timesteps:,} steps)")
    print("=" * 60)
    print(f"  Best eval reward  : {best_reward:7.1f}  (at {best_step:,} steps)")
    print(f"  Final reward      : {final_reward:7.1f}  +/- {final_std:.1f}  (last {tail} evals)")
    print(f"  Trend (per 100k)  : {slope_per_100k:+.1f}  (last 20% of run)")
    print(f"  Max possible      :  2000.0  (2000 steps x +1.0/step)")
    print("-" * 60)

    issues = []
    if still_rising:
        issues.append("Still improving at end -- consider running longer (--timesteps)")
    if collapsed:
        issues.append(f"Late collapse: best {best_reward:.0f} -> final {final_reward:.0f} "
                      f"(try lower --ent-coef or smaller --arm-penalty)")
    if low_plateau:
        issues.append("Plateau well below 1200 -- likely a reward shaping issue, "
                      "not a time issue (check reward function)")
    if inconsistent:
        issues.append(f"High variance at end (std={final_std:.0f}) -- policy is unstable "
                      f"(try lower --learning-rate)")

    if issues:
        print("  WARNINGS:")
        for w in issues:
            print(f"    * {w}")
    else:
        print("  Converged -- no issues detected.")
    print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def _available_models() -> str:
    """Return a human-readable list of saved models for the --help text.

    Top-level zips (e.g. balance_p1_best.zip) are listed individually.
    Subdirectories (e.g. balance_checkpoints/) are summarised by file count
    to keep the help output readable.
    """
    import glob as _glob
    from collections import defaultdict

    all_zips = sorted(set(_glob.glob("models/**/*.zip", recursive=True)))
    if not all_zips:
        return "(no saved models found in models/)"

    top_level = []
    subdirs: dict[str, int] = defaultdict(int)
    for path in all_zips:
        parts = path.replace("\\", "/").split("/")
        if len(parts) == 2:          # models/foo.zip
            top_level.append(path)
        else:                        # models/subdir/foo.zip
            subdirs[parts[1]] += 1

    parts = top_level[:]
    for subdir, count in sorted(subdirs.items()):
        parts.append(f"models/{subdir}/ ({count} checkpoints)")
    return "available: " + ", ".join(parts)


def parse_args():
    p = argparse.ArgumentParser(description="Train Furuta pendulum PPO policy")
    p.add_argument(
        "--mode",
        choices=["balance", "swingup"],
        default="balance",
        help="Which controller to train (default: balance)",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total environment steps (default: 2M)",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel training environments (default: 8)",
    )
    p.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving models (useful for quick smoke tests)",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate (default: 3e-4)",
    )
    p.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="PPO entropy coefficient (default: 0.01)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="PPO discount factor. Effective planning horizon is roughly "
             "1/(1-gamma) steps (default 0.99 -> ~100 steps). Raise this "
             "(e.g. 0.995/0.999) if failures stem from a slow-building drift "
             "whose consequence arrives too many steps later for the current "
             "horizon to credit-assign correctly (default: 0.99)",
    )
    p.add_argument(
        "--arm-penalty",
        type=float,
        default=2.0,
        help="Arm velocity penalty coefficient — gated by cos(θ_err) so the "
             "arm can move freely when the pendulum is falling (default: 2.0)",
    )
    p.add_argument(
        "--linear-penalty",
        action="store_true",
        help="Use linear |w1| arm penalty instead of quadratic w1^2",
    )
    p.add_argument(
        "--load-model",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a saved model zip to continue training from. "
             f"{_available_models()}",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\nTraining {args.mode} controller")
    print(f"  timesteps      : {args.timesteps:,}")
    print(f"  n_envs         : {args.n_envs}")
    print(f"  learning_rate  : {args.learning_rate}")
    print(f"  ent_coef       : {args.ent_coef}")
    print(f"  gamma          : {args.gamma}")
    print(f"  arm_penalty    : {args.arm_penalty} ({'linear' if args.linear_penalty else 'quadratic'})")
    print(f"  load_model     : {args.load_model or '(none - fresh run)'}")
    print(f"  save           : {not args.no_save}\n")
    train(
        mode=args.mode,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        save=not args.no_save,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        arm_penalty_coeff=args.arm_penalty,
        linear_penalty=args.linear_penalty,
        load_model=args.load_model,
    )
