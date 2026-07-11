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
# timing issues.  Requires at least MIN_EPISODES recent episodes before
# evaluating so the mean is stable.
#
# Reward per step ≈ -(theta_err_rad)².  At the episode lengths below:
#   200 steps, avg 8° off  → ep_reward ≈ -12   → threshold -20
#   500 steps, avg 8° off  → ep_reward ≈ -30   → threshold -50
#  1000 steps, avg 8° off  → ep_reward ≈ -60   → threshold -100
#
# Difficulty: 0.0=±5°, 0.15=±30°, 0.50=±90°, 1.0=full random
BALANCE_CURRICULUM = [
    # (ep_rew_mean threshold, new_difficulty, new_max_steps)
    # Reward is cos(theta_err) - 0.1*(w1_norm)² per step, range ≈ [-1.1, +1.0].
    # Stage 0: ±5°,  200 steps → max episode = +200.  Advance when avg ≥ 150.
    # Stage 1: ±30°, 500 steps → max episode = +500.  Advance when avg ≥ 350.
    # Stage 2: ±90°, 1000 steps → max episode = +1000. Advance when avg ≥ 700.
    # Stage 3: full random, 2000 steps → final difficulty.
    (150,  0.15,  500),   # holding ±5°/0.4s well   → widen to ±30°/1s
    (350,  0.50, 1000),   # holding ±30°/1s well    → widen to ±90°/2s
    (700,  1.00, 2000),   # holding ±90°/2s well    → full random/4s
]

# Number of recent episodes required before checking the threshold.
# Prevents advancing on a lucky streak of just a few episodes.
MIN_EPISODES = 50


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

    def _on_step(self) -> bool:
        if self._stage >= len(BALANCE_CURRICULUM):
            return True

        buf = self.model.ep_info_buffer
        if len(buf) < MIN_EPISODES:
            return True

        threshold, new_difficulty, new_max_steps = BALANCE_CURRICULUM[self._stage]
        mean_reward = np.mean([ep['r'] for ep in buf])

        if mean_reward < threshold:
            return True

        self._stage += 1
        self.model.ep_info_buffer.clear()  # discard old easy-stage episodes
        self._train_envs.env_method('set_difficulty', new_difficulty)
        self._train_envs.env_method('set_max_steps',  new_max_steps)
        self._eval_envs.env_method('set_difficulty',  new_difficulty)
        self._eval_envs.env_method('set_max_steps',   new_max_steps)
        if self.verbose:
            print(
                f"\n[Curriculum] step={self.num_timesteps:,}  "
                f"ep_rew_mean={mean_reward:.1f} (over {len(buf)} eps) → "
                f"difficulty={new_difficulty:.2f}, max_steps={new_max_steps} "
                f"(stage {self._stage}/{len(BALANCE_CURRICULUM)})"
            )
        return True


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #

def make_env(mode: str, arm_penalty_coeff: float = 0.175):
    """Factory returning a thunk for make_vec_env."""
    def _init():
        # Balance starts within ±45° of upright — the range it will actually
        # receive from the swing-up handoff.  Cosine reward gives smooth
        # gradient signal across the full ±45° without needing a curriculum.
        difficulty = 0.23 if mode == "balance" else 1.0  # 0.23 → max_angle=±45°
        env = FurutaEnv(mode=mode, domain_randomisation=True, difficulty=difficulty)
        env._arm_penalty_coeff = arm_penalty_coeff
        return env
    return _init


def train(
    mode: str,
    timesteps: int,
    n_envs: int,
    save: bool,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.01,
    arm_penalty_coeff: float = 0.175,
    load_model: str | None = None,
) -> PPO:
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Environments                                                         #
    # ------------------------------------------------------------------ #
    train_env = make_vec_env(make_env(mode, arm_penalty_coeff), n_envs=n_envs)
    eval_env  = make_vec_env(make_env(mode, arm_penalty_coeff), n_envs=1)

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
        )
        model.learning_rate = learning_rate
        model.ent_coef = ent_coef
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=learning_rate,
            n_steps=2048,           # steps per env per update
            batch_size=256,         # larger batch → more stable gradient estimates
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log="logs",
            verbose=0,
        )

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #
    callbacks = []

    if mode == "balance":
        pass  # curriculum disabled — cosine reward gives useful gradient from ±45°

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
    return model


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

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
        "--arm-penalty",
        type=float,
        default=0.175,
        help="Arm angular velocity penalty coefficient (default: 0.175)",
    )
    p.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to a saved model zip to continue training from "
             "(e.g. models/balance_best/best_model.zip)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\nTraining {args.mode} controller")
    print(f"  timesteps      : {args.timesteps:,}")
    print(f"  n_envs         : {args.n_envs}")
    print(f"  learning_rate  : {args.learning_rate}")
    print(f"  ent_coef       : {args.ent_coef}")
    print(f"  arm_penalty    : {args.arm_penalty}")
    print(f"  load_model     : {args.load_model or '(none — fresh run)'}")
    print(f"  save           : {not args.no_save}\n")
    train(
        mode=args.mode,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        save=not args.no_save,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        arm_penalty_coeff=args.arm_penalty,
        load_model=args.load_model,
    )
