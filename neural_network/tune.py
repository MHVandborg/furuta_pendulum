"""
Hyperparameter tuner for the Furuta balance controller.

Uses Optuna (Bayesian / TPE optimisation) to find the best combination of:
  - arm_penalty_coeff  — how hard to penalise continuous arm spinning
  - ent_coef           — PPO entropy bonus (exploration vs exploitation)
  - learning_rate      — PPO learning rate

Each trial runs a short 500 k-step training and is scored by the mean
eval reward over the last N eval points.  Trials run in parallel (one
per CPU core by default) using a shared SQLite study database so results
survive interruption and can be resumed.

Usage
-----
  # Run 40 trials using 4 parallel workers (adjust -j to your core count)
  python tune.py --trials 40 --jobs 4

  # Resume a previous study
  python tune.py --trials 40 --jobs 4

  # View results in the browser (Optuna Dashboard)
  optuna-dashboard sqlite:///tune_study.db

  # Print best params found so far
  python tune.py --best
"""

import argparse
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import optuna
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from envs import FurutaEnv


# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

STUDY_NAME   = "furuta_balance"
STORAGE_PATH = "sqlite:///tune_study.db"
TIMESTEPS    = 500_000     # steps per trial — short enough to be fast, long
                           # enough to see convergence past the noisy early phase
N_ENVS       = 8           # parallel envs per trial (matches train.py)
EVAL_FREQ    = 50_000      # eval every N steps
N_EVAL_EPS   = 10          # eval episodes per checkpoint
# Score = mean of eval rewards in the final SCORE_WINDOW checkpoints.
# This rewards policies that converge AND stay stable rather than just
# peaking once and collapsing.
SCORE_WINDOW = 5


# --------------------------------------------------------------------------- #
# Patched environment factory                                                  #
# --------------------------------------------------------------------------- #

def make_env_fn(arm_penalty_coeff: float):
    """Return a thunk that builds a FurutaEnv with a patched reward coeff."""
    def _init():
        env = FurutaEnv(mode="balance", difficulty=0.23, domain_randomisation=True)
        # Monkey-patch the reward coefficient onto the env instance so the
        # existing _reward() method can read it without requiring a new arg.
        env._arm_penalty_coeff = arm_penalty_coeff
        return env
    return _init


# --------------------------------------------------------------------------- #
# Environment factory                                                          #
# --------------------------------------------------------------------------- #

def objective(trial: optuna.Trial) -> float:
    # ------------------------------------------------------------------ #
    # Sample hyperparameters                                               #
    # ------------------------------------------------------------------ #
    arm_penalty_coeff = trial.suggest_float("arm_penalty_coeff", 0.05, 0.40, log=True)
    ent_coef          = trial.suggest_float("ent_coef",          0.001, 0.05, log=True)
    learning_rate     = trial.suggest_float("learning_rate",     1e-4,  5e-4, log=True)

    # ------------------------------------------------------------------ #
    # Environments                                                         #
    # ------------------------------------------------------------------ #
    train_env = make_vec_env(make_env_fn(arm_penalty_coeff), n_envs=N_ENVS)
    eval_env  = make_vec_env(make_env_fn(arm_penalty_coeff), n_envs=1)

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=__import__("torch").nn.Tanh,
        ),
        verbose=0,  # suppress per-step output — Optuna prints trial summaries
    )

    # ------------------------------------------------------------------ #
    # Eval callback — stores results to a per-trial log path              #
    # ------------------------------------------------------------------ #
    log_path = f"logs/tune/trial_{trial.number}"
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=None,   # don't save models during tuning
        log_path=log_path,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=N_EVAL_EPS,
        deterministic=True,
        verbose=0,
    )

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    try:
        model.learn(total_timesteps=TIMESTEPS, callback=eval_cb, progress_bar=False)
    except Exception as e:
        train_env.close()
        eval_env.close()
        raise optuna.exceptions.TrialPruned(f"Training crashed: {e}")

    train_env.close()
    eval_env.close()

    # ------------------------------------------------------------------ #
    # Score: mean of the last SCORE_WINDOW eval checkpoints               #
    # ------------------------------------------------------------------ #
    npz_path = os.path.join(log_path, "evaluations.npz")
    if not os.path.exists(npz_path):
        raise optuna.exceptions.TrialPruned("No eval results found")

    data   = np.load(npz_path)
    means  = data["results"].mean(axis=1)
    score  = float(means[-SCORE_WINDOW:].mean())

    print(
        f"  Trial {trial.number:3d} | "
        f"arm_penalty={arm_penalty_coeff:.3f}  "
        f"ent_coef={ent_coef:.4f}  "
        f"lr={learning_rate:.5f}  "
        f"->  score={score:.1f}"
    )
    return score


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Tune Furuta balance hyperparameters")
    p.add_argument("--trials", type=int, default=40,
                   help="Total Optuna trials to run (default: 40)")
    p.add_argument("--jobs",   type=int, default=4,
                   help="Parallel workers / trials at once (default: 4, "
                        "set to your physical core count)")
    p.add_argument("--study-name", type=str, default=STUDY_NAME,
                   help=f"Optuna study name (default: {STUDY_NAME}). "
                        "Use a different name to start a fresh study without "
                        "deleting previous results.")
    p.add_argument("--best",   action="store_true",
                   help="Print best params from existing study and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=STORAGE_PATH,
        direction="maximize",
        sampler=TPESampler(seed=42),
        load_if_exists=True,   # resume if the DB already exists
    )

    if args.best:
        if len(study.trials) == 0:
            print("No completed trials yet.")
        else:
            t = study.best_trial
            print(f"\nBest trial #{t.number}  score={t.value:.1f}")
            for k, v in t.params.items():
                print(f"  {k:25s} = {v}")
        sys.exit(0)

    os.makedirs("logs/tune", exist_ok=True)

    print(f"Running {args.trials} trials with {args.jobs} parallel workers …")
    print(f"Results stored in {STORAGE_PATH}")
    print(f"View live in browser: optuna-dashboard {STORAGE_PATH}\n")

    study.optimize(
        objective,
        n_trials=args.trials,
        n_jobs=args.jobs,
        show_progress_bar=True,
    )

    print("\n=== Optimisation complete ===")
    best = study.best_trial
    print(f"Best score : {best.value:.1f} / 2000")
    print(f"Best trial : #{best.number}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k:25s} = {v:.5f}")
