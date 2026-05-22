"""
train_ppo.py – Cell-level PPO training for coverage with random unknown obstacles.

Unlike the Q-learning agent (which picks strip order), this PPO agent
makes cell-by-cell movement decisions (UP/RIGHT/DOWN/LEFT) over the
20 cm grid, learning an efficient coverage path from scratch.

Each episode:
  - 14 random objects are placed (unknown to the agent)
  - Agent navigates cell-by-cell, discovering obstacles via proximity
  - Reward encourages discovering new cells and penalises overlap
  - Episode ends on full coverage (+50 bonus) or timeout

Uses a CNN feature extractor over a 3-channel grid observation:
  Ch 0: discovered obstacles   Ch 1: visited cells   Ch 2: robot position

Usage
-----
    python train_ppo.py                         # 500k timesteps
    python train_ppo.py --timesteps 1000000
    python train_ppo.py --curriculum            # 1st half empty, 2nd half objects
    python train_ppo.py --resume
    python train_ppo.py --no-objects
"""

import argparse, sys, time
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from coverage_env.room                import load_room_grid
from ppo_training.cell_env            import CellCoverageEnv
from ppo_training.feature_extractor   import GridCNN

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure as configure_logger

DEFAULT_RESULTS_DIR = "results_ppo_3"
MAPS_DIR            = ROOT / "maps"
DEFAULT_SDF         = str(MAPS_DIR / "env1_bedroom.sdf")

DEFAULT_TIMESTEPS   = 500_000
DEFAULT_N_OBJECTS    = 14
DEFAULT_OBJ_MIN     = 1
DEFAULT_OBJ_MAX     = 3

# Default PPO hyperparameters
DEFAULT_LR           = 3e-4
DEFAULT_N_STEPS      = 2048
DEFAULT_BATCH_SIZE   = 64
DEFAULT_N_EPOCHS     = 10
DEFAULT_GAMMA        = 0.99
DEFAULT_GAE_LAMBDA   = 0.95
DEFAULT_CLIP_RANGE   = 0.2
DEFAULT_ENT_COEF     = 0.05
DEFAULT_VF_COEF      = 0.5

# CNN feature extractor
DEFAULT_FEATURES_DIM = 128


def parse_args():
    p = argparse.ArgumentParser(
        description="Train cell-level PPO agent for coverage")
    
    # Environment
    p.add_argument("--timesteps",     type=int,   default=DEFAULT_TIMESTEPS)
    p.add_argument("--n-objects",     type=int,   default=DEFAULT_N_OBJECTS)
    p.add_argument("--obj-min",       type=int,   default=DEFAULT_OBJ_MIN)
    p.add_argument("--obj-max",       type=int,   default=DEFAULT_OBJ_MAX)
    p.add_argument("--no-objects",    action="store_true")
    p.add_argument("--cell-size",     type=float, default=0.20)
    p.add_argument("--max-steps-factor", type=int, default=4,
                   help="max_steps = factor * n_free_cells (default 4)")
    
    # PPO hyperparameters
    p.add_argument("--lr",            type=float, default=DEFAULT_LR)
    p.add_argument("--n-steps",       type=int,   default=DEFAULT_N_STEPS)
    p.add_argument("--batch-size",    type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--n-epochs",      type=int,   default=DEFAULT_N_EPOCHS)
    p.add_argument("--gamma",         type=float, default=DEFAULT_GAMMA)
    p.add_argument("--gae-lambda",    type=float, default=DEFAULT_GAE_LAMBDA)
    p.add_argument("--clip-range",    type=float, default=DEFAULT_CLIP_RANGE)
    p.add_argument("--ent-coef",      type=float, default=DEFAULT_ENT_COEF)
    p.add_argument("--vf-coef",       type=float, default=DEFAULT_VF_COEF)

    # CNN architecture
    p.add_argument("--features-dim",  type=int,   default=DEFAULT_FEATURES_DIM)
    p.add_argument("--net-arch",      type=int,   nargs="+", default=[64, 64],
                   help="Hidden layers after CNN extractor (default [64, 64])")
    
    # Training options
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--curriculum",    action="store_true",
                   help="Curriculum: 1st half no objects, 2nd half with objects")
    p.add_argument("--save-every",    type=int,   default=100_000,
                   help="Save checkpoint every N timesteps")
    p.add_argument("--eval-every",    type=int,   default=100_000,
                   help="Evaluate every N timesteps")
    p.add_argument("--eval-episodes", type=int,   default=10)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--sdf",           type=str,   default=DEFAULT_SDF,
                   help="Path to SDF world file defining the room")
    p.add_argument("--results-dir",   type=str,   default=DEFAULT_RESULTS_DIR,
                   help="Directory (relative to script) for checkpoints, "
                        "curves, logs, and tensorboard output")
    return p.parse_args()


def mask_fn(env):
    return env.action_masks()


def make_env(wall_grid, args, n_objects_override=None):
    if n_objects_override is not None:
        n_objects = n_objects_override
    else:
        n_objects = 0 if args.no_objects else args.n_objects
    env = CellCoverageEnv(
        wall_grid,
        cell_size_m       = args.cell_size,
        n_objects         = n_objects,
        obj_min           = args.obj_min,
        obj_max           = args.obj_max,
        max_steps_factor  = args.max_steps_factor,
    )
    return ActionMasker(env, mask_fn)


def get_inner_env(wrapped_env):
    """Unwrap to the CellCoverageEnv."""
    env = wrapped_env
    while hasattr(env, "env"):
        env = env.env
    return env


# --- Callbacks ---

class EpisodeLoggerCallback(BaseCallback):
    # Logs per-episode reward, coverage, and steps to CSV.

    # Prints a progress line every N episodes
    PRINT_EVERY = 100

    def __init__(self, env, log_path, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.log_path = log_path
        self.total_timesteps = total_timesteps
        self.ep_rewards = []
        self.ep_coverages = []
        self.ep_steps = []
        self._current_reward = 0.0
        self._t_start = time.time()

        with open(log_path, "w") as f:
            f.write("episode,timestep,reward,coverage_pct,ep_steps\n")

    def _on_step(self):
        self._current_reward += self.locals["rewards"][0]

        info = self.locals.get("infos", [{}])[0]
        dones = self.locals["dones"][0]

        if dones:
            coverage = info.get("coverage", 0.0)
            steps = info.get("steps", 0)

            self.ep_rewards.append(self._current_reward)
            self.ep_coverages.append(coverage)
            self.ep_steps.append(steps)

            ep_num = len(self.ep_rewards)
            with open(self.log_path, "a") as f:
                f.write(f"{ep_num},{self.num_timesteps},"
                        f"{self._current_reward:.3f},"
                        f"{coverage*100:.2f},{steps}\n")

            if ep_num % self.PRINT_EVERY == 0:
                n = min(self.PRINT_EVERY, ep_num)
                recent_r = self.ep_rewards[-n:]
                recent_c = self.ep_coverages[-n:]
                recent_s = self.ep_steps[-n:]
                full = sum(1 for c in recent_c if c >= 0.999)
                elapsed = time.time() - self._t_start
                pct = 100 * self.num_timesteps / self.total_timesteps

                print(f"\n  ── Training progress: {self.num_timesteps:,}"
                      f"/{self.total_timesteps:,} steps ({pct:.0f}%)"
                      f"  [{elapsed:.0f}s elapsed] ──")
                print(f"  Episodes completed : {ep_num}")
                print(f"  Last {n} episodes:")
                print(f"    Avg reward       : {np.mean(recent_r):+.1f}")
                print(f"    Avg coverage     : {np.mean(recent_c)*100:.1f}%")
                print(f"    Full coverage    : {full}/{n}")
                print(f"    Avg steps/ep     : {np.mean(recent_s):.0f}")

            self._current_reward = 0.0
        return True


class CheckpointCallback(BaseCallback):
    # Saving the model every N timesteps

    def __init__(self, save_every, save_path, verbose=0):
        super().__init__(verbose)
        self.save_every = save_every
        self.save_path = save_path
        self._last_save = 0

    def _on_step(self):
        if self.num_timesteps - self._last_save >= self.save_every:
            path = f"{self.save_path}_step{self.num_timesteps}"
            self.model.save(path)
            self._last_save = self.num_timesteps
            print(f"\n  [Checkpoint] Saved: {path}.zip")
        return True


class CurriculumCallback(BaseCallback):
    # At the midpoint of training, switching from empty room to including random objects

    def __init__(self, train_env, eval_env, n_objects, switch_at, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env = eval_env
        self.n_objects = n_objects
        self.switch_at = switch_at
        self._switched = False

    def _on_step(self):
        if not self._switched and self.num_timesteps >= self.switch_at:
            self._switched = True
            get_inner_env(self.train_env).set_n_objects(self.n_objects)
            get_inner_env(self.eval_env).set_n_objects(self.n_objects)
            print(f"\n  {'='*58}")
            print(f"  [Curriculum] Phase 2: switching to {self.n_objects} "
                  f"random objects at step {self.num_timesteps:,}")
            print(f"  {'='*58}\n")
        return True


# --- Plotting ---

def moving_avg(v, w=50):
    if len(v) < w:
        return np.array(v, dtype=float)
    return np.convolve(v, np.ones(w)/w, mode="valid")


def save_curves(rewards, coverages, steps, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(rewards, alpha=0.15, color="steelblue")
    axes[0].plot(moving_avg(rewards), color="steelblue", lw=2)
    axes[0].set_ylabel("Episode reward"); axes[0].grid(alpha=0.3)

    axes[1].plot(np.array(coverages)*100, alpha=0.15, color="seagreen")
    axes[1].plot(moving_avg(np.array(coverages)*100), color="seagreen", lw=2)
    axes[1].axhline(100, color="seagreen", lw=1, ls="--", alpha=0.5)
    axes[1].set_ylabel("Coverage (%)"); axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3)

    axes[2].plot(steps, alpha=0.15, color="darkorange")
    axes[2].plot(moving_avg(steps), color="darkorange", lw=2)
    axes[2].set_ylabel("Steps / episode"); axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)

    plt.suptitle("PPO Cell-Level Coverage + Random Unknown Obstacles",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()


# --- Main ---

def train(args):
    results_dir  = ROOT / args.results_dir
    model_path   = str(results_dir / "ppo_model")
    final_model  = str(results_dir / "ppo_model_final")
    curve_path   = str(results_dir / "training_curves.png")
    log_path     = str(results_dir / "training_log.csv")

    results_dir.mkdir(exist_ok=True)

    wall_grid, meta = load_room_grid(Path(args.sdf), cell_size=args.cell_size)
    np.save(str(MAPS_DIR / "wall_grid.npy"), wall_grid)

    n_objects = 0 if args.no_objects else args.n_objects

    # Starting with no objects, switch at midpoint
    if args.curriculum:
        train_env = make_env(wall_grid, args, n_objects_override=0)
        eval_env  = make_env(wall_grid, args, n_objects_override=0)
    else:
        train_env = make_env(wall_grid, args)
        eval_env  = make_env(wall_grid, args)

    rows, cols = wall_grid.shape
    n_free = int((wall_grid == 0).sum())

    print(f"\n{'='*62}")
    print(f"  PPO Cell-Level Coverage Training")
    print(f"{'='*62}")
    print(f"  Environment")
    print(f"    Grid             : {rows}x{cols}  ({n_free} free cells)")
    print(f"    Cell size        : {args.cell_size}m")
    if args.curriculum:
        print(f"    Curriculum       : Phase 1 = no objects (0-{args.timesteps//2:,} steps)")
        print(f"                       Phase 2 = {n_objects} objects "
              f"({args.timesteps//2:,}-{args.timesteps:,} steps)")
    else:
        print(f"    Objects/episode  : {n_objects}  "
              f"size {args.obj_min}-{args.obj_max} cells "
              f"({'OFF' if args.no_objects else 'random, unknown to agent'})")
    print(f"    Obj size range   : {args.obj_min}-{args.obj_max} cells")
    print(f"    Max steps/ep     : {args.max_steps_factor} x free_cells")
    print(f"    Observation      : (4, {rows}, {cols})  CNN")
    print(f"    Actions          : 4 (UP / RIGHT / DOWN / LEFT)")
    print(f"  Hyperparameters")
    print(f"    Total timesteps  : {args.timesteps:,}")
    print(f"    Learning rate    : {args.lr}")
    print(f"    Gamma            : {args.gamma}")
    print(f"    GAE lambda       : {args.gae_lambda}")
    print(f"    Clip range       : {args.clip_range}")
    print(f"    Entropy coeff    : {args.ent_coef}")
    print(f"    Batch / n_steps  : {args.batch_size} / {args.n_steps}")
    print(f"    PPO epochs       : {args.n_epochs}")
    print(f"  Network")
    print(f"    CNN features     : {args.features_dim}")
    print(f"    Policy net_arch  : {args.net_arch}")
    print(f"  Schedule")
    print(f"    Checkpoint every : {args.save_every:,} steps")
    print(f"    Eval every       : {args.eval_every:,} steps "
          f"({args.eval_episodes} episodes)")
    print(f"    Seed             : {args.seed}")
    print(f"  Output             : {results_dir}")
    print(f"{'='*62}\n")

    policy_kwargs = dict(
        features_extractor_class=GridCNN,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=args.net_arch,
    )

    if args.resume and Path(final_model + ".zip").exists():
        print(f"[Resume] Loading {final_model}.zip ...")
        model = MaskablePPO.load(final_model, env=train_env)
        model.learning_rate = args.lr
        print(f"[Resume] Loaded successfully.\n")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate  = args.lr,
            n_steps        = args.n_steps,
            batch_size     = args.batch_size,
            n_epochs       = args.n_epochs,
            gamma          = args.gamma,
            gae_lambda     = args.gae_lambda,
            clip_range     = args.clip_range,
            ent_coef       = args.ent_coef,
            vf_coef        = args.vf_coef,
            policy_kwargs  = policy_kwargs,
            seed           = args.seed,
            verbose        = 0,
        )

    sb3_logger = configure_logger(str(results_dir), ["csv", "tensorboard"])
    model.set_logger(sb3_logger)

    ep_logger = EpisodeLoggerCallback(train_env, log_path, args.timesteps)
    ckpt_cb   = CheckpointCallback(args.save_every, model_path)
    eval_cb   = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(results_dir),
        log_path=str(results_dir),
        eval_freq=args.eval_every,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    cb_list = [ep_logger, ckpt_cb, eval_cb]

    if args.curriculum:
        cb_list.append(CurriculumCallback(
            train_env, eval_env,
            n_objects=n_objects,
            switch_at=args.timesteps // 2,
        ))

    callbacks = CallbackList(cb_list)

    print("Starting training ...\n")
    t_start = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    elapsed = time.time() - t_start

    # --- Saving final model and curves ---
    model.save(final_model)

    if ep_logger.ep_rewards:
        save_curves(ep_logger.ep_rewards, ep_logger.ep_coverages,
                    ep_logger.ep_steps, curve_path)

    # --- Final summary ---
    n_eps = len(ep_logger.ep_rewards)
    print(f"\n{'='*62}")
    print(f"  Training Complete")
    print(f"{'='*62}")
    print(f"  Time               : {elapsed:.1f}s  "
          f"({args.timesteps/elapsed:.0f} timesteps/s)")
    print(f"  Episodes           : {n_eps}")

    if n_eps > 0:
        tail = min(200, n_eps)
        recent_c = ep_logger.ep_coverages[-tail:]
        recent_s = ep_logger.ep_steps[-tail:]
        complete = sum(1 for c in recent_c if c >= 0.999)
        print(f"  Best coverage      : "
              f"{max(ep_logger.ep_coverages)*100:.1f}%")
        print(f"  Avg coverage (last {tail})  : "
              f"{np.mean(recent_c)*100:.1f}%")
        print(f"  Full coverage (last {tail}) : {complete}/{tail}")
        print(f"  Avg steps    (last {tail})  : {np.mean(recent_s):.0f}")

    print(f"  Saved files:")
    print(f"    Final model      : {final_model}.zip")
    print(f"    Best model       : {results_dir}/best_model.zip")
    print(f"    Training curves  : {curve_path}")
    print(f"    Training log     : {log_path}")
    print(f"    Tensorboard      : {results_dir}/")
    print(f"{'='*62}")
    print(f"\n  Next steps:")
    print(f"    Evaluate:  .venv/bin/python evaluate_ppo.py --episodes 5 "
          f"--results-dir {args.results_dir} --sdf {args.sdf}")
    print(f"    Resume:    .venv/bin/python train_ppo.py --resume "
          f"--results-dir {args.results_dir} --sdf {args.sdf}")
    print(f"    Monitor:   tensorboard --logdir {results_dir}\n")


if __name__ == "__main__":
    train(parse_args())
