import argparse, sys, time
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # so the coverage_env / q_learning packages import

from coverage_env.room        import load_room_grid
from coverage_env.coverage_env import CoverageEnv
from q_learning.q_agent        import QLearningAgent

# Default Q-learning hyperparameters

DEFAULT_RESULTS_DIR = "ql_env2_corridor"
MAPS_DIR            = ROOT / "maps"
DEFAULT_SDF         = str(MAPS_DIR / "env1_bedroom.sdf")

DEFAULT_EPISODES     = 3_000
DEFAULT_ALPHA        = 0.3
DEFAULT_GAMMA        = 1.0
DEFAULT_EPS          = 1.0
DEFAULT_EPS_MIN      = 0.05
DEFAULT_EPS_DECAY    = 0.001
DEFAULT_SAVE_EVERY   = 1_000
DEFAULT_PRINT_EVERY  = 500
DEFAULT_N_OBJECTS    = 14
DEFAULT_OBJ_MIN      = 1
DEFAULT_OBJ_MAX      = 3


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Q-Learning agent for coverage with random obstacles")
    p.add_argument("--episodes",      type=int,   default=DEFAULT_EPISODES)
    p.add_argument("--alpha",         type=float, default=DEFAULT_ALPHA)
    p.add_argument("--gamma",         type=float, default=DEFAULT_GAMMA)
    p.add_argument("--epsilon",       type=float, default=DEFAULT_EPS)
    p.add_argument("--epsilon-min",   type=float, default=DEFAULT_EPS_MIN)
    p.add_argument("--epsilon-decay", type=float, default=DEFAULT_EPS_DECAY)
    p.add_argument("--save-every",    type=int,   default=DEFAULT_SAVE_EVERY)
    p.add_argument("--print-every",   type=int,   default=DEFAULT_PRINT_EVERY)
    p.add_argument("--n-objects",     type=int,   default=DEFAULT_N_OBJECTS,
                   help="Number of random objects per episode (default 14)")
    p.add_argument("--obj-min",      type=int,   default=DEFAULT_OBJ_MIN,
                   help="Min object dimension in cells (default 1 = 20cm)")
    p.add_argument("--obj-max",      type=int,   default=DEFAULT_OBJ_MAX,
                   help="Max object dimension in cells (default 3 = 60cm)")
    p.add_argument("--no-objects",    action="store_true",
                   help="Train without objects (useful for initial debugging)")
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--cell-size",     type=float, default=0.20)
    p.add_argument("--sdf",           type=str,   default=DEFAULT_SDF,
                   help="Path to SDF world file defining the room")
    p.add_argument("--results-dir",   type=str,   default=DEFAULT_RESULTS_DIR,
                   help="Directory (relative to script) where checkpoints, "
                        "curves, and logs are written")
    return p.parse_args()


def moving_avg(v, w=50):
    if len(v) < w: return np.array(v, dtype=float)
    return np.convolve(v, np.ones(w)/w, mode="valid")


def save_curves(rewards, coverages, epsilons, path):
    # Stacking reward / coverage / epsilon-vs-episode into one figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(rewards, alpha=0.2, color="steelblue")
    axes[0].plot(moving_avg(rewards), color="steelblue", lw=2)
    axes[0].set_ylabel("Total reward"); axes[0].grid(alpha=0.3)

    axes[1].plot(np.array(coverages)*100, alpha=0.2, color="seagreen")
    axes[1].plot(moving_avg(np.array(coverages)*100), color="seagreen", lw=2)
    axes[1].axhline(100, color="seagreen", lw=1, ls="--", alpha=0.5)
    axes[1].set_ylabel("Coverage (%)"); axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epsilons, color="firebrick", lw=1.5)
    axes[2].set_ylabel("Epsilon"); axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Q-Learning: Boustrophedon + Random Unknown Obstacles",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"  → Saved curves to {path}")


def train(args):
    results_dir    = ROOT / args.results_dir
    checkpoint_fmt = str(results_dir / "q_table_ep{}.pkl")
    final_qtable   = str(results_dir / "q_table_final.pkl")
    curve_path     = str(results_dir / "training_curves.png")
    log_path       = str(results_dir / "training_log.csv")

    results_dir.mkdir(exist_ok=True)

    # Building the occupancy grid from the SDF
    wall_grid, meta = load_room_grid(Path(args.sdf), cell_size=args.cell_size)
    np.save(str(MAPS_DIR / "wall_grid.npy"), wall_grid)

    # objects are unknown to the agent
    n_objects = 0 if args.no_objects else args.n_objects  
    env = CoverageEnv(
        wall_grid,
        cell_size_m  = args.cell_size,
        n_objects    = n_objects,
        obj_min      = args.obj_min,
        obj_max      = args.obj_max,
    )

    agent = QLearningAgent(
        n_actions     = env.N_ACTIONS,
        alpha         = args.alpha,
        gamma         = args.gamma,
        epsilon       = args.epsilon,
        epsilon_min   = args.epsilon_min,
        epsilon_decay = args.epsilon_decay,
    )

    if args.resume and Path(final_qtable).exists():
        # warm-start from a previous run
        agent.load(final_qtable)  

    ep_rewards, ep_coverages, ep_epsilons = [], [], []

    with open(log_path, "w") as f:
        f.write("episode,reward,coverage_pct,epsilon,q_states\n")

    print(f"\n{'='*62}")
    print(f"  Q-Learning Coverage  –  {args.episodes:,} episodes")
    print(f"  Room grid: {env.rows}×{env.cols}  strips: {env.n_strips}")
    print(f"  Objects per episode: {n_objects}  "
          f"size {args.obj_min}–{args.obj_max} cells "
          f"({args.obj_min*10}–{args.obj_max*10}cm)  "
          f"({'OFF' if args.no_objects else 'random, unknown to agent'})")
    print(f"  State space: {env.n_strips}×2×2^{env.n_strips}×8 = "
          f"{env.n_strips*2*(2**env.n_strips)*8:,}")
    print(f"  α={args.alpha}  γ={args.gamma}  "
          f"ε:{args.epsilon}→{args.epsilon_min}  decay={args.epsilon_decay}")
    print(f"  Full coverage of all FLOOR cells is guaranteed by construction.")
    print(f"{'='*62}\n")

    t_start = time.time()

    for episode in range(1, args.episodes + 1):
        # new random object layout each episode
        state        = env.reset() 
        total_reward = 0.0
        done         = False

        # Pick a strip, step the env, update Q until all strips covered
        while not done:
            valid  = env.unvisited_strips()  # only choose among uncovered strips
            if not valid: break
            action = agent.select_action(state, valid_actions=valid)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done,
                         next_valid=env.unvisited_strips())
            state        = next_state
            total_reward += reward

        agent.decay_epsilon()  # decaying exploration toward greedy over training
        coverage = env.coverage_fraction()
        ep_rewards.append(total_reward)
        ep_coverages.append(coverage)
        ep_epsilons.append(agent.epsilon)

        with open(log_path, "a") as f:
            f.write(f"{episode},{total_reward:.3f},{coverage*100:.2f},"
                    f"{agent.epsilon:.5f},{agent.q_table_size}\n")

        if episode % args.print_every == 0:
            r   = ep_rewards[-args.print_every:]
            cov = ep_coverages[-args.print_every:]
            complete = sum(1 for c in cov if c >= 0.999)
            print(f"Ep {episode:5d}/{args.episodes}  "
                  f"avgR={np.mean(r):+8.1f}  "
                  f"avgCov={np.mean(cov)*100:5.1f}%  "
                  f"full={complete}/{args.print_every}  "
                  f"ε={agent.epsilon:.4f}  "
                  f"Q-states={agent.q_table_size}  "
                  f"{time.time()-t_start:.0f}s")

        # Periodically checkpointing the Q-table 
        if episode % args.save_every == 0:
            agent.save(checkpoint_fmt.format(episode))
            save_curves(ep_rewards, ep_coverages, ep_epsilons, curve_path)

    agent.save(final_qtable)  # final trained policy
    save_curves(ep_rewards, ep_coverages, ep_epsilons, curve_path)

    elapsed  = time.time() - t_start
    recent   = ep_coverages[-200:]
    complete = sum(1 for c in recent if c >= 0.999)
    print(f"\n{'='*62}")
    print(f"  Done  ({elapsed:.1f}s  {args.episodes/elapsed:.0f} eps/s)")
    print(f"  Best coverage            : {max(ep_coverages)*100:.1f}%")
    print(f"  Avg coverage (last 200)  : {np.mean(recent)*100:.1f}%")
    print(f"  Full coverage (last 200) : {complete}/200")
    print(f"  Q-table states           : {agent.q_table_size}")
    print(f"  Results in               : {results_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    train(parse_args())
