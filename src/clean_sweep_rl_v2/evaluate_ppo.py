import argparse, sys, time
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from coverage_env.room             import load_room_grid
from ppo_training.cell_env         import CellCoverageEnv
from ppo_training.feature_extractor import GridCNN

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

DEFAULT_RESULTS_DIR = "results_ppo_3"
MAPS_DIR            = ROOT / "maps"


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate trained PPO coverage agent")
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                   help="Directory (relative to script) holding the model "
                        "and where eval images are written")
    p.add_argument("--model",      default=None,
                   help="Path to PPO model zip (defaults to "
                        "<results-dir>/ppo_model_final)")
    p.add_argument("--episodes",   type=int,   default=5)
    p.add_argument("--n-objects",  type=int,   default=14)
    p.add_argument("--obj-min",    type=int,   default=1)
    p.add_argument("--obj-max",    type=int,   default=3)
    p.add_argument("--cell-size",  type=float, default=0.20)
    p.add_argument("--sdf", default=str(MAPS_DIR/"env1_bedroom.sdf"),
                   help="Path to SDF world file defining the room")
    p.add_argument("--render",     action="store_true",
                   help="Print ASCII grid at each step")
    p.add_argument("--out-dir",    default=None,
                   help="Where to write eval images (defaults to <results-dir>)")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy distribution instead "
                        "of taking the argmax (deterministic=False)")
    return p.parse_args()


def mask_fn(env):
    return env.action_masks()


def save_image(env, ep, path_out):
    max_vc = max(int(env.visit_count.max()), 2)

    log_max = np.log1p(max_vc - 1)

    display = np.ones((env.rows, env.cols, 3))
    for r in range(env.rows):
        for c in range(env.cols):
            if env.wall_grid[r, c] == 1:
                display[r, c] = [0.15, 0.15, 0.15]
            elif env.live_grid[r, c] == 1:
                display[r, c] = [0.85, 0.50, 0.15]
            elif env.visit_count[r, c] >= 1:
                t = (np.log1p(env.visit_count[r, c] - 1) / log_max
                     if log_max > 0 else 0)
                display[r, c] = [0.82 - 0.52*t,
                                 0.94 - 0.34*t,
                                 0.82 - 0.62*t]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(display, interpolation="nearest", origin="upper",
              extent=[-0.5, env.cols-0.5, env.rows-0.5, -0.5])

    # Drawing the robot's path as colour-graded line
    path = env.path
    if len(path) >= 2:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        t = np.linspace(0, 1, len(segs))
        lc = LineCollection(segs, cmap="plasma", linewidth=1.2, alpha=0.70)
        lc.set_array(t)
        ax.add_collection(lc)
        cb = plt.colorbar(lc, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("Path progression (start -> end)", fontsize=7)
        cb.set_ticks([0, 0.5, 1])
        cb.set_ticklabels(["Start", "Mid", "End"])

    # Visit count colourbar 
    if max_vc > 1:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                "v", [[0.82, 0.94, 0.82], [0.30, 0.60, 0.20]]),
            norm=LogNorm(vmin=1, vmax=max_vc))
        sm.set_array([])
        cb2 = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.10)
        cb2.ax.yaxis.set_ticks_position("left")
        cb2.ax.yaxis.set_label_position("left")
        cb2.set_label("Visit count", fontsize=7)
        cb2.set_ticks([1, max_vc])
        cb2.set_ticklabels(["1x", f"{max_vc}x"])
        cb2.minorticks_off()

    # Start and finish markers
    if path:
        sr, sc = path[0]
        fr, fc = path[-1]
        ax.plot(sc, sr, "o", color="red", markersize=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(fc, fr, "o", color="#1060E0", markersize=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)

    # Overlap stats
    floor_mask = (env.live_grid == 0)
    counts = env.visit_count[floor_mask]
    overlap_cells = int(np.sum(counts > 1))
    overlap_pct = 100.0 * overlap_cells / env.n_free if env.n_free else 0
    extra_visits = int(np.sum(np.maximum(counts - 1, 0)))
    max_visits = int(np.max(counts)) if len(counts) else 0

    # Legend
    handles = [
        mpatches.Patch(color=[0.82, 0.94, 0.82], label="Visited once"),
        mpatches.Patch(color=[0.30, 0.60, 0.20],
                       label=f"Overlap (max {max_vc}x)"),
        mpatches.Patch(color=[0.85, 0.50, 0.15], label="Object"),
        mpatches.Patch(color=[0.15, 0.15, 0.15], label="Wall"),
        plt.Line2D([0], [0], color="red", marker="o", ms=7,
                   linestyle="None", label="Start"),
        plt.Line2D([0], [0], color="#1060E0", marker="o", ms=7,
                   linestyle="None", label="Finish"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.02),
              ncol=len(handles),
              fontsize=7, framealpha=0.85, borderaxespad=0.0)

    cov = env.coverage_fraction()
    n_objs = int(np.sum((env.live_grid == 1) & (env.wall_grid == 0)))
    ax.set_title(
        f"PPO Episode {ep}  |  coverage={cov*100:.1f}%  "
        f"|  steps={env.nb_steps}  |  obstacle cells={n_objs}\n"
        f"Overlap: {overlap_cells} cells ({overlap_pct:.1f}% of floor)  "
        f"|  extra visits: {extra_visits}  |  max: {max_visits}x",
        fontsize=8, pad=8)
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()

    return dict(overlap_cells=overlap_cells, overlap_pct=overlap_pct,
                extra_visits=extra_visits, max_visits=max_visits)


def evaluate(args):
    results_dir = ROOT / args.results_dir
    out_dir     = Path(args.out_dir) if args.out_dir else results_dir
    out_dir.mkdir(exist_ok=True)
    model_path = args.model or str(results_dir / "ppo_model_final")

    wall_grid, _ = load_room_grid(Path(args.sdf), cell_size=args.cell_size)

    env = CellCoverageEnv(
        wall_grid, cell_size_m=args.cell_size,
        n_objects=args.n_objects,
        obj_min=args.obj_min, obj_max=args.obj_max,
    )
    masked_env = ActionMasker(env, mask_fn)

    model = MaskablePPO.load(model_path, env=masked_env)
    print(f"Loaded model: {model_path}")

    # if less than 3 unique cells in the last 8 steps, 
    # then the agent is looping
    LOOP_WINDOW = 8
    LOOP_THRESHOLD = 3  

    all_cov, all_ovl, all_steps = [], [], []
    all_ppo_actions, all_bfs_actions = [], []

    for ep in range(1, args.episodes + 1):
        obs, _ = masked_env.reset()
        done = False
        truncated = False
        total_r = 0.0
        recent_positions = []
        ppo_actions = 0
        bfs_actions = 0

        print(f"\n-- Episode {ep}  (start={env.pos}  "
              f"free_cells={env.n_free}) --")

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=not args.stochastic,
                                      action_masks=env.action_masks())

            # If a loop is detected (the agent is stuck), override with BFS
            recent_positions.append(env.pos)
            if len(recent_positions) > LOOP_WINDOW:
                recent_positions.pop(0)

            if len(recent_positions) == LOOP_WINDOW:
                unique_recent = len(set(recent_positions))
                if unique_recent <= LOOP_THRESHOLD:
                    bfs_action = env.bfs_to_nearest_unvisited()
                    if bfs_action is not None:
                        action = bfs_action
                        bfs_actions += 1
                    else:
                        ppo_actions += 1
                else:
                    ppo_actions += 1
            else:
                ppo_actions += 1

            obs, reward, done, truncated, info = masked_env.step(action)
            total_r += reward

            if args.render and env.nb_steps % 50 == 0:
                env.render()
                print()

        cov = env.coverage_fraction()
        total_actions = ppo_actions + bfs_actions
        all_cov.append(cov)
        all_steps.append(env.nb_steps)
        all_ppo_actions.append(ppo_actions)
        all_bfs_actions.append(bfs_actions)

        status = "COMPLETE" if cov >= 0.999 else "INCOMPLETE"
        if truncated and not done:
            status = "TIMEOUT"
        ppo_pct = 100 * ppo_actions / total_actions if total_actions else 0
        print(f"  steps={env.nb_steps}  reward={total_r:.1f}  "
              f"coverage={cov*100:.1f}%  {status}")
        print(f"  actions: PPO={ppo_actions} ({ppo_pct:.0f}%)  "
              f"BFS-assist={bfs_actions} ({100-ppo_pct:.0f}%)")

        img_path = str(out_dir / f"ppo_eval_ep{ep}.png")
        ovl = save_image(env, ep, img_path)
        all_ovl.append(ovl)
        print(f"  overlap: {ovl['overlap_cells']} cells "
              f"({ovl['overlap_pct']:.1f}%)  "
              f"extra_visits={ovl['extra_visits']}  "
              f"max={ovl['max_visits']}x")
        print(f"  -> {img_path}")

    # Summary
    total_ppo = sum(all_ppo_actions)
    total_bfs = sum(all_bfs_actions)
    total_all = total_ppo + total_bfs
    print(f"\n{'='*62}")
    print(f"  Summary ({args.episodes} episodes, {args.n_objects} objects each)")
    print(f"{'='*62}")
    print(f"  Mean coverage       : {np.mean(all_cov)*100:.1f}%")
    print(f"  Min  coverage       : {np.min(all_cov)*100:.1f}%")
    print(f"  Full coverage       : "
          f"{sum(1 for c in all_cov if c >= 0.999)}/{args.episodes}")
    print(f"  Mean steps          : {np.mean(all_steps):.0f}")
    print(f"  Actions             : PPO={total_ppo} "
          f"({100*total_ppo/total_all:.0f}%)  "
          f"BFS-assist={total_bfs} "
          f"({100*total_bfs/total_all:.0f}%)" if total_all else "")
    print(f"  Mean overlap cells  : "
          f"{np.mean([o['overlap_cells'] for o in all_ovl]):.1f}")
    print(f"  Mean overlap %      : "
          f"{np.mean([o['overlap_pct'] for o in all_ovl]):.1f}%")
    print(f"  Mean extra visits   : "
          f"{np.mean([o['extra_visits'] for o in all_ovl]):.1f}")
    print(f"{'='*62}")


if __name__ == "__main__":
    evaluate(parse_args())
