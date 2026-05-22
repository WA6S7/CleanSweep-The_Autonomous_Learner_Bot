import argparse, sys, time
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))  # so the coverage_env / q_learning packages import

from coverage_env.room         import load_room_grid
from coverage_env.coverage_env import CoverageEnv
from q_learning.q_agent        import QLearningAgent

DEFAULT_RESULTS_DIR = "results_20cm_cells"
MAPS_DIR            = ROOT / "maps"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                   help="Directory (relative to script) holding q_table_final.pkl "
                        "and where eval images are written")
    p.add_argument("--qtable",     default=None,
                   help="Path to Q-table pickle (defaults to "
                        "<results-dir>/q_table_final.pkl)")
    p.add_argument("--episodes",   type=int,   default=5)
    p.add_argument("--n-objects",  type=int,   default=14)
    p.add_argument("--obj-min",    type=int,   default=1)
    p.add_argument("--obj-max",    type=int,   default=3)
    p.add_argument("--render",     action="store_true")
    p.add_argument("--cell-size",  type=float, default=0.20)
    p.add_argument("--sdf", default=str(MAPS_DIR/"env1_bedroom.sdf"),
                   help="Path to SDF world file defining the room")
    return p.parse_args()


def save_image(env, ep, path_out):
    # Rendering one episode as a coverage map (walls, objects, visit heat, path)
    from matplotlib.collections import LineCollection

    # Building the RGB grid: walls grey, objects orange, visited cells shaded by count
    max_vc = max(int(env.visit_count.max()), 2)
    display = np.ones((env.rows, env.cols, 3))
    for r in range(env.rows):
        for c in range(env.cols):
            if env.wall_grid[r, c] == 1:
                display[r, c] = [0.15, 0.15, 0.15]
            elif env.live_grid[r, c] == 1:
                display[r, c] = [0.85, 0.50, 0.15]
            elif env.visit_count[r, c] >= 1:
                # Darker green = visited more often
                t = (env.visit_count[r, c] - 1) / (max_vc - 1) if max_vc > 1 else 0
                display[r, c] = [0.82 - 0.52*t, 0.94 - 0.34*t, 0.82 - 0.62*t]

    fig, ax = plt.subplots(figsize=(8, 9))
    ax.imshow(display, interpolation="nearest", origin="upper",
              extent=[-0.5, env.cols-0.5, env.rows-0.5, -0.5])

    for s in env.strips:
        ax.axhline(s.rows[0]-0.5, color="#E07820", lw=0.8, ls="--", alpha=0.45)

    # Drawing the robot's path as a colour-graded line (start -> end)
    path = env.path
    if len(path) >= 2:
        xs = [p[1] for p in path]; ys = [p[0] for p in path]
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        t      = np.linspace(0, 1, len(segs))  # colour index along the path
        lc = LineCollection(segs, cmap="plasma", linewidth=1.2, alpha=0.70)
        lc.set_array(t)
        ax.add_collection(lc)
        cb = plt.colorbar(lc, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("Path progression (start → end)", fontsize=7)
        cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(["Start", "Mid", "End"])

    if max_vc > 1:
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.colors.LinearSegmentedColormap.from_list(
                "v", [[0.82,0.94,0.82],[0.30,0.60,0.20]]),
            norm=plt.Normalize(vmin=1, vmax=max_vc))
        sm.set_array([])
        cb2 = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.10)
        cb2.ax.yaxis.set_ticks_position("left")
        cb2.ax.yaxis.set_label_position("left")
        cb2.set_label("Visit count", fontsize=7)
        cb2.set_ticks([1, max_vc]); cb2.set_ticklabels(["1×", f"{max_vc}×"])

    # Marking start (red) and finish (blue) cells
    if path:
        sr,sc = path[0]; fr,fc = path[-1]
        ax.plot(sc, sr, "o", color="red",     markersize=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.plot(fc, fr, "o", color="#1060E0", markersize=9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)

    ovl = env.overlap_stats()
    handles = [
        mpatches.Patch(color=[0.82,0.94,0.82], label="Visited once"),
        mpatches.Patch(color=[0.30,0.60,0.20], label=f"Overlap (max {max_vc}×)"),
        mpatches.Patch(color=[0.85,0.50,0.15], label="Object"),
        mpatches.Patch(color=[0.15,0.15,0.15], label="Wall"),
        plt.Line2D([0],[0], color="red",     marker="o", ms=7,
                   linestyle="None", label="Start"),
        plt.Line2D([0],[0], color="#1060E0", marker="o", ms=7,
                   linestyle="None", label="Finish"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.02),
              ncol=len(handles),
              fontsize=7, framealpha=0.85, borderaxespad=0.0)

    cov = env.coverage_fraction()
    ax.set_title(
        f"Episode {ep}  |  coverage={cov*100:.1f}%  |  steps={env.nb_steps}  "
        f"|  objects={len(env._placements)}\n"
        f"Overlap: {ovl['overlap_cells']} cells  "
        f"({ovl['overlap_pct']:.1f}% of floor)  "
        f"= {ovl['overlap_area_m2']*1e4:.0f} cm²  "
        f"|  extra visits: {ovl['extra_visits']}",
        fontsize=8, pad=8)
    ax.set_xlim(-0.5, env.cols-0.5); ax.set_ylim(env.rows-0.5, -0.5)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    return len(path), ovl


def evaluate(args):
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(exist_ok=True)
    qtable_path = args.qtable or str(results_dir / "q_table_final.pkl")

    # Loading the trained Q-table and running it greedily (no exploration)
    wall_grid, _ = load_room_grid(Path(args.sdf), cell_size=args.cell_size)
    env   = CoverageEnv(wall_grid, cell_size_m=args.cell_size,
                        n_objects=args.n_objects,
                        obj_min=args.obj_min, obj_max=args.obj_max)
    agent = QLearningAgent(n_actions=env.N_ACTIONS)
    agent.load(qtable_path)
    agent.epsilon = 0.0  # pure exploitation for evaluation

    all_cov = []
    all_ovl = []

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done  = False; total_r = 0.0; order = []

        print(f"\n── Episode {ep}  "
              f"(start={env.pos}  objects={len(env._placements)}) ──")

        # Always taking the best valid strip
        while not done:
            valid  = env.unvisited_strips()
            if not valid: break
            action = agent.select_action(state, valid_actions=valid, greedy=True)
            order.append(action)
            state, reward, done, info = env.step(action)
            total_r += reward
            if args.render:
                env.render_ascii(); time.sleep(0.05)  

        cov = env.coverage_fraction()
        ovl = env.overlap_stats()
        all_cov.append(cov)
        all_ovl.append(ovl)

        print(f"  strip order  : {order}")
        print(f"  steps={env.nb_steps}  reward={total_r:.1f}  "
              f"coverage={cov*100:.1f}%  "
              f"{'✓ COMPLETE' if cov >= 0.999 else '✗ INCOMPLETE'}")
        print(f"  overlap      : {ovl['overlap_cells']} cells  "
              f"({ovl['overlap_pct']:.1f}% of floor)  "
              f"= {ovl['overlap_area_m2']*1e4:.0f} cm²  "
              f"| extra visits: {ovl['extra_visits']}  "
              f"| max: {ovl['max_visits']}×")

        out = str(results_dir / f"eval_ep{ep}_final.png")
        n_path, _ = save_image(env, ep, out)
        print(f"  → {out}  (path length: {n_path} positions)")

    # Aggregating coverage / overlap stats across all episodes
    print(f"\n── Summary ({args.episodes} episodes, {args.n_objects} objects each) ──")
    print(f"  Mean coverage      : {np.mean(all_cov)*100:.1f}%")
    print(f"  Min  coverage      : {np.min(all_cov)*100:.1f}%")
    print(f"  Full coverage      : {sum(1 for c in all_cov if c>=0.999)}/{args.episodes}")
    print(f"  ── Overlap ──")
    print(f"  Mean overlap cells : {np.mean([o['overlap_cells']   for o in all_ovl]):.1f}")
    print(f"  Mean overlap %     : {np.mean([o['overlap_pct']     for o in all_ovl]):.1f}%")
    print(f"  Mean overlap area  : {np.mean([o['overlap_area_m2'] for o in all_ovl])*1e4:.0f} cm²")
    print(f"  Mean extra visits  : {np.mean([o['extra_visits']    for o in all_ovl]):.1f}")


if __name__ == "__main__":
    evaluate(parse_args())
