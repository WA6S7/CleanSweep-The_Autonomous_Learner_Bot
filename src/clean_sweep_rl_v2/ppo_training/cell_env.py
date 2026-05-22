import gymnasium as gym
import numpy as np
import random
from collections import deque
from gymnasium import spaces

from coverage_env.object_spawner import spawn_objects

# Movements indexed by action: UP, RIGHT, DOWN, LEFT
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

DISCOVER_REWARD  = 1.0
REVISIT_PENALTY  = 0.3
STEP_COST        = 0.02
COVERAGE_BONUS   = 50.0
SHAPING_SCALE    = 0.0


class CellCoverageEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, wall_grid, cell_size_m=0.20,
                 n_objects=14, obj_min=1, obj_max=3,
                 max_steps_factor=4):
        super().__init__()
        self.wall_grid = wall_grid.astype(np.int8)
        self.rows, self.cols = wall_grid.shape
        self.cell_size = cell_size_m
        self.n_objects = n_objects
        self.obj_min = obj_min
        self.obj_max = obj_max
        self.max_steps_factor = max_steps_factor
        self._max_dist = float(self.rows + self.cols)

        self._wall_free = [(r, c) for r in range(self.rows)
                           for c in range(self.cols)
                           if wall_grid[r, c] == 0]

        # Observation: 4-channel grid
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(4, self.rows, self.cols),
            dtype=np.float32,
        )
        # Action: UP(0), RIGHT(1), DOWN(2), LEFT(3)
        self.action_space = spaces.Discrete(4)

        # Episode state (initialized in reset)
        self.live_grid = None
        self.discovered_grid = None
        self.visited = None
        self.visit_count = None
        self.pos = None
        self.path = []
        self.n_free = 0
        self.n_visited = 0
        self.nb_steps = 0
        self.max_steps = 0
        self._frontier_dist = None  # cached frontier distance field
        self._frontier_dirty = True  # recompute flag

        n_wall_free = len(self._wall_free)
        print(f"[CellCoverageEnv] grid={self.rows}x{self.cols}  "
              f"interior={n_wall_free}  cell={cell_size_m}m  "
              f"objects/ep={n_objects}  obj_size={obj_min}-{obj_max}cells  "
              f"obs_shape={self.observation_space.shape}  actions=4")

    # --- Curriculum ---
    def set_n_objects(self, n):
        # Change the number of random objects for subsequent episodes
        self.n_objects = n

    # --- Sensing ---
    def _sense_adjacent(self):
        # Sense all 4 adjacent cells. If a cell contains an unknown object, mark it in discovered_grid
        r, c = self.pos
        for dr, dc in DELTAS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if (self.live_grid[nr, nc] == 1
                        and self.wall_grid[nr, nc] == 0):
                    self.discovered_grid[nr, nc] = 1

    # --- Frontier distance (BFS) ---
    def _compute_frontier_dist(self):
        dist = np.full((self.rows, self.cols), self._max_dist, dtype=np.float32)
        queue = deque()

        # Seed BFS from every unvisited free cell
        for r in range(self.rows):
            for c in range(self.cols):
                if self.live_grid[r, c] == 0 and self.visited[r, c] == 0:
                    dist[r, c] = 0.0
                    queue.append((r, c))

        # BFS expansion
        while queue:
            r, c = queue.popleft()
            nd = dist[r, c] + 1.0
            for dr, dc in DELTAS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and self.live_grid[nr, nc] == 0
                        and nd < dist[nr, nc]):
                    dist[nr, nc] = nd
                    queue.append((nr, nc))

        # Normalize to [0, 1]
        dist /= self._max_dist
        self._frontier_dist = dist
        self._frontier_dirty = False
        return dist

    def _get_frontier_dist_at(self, r, c):
        """Get frontier distance at a specific cell. Recomputes if dirty."""
        if self._frontier_dirty:
            self._compute_frontier_dist()
        return self._frontier_dist[r, c]

    # BFS to nearest unvisited (for a loop escape) 
    def bfs_to_nearest_unvisited(self):
        """BFS from current position to the nearest unvisited free cell.
        Returns the action (0-3) for the first step, or None if unreachable."""
        start = self.pos
        prev = {start: None}
        queue = deque([start])
        goal = None

        while queue:
            r, c = queue.popleft()
            if self.live_grid[r, c] == 0 and self.visited[r, c] == 0 and (r, c) != start:
                goal = (r, c)
                break
            for dr, dc in DELTAS:
                nb = (r + dr, c + dc)
                nr, nc = nb
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and nb not in prev
                        and self.live_grid[nr, nc] == 0):
                    prev[nb] = (r, c)
                    queue.append(nb)

        if goal is None:
            return None

        # Trace back to find first step from start
        cur = goal
        while prev[cur] != start:
            cur = prev[cur]
        first_step = cur

        # Convert to action index
        dr = first_step[0] - start[0]
        dc = first_step[1] - start[1]
        for i, (ddr, ddc) in enumerate(DELTAS):
            if ddr == dr and ddc == dc:
                return i
        return None

    # --- Observation ---
    def _obs(self):
        obs = np.zeros((4, self.rows, self.cols), dtype=np.float32)
        obs[0] = self.discovered_grid.astype(np.float32)
        obs[1] = self.visited.astype(np.float32)
        obs[2, self.pos[0], self.pos[1]] = 1.0
        if self._frontier_dirty:
            self._compute_frontier_dist()
        obs[3] = self._frontier_dist
        return obs

    # --- Action masking ---
    def action_masks(self):
        """Boolean mask: True = action allowed.
        Prevents moves into walls/discovered obstacles and out of bounds."""
        r, c = self.pos
        mask = np.zeros(4, dtype=bool)
        for i, (dr, dc) in enumerate(DELTAS):
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.rows and 0 <= nc < self.cols
                    and self.discovered_grid[nr, nc] == 0):
                mask[i] = True
        return mask

    # --- Reset ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        spawn = random.choice(self._wall_free)

        self.live_grid, _ = spawn_objects(
            self.wall_grid, n_objects=self.n_objects,
            obj_min=self.obj_min, obj_max=self.obj_max, robot_pos=spawn)

        self.n_free = int((self.live_grid == 0).sum())
        self.max_steps = self.max_steps_factor * self.n_free

        self.discovered_grid = self.wall_grid.copy()
        self.visited = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.visit_count = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.pos = spawn
        self.path = [spawn]
        self.nb_steps = 0

        # Mark spawn cell as visited
        self.visited[spawn[0], spawn[1]] = 1
        self.visit_count[spawn[0], spawn[1]] = 1
        self.n_visited = 1

        # Sense obstacles adjacent to spawn
        self._sense_adjacent()

        # Compute initial frontier distances
        self._frontier_dirty = True
        self._compute_frontier_dist()

        return self._obs(), {}

    # --- Step ---
    def step(self, action):
        dr, dc = DELTAS[int(action)]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.nb_steps += 1

        # Frontier distance before move
        dist_before = self._get_frontier_dist_at(self.pos[0], self.pos[1])

        reward = -STEP_COST

        # Check if the move is valid
        in_bounds = (0 <= nr < self.rows and 0 <= nc < self.cols)
        blocked = (not in_bounds) or (self.live_grid[nr, nc] == 1)

        if blocked:
            # Discover the obstacle if it was unknown
            if in_bounds and self.wall_grid[nr, nc] == 0:
                self.discovered_grid[nr, nc] = 1
            # Stay in place results in a penalty
            reward -= REVISIT_PENALTY
        else:
            # Move to a new cell
            self.pos = (nr, nc)
            self.path.append(self.pos)
            self.visit_count[nr, nc] += 1

            if self.visited[nr, nc] == 0:
                self.visited[nr, nc] = 1
                self.n_visited += 1
                reward += DISCOVER_REWARD
                # Map of distances to nearest unvisited is now wrong, so recalculate when needed
                self._frontier_dirty = True
            else:
                reward -= REVISIT_PENALTY

            # Sense obstacles from new position
            self._sense_adjacent()

        # Potential-based reward shaping - reward for moving closer to frontier
        dist_after = self._get_frontier_dist_at(self.pos[0], self.pos[1])
        reward += SHAPING_SCALE * (dist_before - dist_after)

        # Termination
        done = (self.n_visited >= self.n_free)
        truncated = (not done) and (self.nb_steps >= self.max_steps)

        if done:
            reward += COVERAGE_BONUS

        info = {
            "coverage": self.n_visited / self.n_free if self.n_free > 0 else 1.0,
            "steps": self.nb_steps,
            "n_visited": self.n_visited,
            "n_free": self.n_free,
        }

        return self._obs(), reward, done, truncated, info

    # --- Metrics ---
    def coverage_fraction(self):
        return self.n_visited / self.n_free if self.n_free > 0 else 1.0

    def overlap_count(self):
        # Number of cells visited more than once
        floor_mask = (self.live_grid == 0)
        return int(np.sum(self.visit_count[floor_mask] > 1))

    # --- Render ---
    def render(self):
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                if (row, col) == self.pos:
                    line += "R"
                elif self.live_grid[row, col] == 1:
                    line += "#" if self.wall_grid[row, col] == 1 else "X"
                elif self.visit_count[row, col] > 1:
                    line += "+"
                elif self.visited[row, col] == 1:
                    line += "o"
                else:
                    line += "."
            print(line)
        cov = self.coverage_fraction() * 100
        olap = self.overlap_count()
        print(f"coverage={cov:.1f}%  steps={self.nb_steps}/{self.max_steps}  "
              f"overlap_cells={olap}")
