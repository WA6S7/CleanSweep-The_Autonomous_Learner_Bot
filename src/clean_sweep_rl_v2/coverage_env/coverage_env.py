import numpy as np
import random
from collections import deque
from typing import List, Tuple

from coverage_env.object_spawner import spawn_objects

DISCOVER_REWARD = 1.0
MOVE_PUNISHMENT = 0.05
COVERAGE_REWARD = 50.0
STRIP_HEIGHT    = 4
N_OBJECTS       = 14
OBJ_MIN_SIZE    = 1
OBJ_MAX_SIZE    = 3

HEADING_DELTA = {0:(-1,0), 1:(0,1), 2:(1,0), 3:(0,-1)}
def turn_left(h):  return (h-1)%4
def turn_right(h): return (h+1)%4
def opposite(h):   return (h+2)%4


class Strip:
    def __init__(self, sid, rows, all_free):
        self.id         = sid
        self.rows       = rows
        self.base_cells = [c for c in all_free if c[0] in set(rows)]
        self.col_min    = min(c[1] for c in self.base_cells) if self.base_cells else 0
        self.col_max    = max(c[1] for c in self.base_cells) if self.base_cells else 0

    @property
    def is_empty(self): return len(self.base_cells) == 0


class CoverageEnv:

    def __init__(self, wall_grid, cell_size_m=0.20, strip_height=STRIP_HEIGHT,
                 n_objects=N_OBJECTS, obj_min=OBJ_MIN_SIZE, obj_max=OBJ_MAX_SIZE):

        self.wall_grid       = wall_grid.astype(np.int8)
        self.rows, self.cols = wall_grid.shape
        self.cell_size       = cell_size_m
        self.n_objects       = n_objects
        self.obj_min         = obj_min
        self.obj_max         = obj_max

        self._wall_free = [(r,c) for r in range(self.rows)
                                  for c in range(self.cols)
                                  if wall_grid[r,c] == 0]

        self.strips    = self._build_strips(strip_height)
        self.n_strips  = len(self.strips)
        self.N_ACTIONS = self.n_strips

        state_space = self.n_strips * 2 * (2**self.n_strips) * 8
        print(f"[CoverageEnv] grid={self.rows}×{self.cols}  "
              f"interior={len(self._wall_free)}  strips={self.n_strips}  "
              f"objects_per_ep={n_objects}  "
              f"obj_size={obj_min}–{obj_max}cells ({obj_min*10}–{obj_max*10}cm)  "
              f"state_space={state_space:,}")

        # Episode state
        self.live_grid       = wall_grid.copy()
        self.discovered_grid = wall_grid.copy()
        self.visited         = np.zeros_like(wall_grid)            # binary: entered at all
        self.visit_count     = np.zeros_like(wall_grid, dtype=np.int32)  # total entries
        self.free_cells      = list(self._wall_free)
        self.n_free          = len(self.free_cells)
        self.pos             = self._wall_free[0]
        self.heading         = 1
        self.nb_steps        = 0
        self._strip_visited  = [False] * self.n_strips
        self._current_strip  = 0
        self._entry_side     = 0
        self._placements     = []
        self.path            = []

    # --- Strip structure ---

    def _build_strips(self, strip_height):
        strips, sid, r = [], 0, 0
        while r < self.rows:
            row_range = list(range(r, min(r+strip_height, self.rows)))
            s = Strip(sid, row_range, self._wall_free)
            if not s.is_empty:
                strips.append(s); sid += 1
            r += strip_height
        return strips

    # --- Reset ---

    def reset(self):
        spawn = random.choice(self._wall_free)

        self.live_grid, self._placements = spawn_objects(
            self.wall_grid, n_objects=self.n_objects,
            obj_min=self.obj_min, obj_max=self.obj_max, robot_pos=spawn)

        self.free_cells = [(r,c) for r in range(self.rows)
                                  for c in range(self.cols)
                                  if self.live_grid[r,c] == 0]
        self.n_free = len(self.free_cells)

        self.visited[:]         = 0
        self.visit_count[:]     = 0
        self.discovered_grid[:] = self.wall_grid.copy()
        self.nb_steps           = 0
        self._strip_visited     = [False] * self.n_strips
        self.pos                = spawn
        self.heading            = random.randint(0, 3)
        self.path               = [spawn]

        # Count the spawn cell as entered
        self._enter_cell(spawn[0], spawn[1])

        self._current_strip = self._strip_for_pos(spawn)
        self._entry_side    = 0
        return self.get_state_key()

    def _strip_for_pos(self, pos):
        for s in self.strips:
            if pos[0] in s.rows: return s.id
        return 0

    # --- Enter a cell ---

    def _enter_cell(self, r, c):
        # First entry: +DISCOVER_REWARD and marks visited.
        # Subsequent entries: -MOVE_PUNISHMENT (overlap penalty).
        
        self.visit_count[r, c] += 1
        if self.visited[r, c] == 0:
            self.visited[r, c] = 1
            return DISCOVER_REWARD
        return -MOVE_PUNISHMENT

    # --- BFS pathfinder ---

    def _bfs(self, start, goal):
        # BFS on live_grid
        # Returns the step list from the start to the goal 
        if start == goal:
            return []
        prev  = {start: None}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                break
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nb = (r+dr, c+dc)
                nr, nc = nb
                if (0 <= nr < self.rows and 0 <= nc < self.cols
                        and nb not in prev
                        and self.live_grid[nr, nc] == 0):
                    prev[nb] = (r, c)
                    queue.append(nb)
        if goal not in prev:
            return []
        path, cur = [], goal
        while cur != start:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def _walk(self, bfs_steps):
        steps, reward = 0, 0.0
        for nr, nc in bfs_steps:
            self._sense_cell(nr, nc)
            self.pos = (nr, nc)
            reward += self._enter_cell(nr, nc)
            self.path.append(self.pos)
            steps  += 1

            # Constant movement cost 
            reward -= MOVE_PUNISHMENT   
        return steps, reward

    # --- Sensors ---

    def _sense_cell(self, r, c):
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return 1
        if self.live_grid[r, c] == 1 and self.wall_grid[r, c] == 0:
            self.discovered_grid[r, c] = 1
        return int(self.live_grid[r, c] == 1)

    def _get_sensors(self):
        r, c = self.pos; h = self.heading
        fdr,fdc = HEADING_DELTA[h]
        ldr,ldc = HEADING_DELTA[turn_left(h)]
        rdr,rdc = HEADING_DELTA[turn_right(h)]
        return (self._sense_cell(r+fdr,c+fdc),
                self._sense_cell(r+ldr,c+ldc),
                self._sense_cell(r+rdr,c+rdc))

    # --- State ---

    def get_state_key(self):
        mask = sum(1<<i for i,v in enumerate(self._strip_visited) if v)
        f,l,r = self._get_sensors()
        return (self._current_strip, self._entry_side, mask, f, l, r)

    def unvisited_strips(self):
        return [i for i,s in enumerate(self.strips)
                if not s.is_empty and not self._strip_visited[i]]

    # --- Step ---

    def step(self, action):
        strip = self.strips[action]
        if self._strip_visited[action] or strip.is_empty:
            return self.get_state_key(), -1.0, False, {"revisit": True}

        nav_s, nav_r      = self._navigate_to(strip)
        sw_s, sw_r, newly = self._sweep_strip(strip)

        self._strip_visited[action] = True
        self._current_strip         = action
        self.nb_steps              += nav_s + sw_s
        total_r = nav_r + sw_r

        done = len(self.unvisited_strips()) == 0
        if done: total_r += COVERAGE_REWARD

        return self.get_state_key(), total_r, done, {
            "newly_visited": newly, "steps": nav_s + sw_s}

    # --- Navigation ---

    def _navigate_to(self, target):
        # BFS walking to the nearest free cell in target strip
        if not target.base_cells:
            return 0, 0.0
        reachable = [c for c in target.base_cells if self.live_grid[c[0],c[1]] == 0]
        if not reachable:
            return 0, 0.0
        r, c = self.pos
        dest  = min(reachable, key=lambda x: abs(x[0]-r)+abs(x[1]-c))
        route = self._bfs(self.pos, dest)
        s, rw = self._walk(route)
        mid = (target.col_min + target.col_max) // 2
        self._entry_side = 0 if self.pos[1] <= mid else 1
        return s, rw

    # --- Sweep ---

    def _sweep_strip(self, strip):
        steps, reward, newly_visited = 0, 0.0, 0
        go_right = (self._entry_side == 0)

        for row in strip.rows:
            all_cols = [c for c in range(self.cols) if self.wall_grid[row, c] == 0]
            ordered  = all_cols if go_right else list(reversed(all_cols))

            for col in ordered:
                # Sense before approaching
                if self._sense_cell(row, col):
                    steps += 1
                    continue   # object — skip

                if self.pos != (row, col):
                    route = self._bfs(self.pos, (row, col))
                    if not route:
                        steps += 1
                        continue   # unreachable
                    s, rw = self._walk(route)
                    steps  += s
                    reward += rw
                    
                else:
                    # Already at this cell — count it if not yet entered
                    pass  

                # Track newly_visited for info dict
                if self.visit_count[row, col] == 1 and self.visited[row, col] == 1:
                    newly_visited += 1

            go_right = not go_right

        return steps, reward, newly_visited

    # --- Metrics ---

    def coverage_fraction(self):
        """Fraction of floor cells entered at least once."""
        return float(np.sum(self.visited)) / self.n_free

    def overlap_stats(self):
        floor_mask   = (self.live_grid == 0)
        counts       = self.visit_count[floor_mask]
        cell_area    = self.cell_size ** 2

        overlap_cells = int(np.sum(counts > 1))
        extra_visits  = int(np.sum(np.maximum(counts - 1, 0)))

        return dict(
            overlap_cells    = overlap_cells,
            overlap_pct      = 100.0 * overlap_cells / self.n_free if self.n_free else 0.0,
            overlap_area_m2  = overlap_cells * cell_area,
            extra_visits     = extra_visits,
            max_visits       = int(np.max(counts)) if len(counts) else 0,
            mean_visits      = float(np.mean(counts)) if len(counts) else 0.0,
            cell_size_m2     = cell_area,
        )

    def render_ascii(self):
        hchar = ["^",">","v","<"][self.heading]
        strip_rows = {s.rows[0] for s in self.strips if s.rows}
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                if (row,col) == self.pos:          line += hchar
                elif self.live_grid[row,col] == 1: line += "#" if self.wall_grid[row,col]==1 else "X"
                elif self.visit_count[row,col] > 1: line += "+"   # overlap
                elif self.visited[row,col] == 1:   line += "o"
                else:                              line += "."
            print(line + (" |" if row in strip_rows else ""))
        print(f"coverage={self.coverage_fraction()*100:.1f}%  "
              f"strips={sum(self._strip_visited)}/{self.n_strips}  "
              f"overlap={self.overlap_stats()['overlap_pct']:.1f}%\n")
