import numpy as np
import random
from typing import List, Tuple, Optional

try:
    from scipy.ndimage import label as ndlabel
    _SCIPY = True
except ImportError:
    _SCIPY = False

# Default size range (cells)
OBJ_MIN_SIZE   = 1   # 1×1 cells = 20×20 cm 
OBJ_MAX_SIZE   = 3   # 3×3 cells = 60×60 cm  
N_OBJECTS      = 14
WALL_CLEARANCE = 1


def _is_connected(grid: np.ndarray) -> bool:
    # True if all the free cells form a single connected component
    free_mask = (grid == 0)
    if not free_mask.any():
        return False
    if _SCIPY:
        _, n = ndlabel(free_mask)
        return n == 1
    
    # BFS fallback
    free_cells = list(zip(*np.where(free_mask)))
    visited = np.zeros_like(grid, dtype=bool)
    stack = [free_cells[0]]
    visited[free_cells[0]] = True
    rows, cols = grid.shape
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and not visited[nr,nc] and grid[nr,nc]==0:
                visited[nr,nc] = True
                stack.append((nr,nc))
    return count == int(free_mask.sum())


def _fits(grid: np.ndarray, r: int, c: int,
          h: int, w: int, clearance: int) -> bool:
    rows, cols = grid.shape
    r0 = r - clearance
    r1 = r + h + clearance
    c0 = c - clearance
    c1 = c + w + clearance
    if r0 < 0 or r1 > rows or c0 < 0 or c1 > cols:
        return False  # margin would fall outside the grid
    return bool(np.all(grid[r0:r1, c0:c1] == 0))


def spawn_objects(wall_grid:  np.ndarray,
                  n_objects:  int = N_OBJECTS,
                  obj_min:    int = OBJ_MIN_SIZE,
                  obj_max:    int = OBJ_MAX_SIZE,
                  clearance:  int = WALL_CLEARANCE,
                  robot_pos:  Optional[Tuple[int,int]] = None,
                  ) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    
    # Randomly placing n_objects rectangular obstacles onto a copy of the wall grid
    rows, cols = wall_grid.shape
    live_grid  = wall_grid.copy()
    placements: List[Tuple[int,int,int,int]] = []

    placed = 0
    max_attempts = n_objects * 300   # give up after this many tries

    # Rejection sampling by trying random size/position until enough fit
    attempts = 0
    while placed < n_objects and attempts < max_attempts:
        attempts += 1

        # Sampling a random size for this object
        h = random.randint(obj_min, obj_max)
        w = random.randint(obj_min, obj_max)

        # Sampling a random position
        r = random.randint(clearance, rows - h - clearance)
        c = random.randint(clearance, cols - w - clearance)

        # Check clearance
        if not _fits(live_grid, r, c, h, w, clearance):
            continue

        # Checking that the robot spawn cell is not covered
        if robot_pos is not None:
            rr, rc = robot_pos
            if r <= rr < r+h and c <= rc < c+w:
                continue

        live_grid[r:r+h, c:c+w] = 1

        # Connectivity check, rejects object if it isolates any free cells
        if not _is_connected(live_grid):
            live_grid[r:r+h, c:c+w] = 0   # undo and retry
            continue

        placements.append((r, c, h, w))
        placed += 1

    if placed < n_objects:
        print(f"[object_spawner] Placed {placed}/{n_objects} objects "
              f"(connectivity or space constraints prevented more)")

    return live_grid, placements
