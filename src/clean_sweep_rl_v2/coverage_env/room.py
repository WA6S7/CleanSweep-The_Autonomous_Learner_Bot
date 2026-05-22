import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List

CELL_SIZE_M = 0.20
_SDF_DEFAULT = Path(__file__).parent.parent / "maps" / "env1_bedroom.sdf"
_EPS = 1e-6


def load_room_grid(sdf_path: Path = _SDF_DEFAULT,
                   cell_size: float = CELL_SIZE_M
                   ) -> Tuple[np.ndarray, Dict]:
    # Parsing the SDF into an occupancy grid
    tree   = ET.parse(sdf_path)
    root   = tree.getroot()
    world  = root.find("world") or root
    boxes  = []

    # Collecting every static box as (centre_x, centre_y, size_x, size_y)
    for model in world.findall("model"):
        static_tag = model.find("static")
        if static_tag is None or static_tag.text.strip().lower() != "true":
            continue  # only static models are walls
        pose_tag = model.find("pose")
        if pose_tag is None:
            continue
        px, py = [float(v) for v in pose_tag.text.split()][:2]
        for box_tag in model.iter("box"):
            size_tag = box_tag.find("size")
            if size_tag is None:
                continue
            sx, sy, _ = [float(v) for v in size_tag.text.split()]
            boxes.append((px, py, sx, sy))

    if not boxes:
        raise ValueError(f"No static boxes found in {sdf_path}")

    
     # dropping duplicate boxes
    boxes = list(set(boxes)) 

    all_x = [px-sx/2 for px,py,sx,sy in boxes] + [px+sx/2 for px,py,sx,sy in boxes]
    all_y = [py-sy/2 for px,py,sx,sy in boxes] + [py+sy/2 for px,py,sx,sy in boxes]
    world_xmin, world_xmax = min(all_x), max(all_x)
    world_ymin, world_ymax = min(all_y), max(all_y)

    cols = int(np.ceil((world_xmax - world_xmin - _EPS) / cell_size))
    rows = int(np.ceil((world_ymax - world_ymin - _EPS) / cell_size))

    wall_grid = np.zeros((rows, cols), dtype=np.int8)

    eps = cell_size * 0.01
    for px, py, sx, sy in boxes:
        xs = np.arange(px-sx/2+eps, px+sx/2, cell_size)
        ys = np.arange(py-sy/2+eps, py+sy/2, cell_size)
        if len(xs) == 0: xs = np.array([px])  
        if len(ys) == 0: ys = np.array([py])
        for wx in xs:
            for wy in ys:
                col = int((wx - world_xmin) / cell_size)
                row = int((world_ymax - wy)  / cell_size)
                if 0 <= row < rows and 0 <= col < cols:
                    wall_grid[row, col] = 1

    # Keeping only the largest connected free region
    try:
        from scipy.ndimage import label as _ndlabel
        lab, n_comp = _ndlabel(wall_grid == 0)
        if n_comp > 1:
            sizes = [(lab == i).sum() for i in range(1, n_comp + 1)]
            keep_label = int(np.argmax(sizes)) + 1
            removed = int((wall_grid == 0).sum() - sizes[keep_label - 1])
            wall_grid[(lab != keep_label) & (lab != 0)] = 1
            print(f"[room] Pruned {n_comp - 1} disconnected free region(s) "
                  f"({removed} cells) outside the main interior")
    except ImportError:
        pass

    # Grid metadata for converting between world coords and cells
    meta = dict(
        origin_x   = world_xmin,
        origin_y   = world_ymax,
        cell_size  = cell_size,
        rows       = rows,
        cols       = cols,
        world_xmin = world_xmin,
        world_xmax = world_xmin + cols * cell_size,
        world_ymin = world_ymax - rows * cell_size,
        world_ymax = world_ymax,
    )

    n_free = int((wall_grid == 0).sum())
    print(f"[room] Loaded '{sdf_path.name}'  "
          f"grid={rows}×{cols}  interior_free={n_free}  walls={rows*cols-n_free}")
    return wall_grid, meta


# Continuous world point 
def world_to_cell(wx: float, wy: float, meta: Dict) -> Tuple[int, int]:
    col = int((wx - meta["origin_x"]) / meta["cell_size"])
    row = int((meta["origin_y"] - wy)  / meta["cell_size"])
    return row, col


# Grid cell into world point at the cell's centre
def cell_to_world(row: int, col: int, meta: Dict) -> Tuple[float, float]:
    wx = meta["origin_x"] + (col + 0.5) * meta["cell_size"]
    wy = meta["origin_y"] - (row + 0.5) * meta["cell_size"]
    return wx, wy
