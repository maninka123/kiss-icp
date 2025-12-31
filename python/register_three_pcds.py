"""
Register three sample PCDs using the KISS-ICP registration and voxel map bindings.
Loads the first three .pcd files under Datasets/, seeds a voxel map with the first cloud,
aligns the second and third clouds to the map, and prints the resulting poses.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

from kiss_icp.mapping import VoxelHashMap
from kiss_icp.registration import Registration

# Defaults
VOXEL_SIZE = 0.2  # meters
MAX_DISTANCE = 100.0  # meters, map pruning distance
MAX_POINTS_PER_VOXEL = 20
MAX_ITERS = 20
CONV_EPS = 1e-3

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_first_three_pcds() -> List[Path]:
    pcds = sorted((REPO_ROOT / "Datasets").rglob("*.pcd"))
    if len(pcds) < 3:
        raise FileNotFoundError("Need at least three .pcd files under Datasets/")
    return pcds[:3]


def load_xyz_from_pcd(path: Path) -> np.ndarray:
    """
    Minimal binary PCD loader for the provided samples: reads header for POINTS,
    assumes 9 float32 words per point and returns XYZ as float64.
    """
    with path.open("rb") as f:
        points = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading header in {path}")
            if line.startswith(b"POINTS"):
                points = int(line.split()[1])
            if line.startswith(b"DATA"):
                if b"binary" not in line:
                    raise ValueError(f"Only binary PCD supported ({path})")
                break
        if points is None:
            raise ValueError(f"POINTS entry missing in {path}")
        raw = np.fromfile(f, dtype="<f4", count=points * 9)
        if raw.size != points * 9:
            raise ValueError(f"Unexpected data length in {path}: {raw.size} floats")
        raw = raw.reshape(points, 9)
        return raw[:, 5:8].astype(np.float64)


def pretty_pose(label: str, T: np.ndarray) -> None:
    print(f"{label} pose:")
    with np.printoptions(precision=4, suppress=True):
        print(T)


def main(voxel_size: float = VOXEL_SIZE) -> int:
    pcd_paths = find_first_three_pcds()
    clouds = [load_xyz_from_pcd(p) for p in pcd_paths]
    for idx, (path, cloud) in enumerate(zip(pcd_paths, clouds), start=1):
        print(f"PCD{idx}: {path} -> {len(cloud):,} points")

    voxel_map = VoxelHashMap(
        voxel_size=voxel_size,
        max_distance=MAX_DISTANCE,
        max_points_per_voxel=MAX_POINTS_PER_VOXEL,
    )
    reg = Registration(
        max_num_iterations=MAX_ITERS,
        convergence_criterion=CONV_EPS,
        max_num_threads=0,
    )

    # Seed map with first cloud at identity
    pose1 = np.eye(4)
    voxel_map.update(clouds[0], pose1)

    max_corr = 3 * voxel_size
    kernel = voxel_size

    pose2 = reg.align_points_to_map(
        points=clouds[1],
        voxel_map=voxel_map,
        initial_guess=pose1,
        max_correspondance_distance=max_corr,
        kernel=kernel,
    )
    voxel_map.update(clouds[1], pose2)

    pose3 = reg.align_points_to_map(
        points=clouds[2],
        voxel_map=voxel_map,
        initial_guess=pose2,
        max_correspondance_distance=max_corr,
        kernel=kernel,
    )
    voxel_map.update(clouds[2], pose3)

    pretty_pose("Cloud 1 (seed)", pose1)
    pretty_pose("Cloud 2 -> map", pose2)
    pretty_pose("Cloud 3 -> map", pose3)
    return 0


if __name__ == "__main__":
    vox = float(sys.argv[1]) if len(sys.argv) > 1 else VOXEL_SIZE
    raise SystemExit(main(vox))
