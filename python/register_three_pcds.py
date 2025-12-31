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
MAX_PLOT_POINTS = 40000

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


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply homogeneous transform T to Nx3 points."""
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


def _visualize(clouds: List[np.ndarray], labels: List[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"matplotlib not available ({exc}); skipping visualization.")
        return

    def _sample(points: np.ndarray) -> np.ndarray:
        if len(points) <= MAX_PLOT_POINTS:
            return points
        idx = np.random.choice(len(points), size=MAX_PLOT_POINTS, replace=False)
        return points[idx]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for pts, lbl, col in zip(clouds, labels, colors):
        pts_s = _sample(pts)
        ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], s=0.2, c=col, label=lbl)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main(voxel_size: float = VOXEL_SIZE, visualize: bool = False) -> int:
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

    if visualize:
        cloud1_map = _transform_points(clouds[0], pose1)
        cloud2_map = _transform_points(clouds[1], pose2)
        cloud3_map = _transform_points(clouds[2], pose3)
        _visualize(
            [cloud1_map, cloud2_map, cloud3_map],
            ["Cloud 1", "Cloud 2 aligned", "Cloud 3 aligned"],
        )
    return 0


if __name__ == "__main__":
    vox = VOXEL_SIZE
    show = False
    for arg in sys.argv[1:]:
        if arg in ("--show", "--plot", "-v"):
            show = True
        else:
            vox = float(arg)
    raise SystemExit(main(vox, visualize=show))
