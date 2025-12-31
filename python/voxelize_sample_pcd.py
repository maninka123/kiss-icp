"""
Load a sample PCD from Datasets/Sample PCDs, extract XYZ points, and show a simple
before/after summary when running voxel_down_sample from the KISS-ICP bindings.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from kiss_icp.voxelization import voxel_down_sample
except ImportError as exc:
    raise SystemExit(
        "kiss_icp_pybind is not available. Build/install the Python extension (e.g. "
        "`pip install -e python` from the repo root) and retry."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VOXEL_SIZE = 0.2  # meters


def _find_sample_pcd() -> Path:
    pcds = sorted((REPO_ROOT / "Datasets").rglob("*.pcd"))
    if not pcds:
        raise FileNotFoundError("No .pcd files found under Datasets/")
    return pcds[0]


def _load_xyz_from_pcd(path: Path) -> np.ndarray:
    """
    Minimal binary PCD loader specialized for the Sample PCDs shipped here.
    It reads the header to find the POINTS count, then interprets each record
    as 9 float32 words (8 real fields + 4 unused bytes) and returns XYZ.
    """
    with path.open("rb") as f:
        points = None
        # Read header
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected end of file in header for {path}")
            if line.startswith(b"POINTS"):
                points = int(line.split()[1])
            if line.startswith(b"DATA"):
                if b"binary" not in line:
                    raise ValueError(f"Only binary PCD files are supported ({path})")
                break
        if points is None:
            raise ValueError(f"POINTS entry missing in PCD header for {path}")

        # Each record is 36 bytes: 8 float32 values + 4 unused bytes.
        raw = np.fromfile(f, dtype="<f4", count=points * 9)
        if raw.size != points * 9:
            raise ValueError(f"Unexpected data length in {path}, got {raw.size} floats")
        raw = raw.reshape(points, 9)
        return raw[:, 5:8].astype(np.float64)


def _summarize(label: str, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    print(f"{label}: {len(points):,} points")
    print(f"  range x[{mins[0]:.2f}, {maxs[0]:.2f}] "
          f"y[{mins[1]:.2f}, {maxs[1]:.2f}] "
          f"z[{mins[2]:.2f}, {maxs[2]:.2f}]")


def main(voxel_size: float = DEFAULT_VOXEL_SIZE) -> int:
    pcd_path = _find_sample_pcd()
    print(f"Using sample PCD: {pcd_path}")

    points = _load_xyz_from_pcd(pcd_path)
    _summarize("Before voxelization", points)

    down = voxel_down_sample(points, voxel_size)
    _summarize(f"After voxelization (voxel_size={voxel_size} m)", down)
    return 0


if __name__ == "__main__":
    size = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VOXEL_SIZE
    raise SystemExit(main(size))
