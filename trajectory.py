from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from track import Track


@dataclass
class Trajectory:
    """Dense geometric representation of a candidate racing line."""

    path: np.ndarray
    s: np.ndarray
    offsets_base: np.ndarray
    offsets_dense: np.ndarray
    valid: bool


def build_trajectory(track: Track, offsets: np.ndarray, path_resolution: float = 1.0) -> Trajectory:
    """Construct a dense trajectory from lateral offsets along the centerline."""
    if path_resolution <= 0:
        raise ValueError("path_resolution must be positive.")

    offsets = np.asarray(offsets, dtype=float)
    num_ctrl = len(offsets)
    offsets_clipped = np.clip(offsets, -track.width / 2.0, track.width / 2.0)

    base_s = np.linspace(0.0, track.length, num_ctrl, endpoint=False)
    offsets_periodic = np.concatenate([offsets_clipped, offsets_clipped[:1]])
    base_s_periodic = np.concatenate([base_s, [track.length]])
    offset_spline = CubicSpline(base_s_periodic, offsets_periodic, bc_type="periodic")

    dense_s = np.arange(0.0, track.length, path_resolution)
    if dense_s.size == 0 or dense_s[-1] < track.length:
        dense_s = np.append(dense_s, track.length)

    dense_offsets = np.clip(offset_spline(dense_s), -track.width / 2.0, track.width / 2.0)
    center_positions = track.position(dense_s)
    normals = track.normal(dense_s)
    path = center_positions + dense_offsets[:, None] * normals

    # Close the loop explicitly.
    if np.linalg.norm(path[0] - path[-1]) > 1e-6:
        path = np.vstack([path, path[0]])
        dense_s = np.append(dense_s, track.length)
        dense_offsets = np.append(dense_offsets, dense_offsets[0])

    s_path = np.zeros(len(path))
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s_path[1:] = np.cumsum(segment_lengths)

    center_at_path = track.position(np.mod(dense_s, track.length))
    normals_at_path = track.normal(np.mod(dense_s, track.length))
    lateral = np.einsum("ij,ij->i", path[: len(center_at_path)] - center_at_path, normals_at_path)
    valid = bool(np.all(np.abs(lateral) <= track.width / 2.0 + 1e-3))

    return Trajectory(
        path=path,
        s=s_path,
        offsets_base=offsets_clipped,
        offsets_dense=dense_offsets,
        valid=valid,
    )
