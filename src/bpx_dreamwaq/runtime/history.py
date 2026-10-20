"""Complete observation history in Isaac Lab's flattened, term-major order."""

import numpy as np


class ObservationHistory:
    """Append once per policy tick; a new instance starts a new episode.

    The first frame fills the window. Each term is ordered oldest to newest,
    matching ObservationGroupCfg(history_length=H, flatten_history_dim=True).
    """

    def __init__(self, length: int, joint_count: int = 12, include_base_lin_vel: bool = True):
        if length < 1:
            raise ValueError("history length must be positive")
        self.length = length
        self.widths = (3, 3, 3, joint_count, joint_count, joint_count)
        if include_base_lin_vel:
            self.widths = (3,) + self.widths
        self.frames = None

    def append(self, observation: np.ndarray) -> np.ndarray:
        frame = np.asarray(observation, dtype=np.float32)
        if frame.shape != (sum(self.widths),) or not np.isfinite(frame).all():
            raise ValueError("history requires one finite complete observation frame")
        if self.frames is None:
            self.frames = np.repeat(frame[None, :], self.length, axis=0)
        else:
            self.frames[:-1] = self.frames[1:].copy()
            self.frames[-1] = frame
        return np.concatenate([
            term.reshape(-1) for term in
            np.split(self.frames, np.cumsum(self.widths)[:-1], axis=1)
        ])
