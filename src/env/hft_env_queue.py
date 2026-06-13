from __future__ import annotations

import numpy as np
from gymnasium import spaces

from src.env.hft_env import (
    HFTEnv,
    BASE_OBS_DIM,
    ACTION_JOIN,
    ACTION_BEAT,
    ACTION_SNIPE,
    ACTION_HOLD,
    ACTION_CANCEL,
    N_ACTIONS,
)

# Re-exported for callers that imported these from the queue module.
N_ACTIONS_QUEUE = N_ACTIONS

QUEUE_FEATURE_DIM = 3
OBS_DIM_QUEUE = BASE_OBS_DIM + QUEUE_FEATURE_DIM


class HFTEnvQueue(HFTEnv):
    """Queue-aware environment: adds the agent's queue state to the observation.

    Shares the action set, dynamics and reward of HFTEnv, and appends three
    features describing where the order sits in the book: best-bid depth, the
    relative queue position of our order, and the spread in ticks.
    """

    def _obs_dim(self) -> int:
        return OBS_DIM_QUEUE

    def _get_observation(self) -> np.ndarray:
        self._write_base_obs()

        base = BASE_OBS_DIM
        if hasattr(self._kernel, "get_queue_state"):
            qs = self._kernel.get_queue_state()
            self._obs_buf[base + 0] = np.float32(qs["queue_depth_norm"])
            self._obs_buf[base + 1] = np.float32(qs["our_queue_position"])
            self._obs_buf[base + 2] = np.float32(qs.get("spread_ticks", 1.0) / 10.0)
        else:
            self._obs_buf[base: base + QUEUE_FEATURE_DIM] = 0.0

        return self._obs_buf.copy()
