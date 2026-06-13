from __future__ import annotations

import numpy as np

from src.env.hft_env import ACTION_JOIN, ACTION_BEAT, ACTION_SNIPE, ACTION_CANCEL
from src.env.hft_env_queue import HFTEnvQueue, OBS_DIM_QUEUE

LATENCY_FEATURE_DIM = 3
OBS_DIM_LATENCY     = OBS_DIM_QUEUE + LATENCY_FEATURE_DIM

MAX_HOLD_STEPS = 50


class HFTEnvLatency(HFTEnvQueue):
    """Latency-aware environment: adds order hold time and competitiveness.

    Extends the queue observation with three features that describe the timing
    of the resting order: how long it has been held, its relative queue
    position, and a competitiveness score combining price advantage with how
    far the order is from the front of a thin queue.
    """

    def __init__(self, **kwargs) -> None:
        self._order_submit_step: int | None = None
        super().__init__(**kwargs)

    def _obs_dim(self) -> int:
        return OBS_DIM_LATENCY

    def reset(self, *, seed=None, options=None):
        self._order_submit_step = None
        return super().reset(seed=seed, options=options)

    def step(self, action: int):
        if action in (ACTION_JOIN, ACTION_BEAT, ACTION_SNIPE):
            self._order_submit_step = self._step_count
        elif action == ACTION_CANCEL:
            self._order_submit_step = None
        return super().step(action)

    def _get_observation(self) -> np.ndarray:
        super()._get_observation()  # fills base + queue features in self._obs_buf

        if self._order_submit_step is not None:
            hold      = self._step_count - self._order_submit_step
            hold_norm = min(1.0, hold / MAX_HOLD_STEPS)
        else:
            hold_norm = 0.0

        if hasattr(self._kernel, "get_queue_state"):
            qs          = self._kernel.get_queue_state()
            rqp         = float(qs["our_queue_position"])
            queue_depth = float(qs["queue_depth_norm"])
        else:
            rqp         = 0.5
            queue_depth = 0.5

        price_advantage = 1.0 if self._last_action == ACTION_BEAT else 0.5
        competitiveness = float(
            price_advantage * (1.0 - rqp) * (1.0 - 0.5 * queue_depth)
        )

        self._obs_buf[OBS_DIM_QUEUE + 0] = np.float32(hold_norm)
        self._obs_buf[OBS_DIM_QUEUE + 1] = np.float32(rqp)
        self._obs_buf[OBS_DIM_QUEUE + 2] = np.float32(competitiveness)

        return self._obs_buf.copy()
