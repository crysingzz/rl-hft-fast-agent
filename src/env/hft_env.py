from __future__ import annotations

import time
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

N_LEVELS = 5
LOB_FEATURE_DIM = N_LEVELS * 4 + 4
EXEC_FEATURE_DIM = 5
OBS_DIM = LOB_FEATURE_DIM + EXEC_FEATURE_DIM

ACTION_AGGRESSIVE_LIMIT = 0
ACTION_MARKET_ORDER     = 1
ACTION_CANCEL_REPOST    = 2
ACTION_WAIT             = 3
N_ACTIONS = 4

DEPTH_PENALTY_ALPHA   = 2.0
TERMINAL_PENALTY_BETA = 5.0
LATENCY_PENALTY_SCALE = 0.5
LATENCY_THRESHOLD_NS  = 1_000


class HFTEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        session_minutes: int = 30,
        parent_order_size: int = 200,
        child_order_size: int = 1,
        depth_penalty: float = DEPTH_PENALTY_ALPHA,
        terminal_penalty: float = TERMINAL_PENALTY_BETA,
        latency_penalty_scale: float = LATENCY_PENALTY_SCALE,
        latency_threshold_ns: int = LATENCY_THRESHOLD_NS,
        seed: int | None = None,
        feature_fn: Callable | None = None,
    ) -> None:
        super().__init__()

        self.session_minutes       = session_minutes
        self.parent_order_size     = parent_order_size
        self.child_order_size      = child_order_size
        self.depth_penalty         = depth_penalty
        self.terminal_penalty      = terminal_penalty
        self.latency_penalty_scale = latency_penalty_scale
        self.latency_threshold_ns  = latency_threshold_ns

        if feature_fn is not None:
            self._feature_fn = feature_fn
        else:
            from src.features.microstructure import compute_features as _fn
            self._feature_fn = _fn

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._obs_buf    = np.zeros(OBS_DIM,  dtype=np.float32)
        self._bid_prices = np.zeros(N_LEVELS, dtype=np.float64)
        self._bid_sizes  = np.zeros(N_LEVELS, dtype=np.float64)
        self._ask_prices = np.zeros(N_LEVELS, dtype=np.float64)
        self._ask_sizes  = np.zeros(N_LEVELS, dtype=np.float64)

        self._kernel: object | None = None
        self._step_count   = 0
        self._max_steps    = 0
        self._cash         = 0.0
        self._position     = 0
        self._total_filled = 0
        self._last_mid     = 0.0
        self._entry_price  = 0.0
        self._pending_order_id: int | None = None

        self._np_rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self._step_count   = 0
        self._cash         = 0.0
        self._position     = 0
        self._total_filled = 0
        self._pending_order_id = None

        self._kernel    = self._build_kernel()
        self._max_steps = self._kernel.get_max_steps()
        self._advance_kernel_to_next_event()

        book = self._kernel.get_order_book()
        self._unpack_book(book)
        self._entry_price = self._last_mid if self._last_mid > 0 else 100.0

        obs = self._get_observation()
        return obs, {}

    def step(self, action: int):
        t_start = time.perf_counter_ns()

        fills = self._execute_action(action)
        self._advance_kernel_to_next_event()

        inference_ns = int(time.perf_counter_ns() - t_start)

        self._step_count += 1
        terminated = (
            self._step_count >= self._max_steps
            or self._total_filled >= self.parent_order_size
        )

        obs    = self._get_observation()
        reward = self._compute_reward(fills, inference_ns, terminated)

        info = {
            "inference_ns": inference_ns,
            "position":     self._position,
            "cash":         self._cash,
            "total_filled": self._total_filled,
            "fill_ratio":   self._total_filled / max(self.parent_order_size, 1),
        }
        return obs, reward, terminated, False, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        if self._kernel is not None:
            self._kernel.close()
            self._kernel = None

    def _build_kernel(self):
        from src.env.synthetic_market import SyntheticMarket
        return SyntheticMarket(session_minutes=self.session_minutes, rng=self._np_rng)

    def _advance_kernel_to_next_event(self) -> None:
        if self._kernel is not None:
            self._kernel.step()

    def _get_observation(self) -> np.ndarray:
        if self._kernel is not None:
            book = self._kernel.get_order_book()
            self._unpack_book(book)

        session_progress = float(self._step_count / max(self._max_steps, 1))

        lob = self._feature_fn(
            self._bid_prices, self._bid_sizes,
            self._ask_prices, self._ask_sizes,
            session_progress,
        )
        self._obs_buf[:LOB_FEATURE_DIM] = lob

        holdings_pct = self._total_filled / max(self.parent_order_size, 1)
        time_pct     = session_progress
        diff_pct     = holdings_pct - time_pct
        price_impact = (self._last_mid - self._entry_price) / max(self._entry_price, 1e-8)
        pos_norm     = self._position / max(self.parent_order_size, 1)

        self._obs_buf[LOB_FEATURE_DIM + 0] = np.float32(holdings_pct)
        self._obs_buf[LOB_FEATURE_DIM + 1] = np.float32(time_pct)
        self._obs_buf[LOB_FEATURE_DIM + 2] = np.float32(diff_pct)
        self._obs_buf[LOB_FEATURE_DIM + 3] = np.float32(price_impact)
        self._obs_buf[LOB_FEATURE_DIM + 4] = np.float32(pos_norm)

        return self._obs_buf.copy()

    def _unpack_book(self, book: dict) -> None:
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        for i in range(N_LEVELS):
            if i < len(bids):
                self._bid_prices[i], self._bid_sizes[i] = bids[i]
            else:
                self._bid_prices[i] = self._bid_sizes[i] = 0.0
            if i < len(asks):
                self._ask_prices[i], self._ask_sizes[i] = asks[i]
            else:
                self._ask_prices[i] = self._ask_sizes[i] = 0.0
        if bids and asks:
            self._last_mid = (bids[0][0] + asks[0][0]) * 0.5

    def _execute_action(self, action: int) -> list[dict]:
        if self._kernel is None:
            return []

        remaining = self.parent_order_size - self._total_filled
        size = min(self.child_order_size, max(remaining, 0))
        if size <= 0:
            return []

        if action == ACTION_AGGRESSIVE_LIMIT:
            self._pending_order_id = self._kernel.place_limit_order(
                side="BUY", price=self._bid_prices[0], size=size
            )
        elif action == ACTION_MARKET_ORDER:
            self._kernel.place_market_order(side="BUY", size=size)
        elif action == ACTION_CANCEL_REPOST:
            if self._pending_order_id is not None:
                self._kernel.cancel_order(self._pending_order_id)
            self._pending_order_id = self._kernel.place_limit_order(
                side="BUY", price=self._bid_prices[0], size=size
            )

        fill = self._kernel.get_last_fill()
        fills = []
        if fill is not None:
            fills.append(fill)
            self._position     += fill["size"]
            self._total_filled += fill["size"]
            self._cash         -= fill["price"] * fill["size"]
        return fills

    def _compute_reward(self, fills: list[dict], inference_ns: int, terminated: bool) -> float:
        is_reward = 0.0
        depth_consumed = 0.0
        for fill in fills:
            is_reward      += fill["size"] * (self._entry_price - fill["price"])
            depth_consumed += fill["size"]

        norm = max(self.parent_order_size, 1)
        is_reward     = is_reward / norm
        depth_penalty = -self.depth_penalty * depth_consumed / norm

        latency_excess  = max(0, inference_ns - self.latency_threshold_ns)
        latency_penalty = -self.latency_penalty_scale * latency_excess

        terminal = 0.0
        if terminated:
            unexecuted = abs(self.parent_order_size - self._total_filled)
            terminal = -self.terminal_penalty * unexecuted / norm

        return float(is_reward + depth_penalty + latency_penalty + terminal)
