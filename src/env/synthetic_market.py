from __future__ import annotations

import numpy as np

N_LEVELS = 5
TICK_SIZE = 0.01
INITIAL_MID = 100.0
SESSION_STEPS = 3_000

FILL_PROB_JOIN  = 0.30
FILL_PROB_BEAT  = 0.60
FILL_PROB_SNIPE = 0.95


class SyntheticMarket:
    def __init__(
        self,
        session_minutes: int = 30,
        tick_size: float = TICK_SIZE,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.tick_size  = tick_size
        self._rng       = rng if rng is not None else np.random.default_rng()
        self._max_steps = SESSION_STEPS
        self._step      = 0
        self._mid       = INITIAL_MID
        self._last_fill: dict | None = None
        self._order_counter = 0

        self._last_bid = INITIAL_MID - tick_size
        self._last_ask = INITIAL_MID + tick_size

        self._queue_depth_bid: float = float(self._rng.integers(10, 50))
        self._our_queue_position: float = 0.0
        self._order_hold_steps: int = 0
        self._has_pending_limit: bool = False

    def get_max_steps(self) -> int:
        return self._max_steps

    def step(self) -> None:
        noise = self._rng.normal(0, self.tick_size * 0.5)
        self._mid = max(self.tick_size, self._mid + noise)
        self._step += 1
        self._last_fill = None

        if self._has_pending_limit:
            self._order_hold_steps += 1
            self._our_queue_position = max(0.0, self._our_queue_position - 0.03)

        self._queue_depth_bid = max(1.0, self._queue_depth_bid + self._rng.normal(0, 3.0))

    def get_order_book(self) -> dict:
        half_spread = self.tick_size * self._rng.integers(1, 4)
        bids, asks = [], []
        for i in range(N_LEVELS):
            bid_p = round(self._mid - half_spread - i * self.tick_size, 2)
            ask_p = round(self._mid + half_spread + i * self.tick_size, 2)
            bid_sz = float(self._rng.integers(1, 50))
            ask_sz = float(self._rng.integers(1, 50))
            bids.append((bid_p, bid_sz))
            asks.append((ask_p, ask_sz))

        self._last_bid = bids[0][0]
        self._last_ask = asks[0][0]
        return {"bids": bids, "asks": asks}

    def place_limit_order(
        self,
        side: str,
        price: float,
        size: int,
        order_type: str = "JOIN",
    ) -> int:
        self._order_counter += 1
        self._has_pending_limit = True

        if order_type == "BEAT":
            self._our_queue_position = 0.1
            self._order_hold_steps   = 0
            fill_prob = FILL_PROB_BEAT
        else:
            self._our_queue_position = 1.0
            self._order_hold_steps   = 0
            fill_prob = FILL_PROB_JOIN

        if self._rng.random() < fill_prob:
            self._last_fill = {"price": price, "size": size, "side": side}
            self._has_pending_limit  = False
            self._our_queue_position = 0.0
            self._order_hold_steps   = 0

        return self._order_counter

    def place_market_order(
        self,
        side: str,
        size: int,
        order_type: str = "SNIPE",
    ) -> int:
        self._order_counter += 1
        if self._rng.random() < FILL_PROB_SNIPE:
            fill_price = self._last_ask if side == "BUY" else self._last_bid
            self._last_fill = {"price": fill_price, "size": size, "side": side}
        self._has_pending_limit  = False
        self._our_queue_position = 0.0
        self._order_hold_steps   = 0
        return self._order_counter

    def cancel_order(self, order_id: int) -> None:
        self._has_pending_limit  = False
        self._our_queue_position = 0.0
        self._order_hold_steps   = 0

    def get_last_fill(self) -> dict | None:
        return self._last_fill

    def get_queue_state(self) -> dict:
        spread_ticks = max(1, round((self._last_ask - self._last_bid) / self.tick_size))
        return {
            "queue_depth_norm":   min(1.0, self._queue_depth_bid / 100.0),
            "our_queue_position": float(self._our_queue_position),
            "hold_steps_norm":    min(1.0, self._order_hold_steps / 50.0),
            "spread_ticks":       min(10.0, float(spread_ticks)),
        }

    def close(self) -> None:
        pass
