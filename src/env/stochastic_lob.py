from __future__ import annotations

import numpy as np

N_LEVELS = 5
TICK_SIZE = 0.01
INITIAL_MID = 100.0
SESSION_STEPS = 3_000

# Queue-reactive intensities (Huang, Lehalle & Rosenbaum, 2015). The defining
# feature of the model is that order-flow intensities are functions of the
# current queue size, not constants:
#   * limit arrivals  lambda(q) = LAMBDA_0 * exp(-q / Q_REF), decreasing in q, so
#     a thin queue is replenished and a deep one saturates, giving a stable
#     (mean-reverting) queue instead of one that drifts without bound;
#   * cancellations   each resting share leaves at rate THETA, so the total cancel
#     rate is proportional to the queue size;
#   * market orders   arrive at constant rate MU and consume the best queue.
# The reference price follows an order-flow-imbalance random walk and the book is
# carried around it with a small resting spread, which keeps the spread finite
# (the reference-price framework of the queue-reactive model).
LAMBDA_0       = 8.0
Q_REF          = 20.0
MU_MARKET      = 2.0
THETA_CANCEL   = 0.04
INIT_QUEUE     = 20.0
SPREAD_TICKS   = 2      # resting spread; a BEAT order can improve it to one tick
PRICE_MOVE_P   = 0.05   # probability the reference price ticks on a given step
LEVEL_DECAY    = np.array([1.0, 0.7, 0.5, 0.35, 0.25])


class StochasticLOB:
    """Queue-reactive limit order book (Huang, Lehalle & Rosenbaum, 2015).

    Each price level holds a FIFO queue. Limit arrivals thin out as a queue
    fills, cancellations are proportional to resting volume, and market orders
    consume the best opposite queue. The reference price random-walks with order-
    flow imbalance and the book follows it at a small spread, so price formation
    is endogenous and the spread stays finite. The agent's own buy order is
    tracked by absolute price and queue position, and fills once the volume ahead
    of it is consumed while it sits at or above the best bid.
    """

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

        self._mid = INITIAL_MID
        self._bid_sizes = np.full(N_LEVELS, INIT_QUEUE, dtype=np.float64)
        self._ask_sizes = np.full(N_LEVELS, INIT_QUEUE, dtype=np.float64)
        self._best_bid = round(self._mid - SPREAD_TICKS * 0.5 * tick_size, 2)
        self._best_ask = round(self._mid + SPREAD_TICKS * 0.5 * tick_size, 2)

        self._last_fill: dict | None = None
        self._order_counter = 0

        self._has_pending  = False
        self._our_improved = False   # our order improved the touch (a BEAT)
        self._our_price: float = 0.0
        self._our_size: float = 0.0
        self._our_queue_ahead: float = 0.0
        self._order_hold_steps: int = 0

    def get_max_steps(self) -> int:
        return self._max_steps

    @staticmethod
    def _arrival_rate(queue: np.ndarray) -> np.ndarray:
        return LAMBDA_0 * np.exp(-queue / Q_REF)

    # ------------------------------------------------------------------ #
    # Market dynamics
    # ------------------------------------------------------------------ #
    def step(self) -> None:
        self._step += 1
        self._last_fill = None

        self._apply_limit_arrivals()
        self._apply_cancellations()
        self._apply_market_orders()
        self._update_price()

        if self._has_pending:
            self._order_hold_steps += 1

    def _apply_limit_arrivals(self) -> None:
        rate_bid = self._arrival_rate(self._bid_sizes) * LEVEL_DECAY[:N_LEVELS]
        rate_ask = self._arrival_rate(self._ask_sizes) * LEVEL_DECAY[:N_LEVELS]
        self._bid_sizes += self._rng.poisson(rate_bid).astype(np.float64)
        self._ask_sizes += self._rng.poisson(rate_ask).astype(np.float64)

    def _apply_cancellations(self) -> None:
        self._bid_sizes -= self._rng.binomial(self._bid_sizes.astype(np.int64), THETA_CANCEL)
        self._ask_sizes -= self._rng.binomial(self._ask_sizes.astype(np.int64), THETA_CANCEL)
        self._bid_sizes = np.maximum(0.0, self._bid_sizes)
        self._ask_sizes = np.maximum(0.0, self._ask_sizes)
        if self._has_pending and self._our_queue_ahead > 0:
            ahead = self._rng.binomial(int(self._our_queue_ahead), THETA_CANCEL)
            self._our_queue_ahead = max(0.0, self._our_queue_ahead - ahead)

    def _apply_market_orders(self) -> None:
        self._consume_bid(float(self._rng.poisson(MU_MARKET)))
        self._ask_sizes[0] = max(0.0, self._ask_sizes[0] - float(self._rng.poisson(MU_MARKET)))

    def _consume_bid(self, volume: float) -> None:
        # A sell market order walks the bid queue from the front. Our order is in
        # line to fill while it sits at or above the best bid (the touch).
        if volume <= 0:
            return
        at_touch = self._has_pending and self._our_price >= self._best_bid - 1e-9
        if at_touch:
            eat = min(self._our_queue_ahead, volume)
            self._our_queue_ahead -= eat
            volume -= eat
            self._bid_sizes[0] = max(0.0, self._bid_sizes[0] - eat)
            if self._our_queue_ahead <= 0 and volume > 0 and self._our_size > 0:
                fill_qty = min(self._our_size, volume)
                self._register_fill(self._our_price, fill_qty)
                volume -= fill_qty
                self._bid_sizes[0] = max(0.0, self._bid_sizes[0] - fill_qty)
        self._bid_sizes[0] = max(0.0, self._bid_sizes[0] - volume)

    def _register_fill(self, price: float, qty: float) -> None:
        self._last_fill = {"price": price, "size": int(qty), "side": "BUY"}
        self._our_size -= qty
        if self._our_size <= 0:
            self._has_pending = False
            self._our_improved = False
            self._our_queue_ahead = 0.0
            self._order_hold_steps = 0

    def _update_price(self) -> None:
        # Order-flow imbalance nudges the reference price; the whole book follows,
        # so the spread is preserved and the mid does a (biased) random walk.
        if self._rng.random() >= PRICE_MOVE_P:
            return
        bid, ask = self._bid_sizes[0], self._ask_sizes[0]
        p_up = 0.5 + 0.4 * (bid - ask) / (bid + ask + 1e-9)
        direction = 1 if self._rng.random() < p_up else -1

        self._best_bid = round(self._best_bid + direction * self.tick_size, 2)
        self._best_ask = round(self._best_ask + direction * self.tick_size, 2)
        self._mid = round(0.5 * (self._best_bid + self._best_ask), 2)

        if direction < 0:  # price down: bid rolls toward best, a near ask appears
            self._bid_sizes[:-1] = self._bid_sizes[1:]
            self._bid_sizes[-1] = 0.0
            self._ask_sizes[1:] = self._ask_sizes[:-1]
            self._ask_sizes[0] = float(self._rng.integers(3, int(INIT_QUEUE)))
        else:              # price up
            self._ask_sizes[:-1] = self._ask_sizes[1:]
            self._ask_sizes[-1] = 0.0
            self._bid_sizes[1:] = self._bid_sizes[:-1]
            self._bid_sizes[0] = float(self._rng.integers(3, int(INIT_QUEUE)))

    # ------------------------------------------------------------------ #
    # Order book view
    # ------------------------------------------------------------------ #
    def get_order_book(self) -> dict:
        bids, asks = [], []
        for i in range(N_LEVELS):
            bids.append((round(self._best_bid - i * self.tick_size, 2), float(self._bid_sizes[i])))
            asks.append((round(self._best_ask + i * self.tick_size, 2), float(self._ask_sizes[i])))
        return {"bids": bids, "asks": asks}

    # ------------------------------------------------------------------ #
    # Order interface
    # ------------------------------------------------------------------ #
    def _remove_our_order(self) -> None:
        if self._has_pending:
            if self._our_price >= self._best_bid - 1e-9:
                self._bid_sizes[0] = max(0.0, self._bid_sizes[0] - self._our_size)
            if self._our_improved:   # our order had improved the touch; restore it
                self._best_bid = round(self._best_bid - self.tick_size, 2)
                self._mid = round(0.5 * (self._best_bid + self._best_ask), 2)
        self._has_pending = False
        self._our_improved = False
        self._our_queue_ahead = 0.0
        self._our_size = 0.0
        self._order_hold_steps = 0

    def place_limit_order(self, side: str, price: float, size: int,
                          order_type: str = "JOIN") -> int:
        self._order_counter += 1
        self._remove_our_order()
        self._has_pending = True
        self._our_size = float(size)
        self._order_hold_steps = 0

        can_improve = (self._best_ask - self._best_bid) > self.tick_size + 1e-9
        if order_type == "BEAT" and can_improve:
            self._best_bid = round(self._best_bid + self.tick_size, 2)
            self._mid = round(0.5 * (self._best_bid + self._best_ask), 2)
            self._our_improved = True
            self._our_price = self._best_bid
            self._our_queue_ahead = 0.0
            self._bid_sizes[1:] = self._bid_sizes[:-1]
            self._bid_sizes[0] = self._our_size
        else:  # JOIN, or BEAT with no room to improve
            self._our_price = self._best_bid
            self._our_queue_ahead = float(self._bid_sizes[0])
            self._bid_sizes[0] += self._our_size
        return self._order_counter

    def place_market_order(self, side: str, size: int,
                           order_type: str = "SNIPE") -> int:
        self._order_counter += 1
        remaining, cost, filled = float(size), 0.0, 0.0
        for _ in range(N_LEVELS):
            take = min(self._ask_sizes[0], remaining)
            cost += take * self._best_ask
            filled += take
            remaining -= take
            self._ask_sizes[0] -= take
            if remaining <= 0:
                break
            self._best_ask = round(self._best_ask + self.tick_size, 2)
            self._ask_sizes[:-1] = self._ask_sizes[1:]
            self._ask_sizes[-1] = 0.0
        if filled > 0:
            self._mid = round(0.5 * (self._best_bid + self._best_ask), 2)
            self._register_market_fill(round(cost / filled, 4), filled)
        self._remove_our_order()
        return self._order_counter

    def _register_market_fill(self, price: float, qty: float) -> None:
        self._last_fill = {"price": price, "size": int(qty), "side": "BUY"}

    def cancel_order(self, order_id: int) -> None:
        self._remove_our_order()

    def get_last_fill(self) -> dict | None:
        return self._last_fill

    def get_queue_state(self) -> dict:
        spread_ticks = max(1, round((self._best_ask - self._best_bid) / self.tick_size))
        best_bid_depth = max(self._bid_sizes[0], 1.0)
        rqp = (self._our_queue_ahead / best_bid_depth) if self._has_pending else 0.0
        return {
            "queue_depth_norm":   min(1.0, self._bid_sizes[0] / 100.0),
            "our_queue_position": float(min(1.0, rqp)),
            "hold_steps_norm":    min(1.0, self._order_hold_steps / 50.0),
            "spread_ticks":       min(10.0, float(spread_ticks)),
        }

    def close(self) -> None:
        pass
