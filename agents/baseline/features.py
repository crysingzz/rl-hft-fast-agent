from __future__ import annotations

import numpy as np

from src.env.hft_env import N_LEVELS, OBS_DIM


def order_book_imbalance(bid_sizes: list, ask_sizes: list) -> float:
    bid_vol = sum(bid_sizes)
    ask_vol = sum(ask_sizes)
    total = bid_vol + ask_vol
    if total == 0.0:
        return 0.0
    return (bid_vol - ask_vol) / total


def weighted_mid_price(
    bid_prices: list,
    bid_sizes: list,
    ask_prices: list,
    ask_sizes: list,
) -> float:
    best_bid, best_bid_sz = bid_prices[0], bid_sizes[0]
    best_ask, best_ask_sz = ask_prices[0], ask_sizes[0]
    denom = best_bid_sz + best_ask_sz
    if denom == 0.0:
        return (best_bid + best_ask) * 0.5
    return (best_bid * best_ask_sz + best_ask * best_bid_sz) / denom


def compute_features(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    session_progress: float,
) -> np.ndarray:
    out = []
    for p in bid_prices:
        out.append(float(p))
    for s in bid_sizes:
        out.append(float(s))
    for p in ask_prices:
        out.append(float(p))
    for s in ask_sizes:
        out.append(float(s))

    out.append(float(ask_prices[0] - bid_prices[0]))
    out.append(order_book_imbalance(list(bid_sizes), list(ask_sizes)))
    out.append(weighted_mid_price(
        list(bid_prices), list(bid_sizes),
        list(ask_prices), list(ask_sizes),
    ))
    out.append(session_progress)

    return np.array(out, dtype=np.float32)
