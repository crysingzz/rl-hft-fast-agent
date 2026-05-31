import numpy as np
import numba as nb


@nb.jit(nopython=True, cache=True)
def order_book_imbalance(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
    bid_vol = 0.0
    ask_vol = 0.0
    for i in range(len(bid_sizes)):
        bid_vol += bid_sizes[i]
        ask_vol += ask_sizes[i]
    total = bid_vol + ask_vol
    if total == 0.0:
        return 0.0
    return (bid_vol - ask_vol) / total


@nb.jit(nopython=True, cache=True)
def weighted_mid_price(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
) -> float:
    best_bid = bid_prices[0]
    best_ask = ask_prices[0]
    best_bid_sz = bid_sizes[0]
    best_ask_sz = ask_sizes[0]
    denom = best_bid_sz + best_ask_sz
    if denom == 0.0:
        return (best_bid + best_ask) * 0.5
    return (best_bid * best_ask_sz + best_ask * best_bid_sz) / denom


@nb.jit(nopython=True, cache=True)
def _fill_buffer(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    session_progress: float,
    out: np.ndarray,
) -> None:
    n = len(bid_prices)
    idx = 0
    for i in range(n):
        out[idx] = bid_prices[i]; idx += 1
    for i in range(n):
        out[idx] = bid_sizes[i]; idx += 1
    for i in range(n):
        out[idx] = ask_prices[i]; idx += 1
    for i in range(n):
        out[idx] = ask_sizes[i]; idx += 1

    out[idx] = ask_prices[0] - bid_prices[0]; idx += 1
    out[idx] = order_book_imbalance(bid_sizes, ask_sizes); idx += 1
    out[idx] = weighted_mid_price(bid_prices, bid_sizes, ask_prices, ask_sizes); idx += 1
    out[idx] = session_progress


_N_LEVELS = 5
_OBS_DIM = _N_LEVELS * 4 + 4


def compute_features(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    session_progress: float,
) -> np.ndarray:
    out = np.empty(_OBS_DIM, dtype=np.float32)
    _fill_buffer(bid_prices, bid_sizes, ask_prices, ask_sizes, session_progress, out)
    return out
