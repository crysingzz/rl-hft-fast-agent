from __future__ import annotations

from pathlib import Path

import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.env.hft_env import HFTEnv
from src.agent.model import DuelingDQNPolicy
from src.utils import get_device
from src.callbacks import LatencyCallback
from agents.baseline.features import compute_features as _baseline_features

OUTPUT_DIR = Path("results/baseline")
CONFIG_PATH = Path("configs/default.yaml")


def build_env(cfg: dict, seed: int) -> Monitor:
    env = HFTEnv(
        session_minutes=cfg["env"]["session_minutes"],
        parent_order_size=cfg["env"]["parent_order_size"],
        child_order_size=cfg["env"]["child_order_size"],
        depth_penalty=cfg["env"]["depth_penalty"],
        terminal_penalty=cfg["env"]["terminal_penalty"],
        latency_penalty_scale=cfg["env"]["latency_penalty_scale"],
        latency_threshold_ns=cfg["env"]["latency_threshold_ns"],
        seed=seed,
        feature_fn=_baseline_features,
    )
    return Monitor(env)


def train(cfg_path: Path = CONFIG_PATH, output_dir: Path = OUTPUT_DIR) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_env = build_env(cfg, cfg["train"]["seed"])
    eval_env  = build_env(cfg, cfg["train"]["seed"] + 1)

    model = DQN(
        policy=DuelingDQNPolicy,
        env=train_env,
        learning_rate=cfg["train"]["lr"],
        buffer_size=cfg["train"]["buffer_size"],
        learning_starts=cfg["train"]["learning_starts"],
        batch_size=cfg["train"]["batch_size"],
        tau=cfg["train"]["tau"],
        gamma=cfg["train"]["gamma"],
        train_freq=cfg["train"]["train_freq"],
        target_update_interval=cfg["train"]["target_update_interval"],
        exploration_fraction=cfg["train"]["exploration_fraction"],
        exploration_final_eps=cfg["train"]["exploration_final_eps"],
        policy_kwargs={"net_arch": cfg["model"]["hidden_layers"]},
        device=get_device(),
        verbose=1,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    model.learn(
        total_timesteps=cfg["train"]["total_timesteps"],
        callback=[
            EvalCallback(
                eval_env,
                best_model_save_path=str(output_dir / "best"),
                log_path=str(output_dir / "eval_logs"),
                eval_freq=cfg["train"]["eval_freq"],
                deterministic=True,
            ),
            LatencyCallback(measure_freq=1000, n_calls=200, verbose=1),
            CheckpointCallback(
                save_freq=cfg["train"]["checkpoint_freq"],
                save_path=str(output_dir / "checkpoints"),
            ),
        ],
        progress_bar=False,
    )
    model.save(str(output_dir / "model"))


if __name__ == "__main__":
    train()
