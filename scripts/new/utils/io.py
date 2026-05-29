"""Utility functions for I/O, logging, and config loading."""

import os
import logging
import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def sorted_frame_paths(frames_dir: str, exts=(".png", ".jpg")) -> list:
    """Tìm tất cả ảnh trong frames_dir, hỗ trợ cả flat và nested (rglob)."""
    root = Path(frames_dir)
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)
