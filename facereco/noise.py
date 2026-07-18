from __future__ import annotations

import hashlib

import numpy as np


def add_gaussian_noise_bgr(image: np.ndarray, level: int, key: str) -> np.ndarray:
    if level == 0:
        return image
    sigma = 255.0 * (level / 100.0)
    seed = int.from_bytes(hashlib.sha1(f"{key}:{level}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, size=image.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8)

