"""
Backend selection for DAERWEN3.5.

Purpose:
- Centralize CPU / GPU (CuPy) choice.
- Provide thin helpers for array conversion and RNG so后续迁移内核时无需在业务逻辑处散落判断。

Design notes:
- 无逻辑改动：默认仍使用 NumPy；若环境可用且未强制 CPU，则启用 CuPy。
- 仅做基础抽象，不对现有算法做任何行为修改。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Backend:
    name: str              # "cpu" | "gpu"
    xp: Any                # numpy or cupy
    rng: Any               # xp.random-like generator
    gpu_available: bool

    def to_xp(self, array: Any, dtype: Optional[Any] = None):
        """Convert to backend array."""
        if self.name == "gpu":
            return self.xp.asarray(array, dtype=dtype)
        return np.asarray(array, dtype=dtype) if dtype is not None else np.asarray(array)

    def to_cpu(self, array: Any):
        """Ensure array on CPU (numpy)."""
        if self.name == "gpu" and hasattr(self.xp, "asnumpy") and isinstance(array, self.xp.ndarray):  # type: ignore[attr-defined]
            return self.xp.asnumpy(array)
        return array


def select_backend(prefer_gpu: bool = True) -> Backend:
    force_cpu_env = os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    if force_cpu_env:
        prefer_gpu = False

    if prefer_gpu:
        try:
            import cupy as cp  # type: ignore

            rng = cp.random.default_rng()
            return Backend(name="gpu", xp=cp, rng=rng, gpu_available=True)
        except Exception:
            pass

    rng = np.random.default_rng()
    return Backend(name="cpu", xp=np, rng=rng, gpu_available=False)

