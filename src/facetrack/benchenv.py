"""Hardware / runtime capture and latency statistics for benchmark records."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import psutil


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=15,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or platform.machine()


def environment() -> dict:
    pkgs = {}
    for name in ("opencv-python", "onnxruntime", "mediapipe", "numpy", "scipy", "pycocotools"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = None
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    return {
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": sys.version.split()[0],
        "packages": pkgs,
        "accelerator": "none (CPU execution providers only)",
        "power_plugged": None if battery is None else battery.power_plugged,
    }


def latency_stats(ms: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(ms, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "n": int(a.size),
    }


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
