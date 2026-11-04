"""Portable host inspection plus optional NVIDIA observations."""

import os
import platform
import subprocess
from pathlib import Path


def query(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def memory_info():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.strip().split()[0]) * 1024
    return {"total_bytes": values["MemTotal"], "available_bytes": values["MemAvailable"],
            "swap_bytes": values.get("SwapTotal", 0)}


def inspect_hardware():
    board = Path("/proc/device-tree/model")
    gpu = query(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"])
    return {
        "architecture": platform.machine(), "cpu_count": os.cpu_count(),
        "total_memory_bytes": memory_info()["total_bytes"],
        "board": board.read_bytes().decode(errors="replace").strip("\0") if board.exists() else platform.node(),
        "gpu": gpu, "kernel": platform.release(),
        "unified_memory": bool(gpu and ("Thor" in gpu or "Orin" in gpu)),
    }


def gpu_processes():
    result = query(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
    if result is None:
        return None
    processes = []
    for row in result.splitlines():
        try:
            pid, memory = row.split(",", 1)
            processes.append({"pid": int(pid.strip()), "memory_mib": memory.strip()})
        except ValueError:
            continue
    return processes
