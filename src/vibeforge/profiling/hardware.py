"""Portable host inspection plus optional NVIDIA observations."""

import os
import math
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
    devices = visible_gpus()
    visible = {device["uuid"] for device in devices} if devices is not None else None
    result = query(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader,nounits"])
    if result is None:
        return None
    processes = []
    for row in result.splitlines():
        try:
            pid, memory, device = (value.strip() for value in row.split(","))
            if visible is not None and device not in visible:
                continue
            processes.append({"pid": int(pid.strip()), "memory_mib": memory.strip()})
        except ValueError:
            continue
    return processes


def gpu_telemetry():
    """Typed NVIDIA boundary snapshots, restricted to CUDA-visible devices."""
    devices = visible_gpus()
    if devices is None:
        return None
    if not devices:
        return []
    result = query(["nvidia-smi", "--query-gpu=index,uuid,temperature.gpu,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,power.limit",
                    "--format=csv,noheader,nounits"])
    if result is None:
        return None
    visible = {device["uuid"]: i for i, device in enumerate(devices)}
    rows = []
    try:
        for line in result.splitlines():
            index, uuid, *values = [value.strip() for value in line.split(",")]
            if uuid not in visible:
                continue
            if len(values) != 7:
                return None
            def number(value):
                try:
                    parsed = float(value)
                    return parsed if math.isfinite(parsed) else None
                except ValueError:
                    return None
            rows.append({"physical_index": int(index), "visible_index": visible[uuid], "uuid": uuid,
                         **dict(zip(("temperature_c", "gpu_utilization_pct", "memory_utilization_pct",
                                    "sm_clock_mhz", "memory_clock_mhz", "power_watts", "power_limit_watts"),
                                    map(number, values)))})
    except ValueError:
        return None
    return sorted(rows, key=lambda row: row["visible_index"])


def visible_gpus():
    """Observe memory on CUDA-visible devices, preserving visible index order."""
    selection = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selection is not None and selection.strip() in ("", "-1"):
        return []
    result = query(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                    "--format=csv,noheader,nounits"])
    if result is None:
        return None
    devices = []
    try:
        for row in result.splitlines():
            index, uuid, total, free = (value.strip() for value in row.split(","))
            devices.append({"index": int(index), "uuid": uuid,
                            "total_bytes": int(total) * 1024 ** 2,
                            "available_bytes": int(free) * 1024 ** 2})
    except ValueError:
        return None
    if selection is None:
        return devices
    selected = []
    for item in selection.split(","):
        item = item.strip()
        matches = [device for device in devices if
                   (item.isdecimal() and device["index"] == int(item)) or
                   (item.startswith("GPU-") and device["uuid"].startswith(item))]
        if len(matches) != 1 or matches[0] in selected:
            break
        selected.append(matches[0])
    return selected
