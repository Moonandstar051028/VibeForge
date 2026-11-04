"""Linux Landlock write confinement for generated backend processes."""

import ctypes
import os
import platform
import stat
from pathlib import Path


def prepare_device(device):
    """Initialize trusted CUDA libraries before confining generated code.

    Thor's driver initializes system IPC resources outside the engine cache.
    Keep those writes in this trusted setup phase instead of permitting the
    generated backend to mutate shared IPC files.
    """
    if not str(device).startswith('cuda'):
        return None
    import torch
    torch.cuda.init()
    torch.cuda.set_device(device if ':' in device else 0)
    return {'device': torch.cuda.get_device_name(), 'trusted_cuda_initialized': True}


def restrict_writes(directories):
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "x86_64"):
        raise RuntimeError("Generated backends require Linux Landlock on aarch64 or x86_64")
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError(f"Landlock is unavailable: errno {ctypes.get_errno()}")
    # Reads and execution remain available. Character devices need explicit
    # WRITE_FILE permission for Python's /dev/null probes and CUDA driver I/O.
    rights = sum(1 << bit for bit in (1, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    if abi >= 2:
        rights |= 1 << 13
    if abi >= 3:
        rights |= 1 << 14

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    ruleset = Ruleset(rights)
    descriptor = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for directory in directories:
            path = Path(directory).resolve()
            path.mkdir(parents=True, exist_ok=True)
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(rights, parent)
                if libc.syscall(445, descriptor, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), "landlock_add_rule")
            finally:
                os.close(parent)
        device_paths = {Path('/dev/null'), Path('/dev/zero'), Path('/dev/random'), Path('/dev/urandom')}
        for pattern in ('nvidia*', 'nvidia-caps/*', 'nvhost*', 'nvmap', 'dri/renderD*'):
            device_paths.update(Path('/dev').glob(pattern))
        device_rights = (1 << 1) | ((1 << 14) if abi >= 3 else 0)
        for path in sorted(device_paths):
            if not path.exists() or not stat.S_ISCHR(path.stat().st_mode):
                continue
            parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = PathRule(device_rights, parent)
                if libc.syscall(445, descriptor, 1, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), 'landlock_add_device_rule')
            finally:
                os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) < 0:
            raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS")
        if libc.syscall(446, descriptor, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(descriptor)
    return {"landlock_abi": abi, "write_directories": [str(Path(d).resolve()) for d in directories]}
