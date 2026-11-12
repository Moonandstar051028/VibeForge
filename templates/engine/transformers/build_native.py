"""Frozen optional build recipe for candidate-owned C++/CUDA extensions."""

from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    sources = sorted(str(path) for path in (root / "kernels").glob("*")
                     if path.suffix in {".cu", ".cpp", ".c"})
    if not sources:
        return
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=True)
    from torch.utils.cpp_extension import load
    load(
        name="vibeforge_candidate_kernels",
        sources=sources,
        build_directory=str(runtime),
        verbose=True,
        is_python_module=True,
    )


if __name__ == "__main__":
    main()
