#!/usr/bin/env python3
"""
Sets up LD_PRELOAD (Linux) or PATH (Windows) so any HIP program routes
through vkflame shims.

Usage: python -m vkflame.install
       python -m vkflame.install --undo
"""
import sys
import os
import platform
from pathlib import Path


def _find_shims() -> list[str]:
    """Locate the built shim shared-library files."""
    is_win = platform.system() == "Windows"
    candidates = [
        Path(__file__).parent.parent.parent / "build",
        Path(__file__).parent.parent.parent / "build" / "Release",
        Path("/usr/local/lib"),
    ]
    if is_win:
        shims = ["hipblaslt.dll", "hipblas.dll", "amdhip64.dll"]
    else:
        shims = ["libhipblaslt.so.0", "libhipblas.so.2", "libamdhip64.so.6"]

    for base in candidates:
        found = [str(base / s) for s in shims if (base / s).exists()]
        if len(found) == len(shims):
            return found

    raise FileNotFoundError(
        "Could not find vkflame shims. Run: cmake -B build && cmake --build build"
    )


def install() -> None:
    shims = _find_shims()
    is_win = platform.system() == "Windows"

    env_dir = Path.home() / ".config" / "vkflame"
    env_dir.mkdir(parents=True, exist_ok=True)

    if is_win:
        # On Windows write a batch file that prepends to PATH
        env_file = env_dir / "env.bat"
        shim_dir = str(Path(shims[0]).parent)
        env_file.write_text(f'@echo off\r\nset "PATH={shim_dir};%PATH%"\r\n')
        print(f"[vkflame] installed. Shim directory added to PATH via {env_file}")
        print(f"  Run: {env_file}")
    else:
        ld_preload = ":".join(shims)
        env_file = env_dir / "env.sh"
        env_file.write_text(f'export LD_PRELOAD="{ld_preload}"\n')

        # Append source line to .bashrc if not already present
        bashrc = Path.home() / ".bashrc"
        source_line = f'[ -f {env_file} ] && source {env_file}'
        if bashrc.exists() and source_line not in bashrc.read_text():
            with bashrc.open("a") as f:
                f.write(f"\n# vkflame shims\n{source_line}\n")

        print(f"[vkflame] installed. LD_PRELOAD will be set in new shells.")
        print(f"  Run: source {env_file}")
        print(f"  Or start a new terminal.")


def uninstall() -> None:
    env_dir = Path.home() / ".config" / "vkflame"
    removed = False
    for name in ("env.sh", "env.bat"):
        f = env_dir / name
        if f.exists():
            f.unlink()
            print(f"[vkflame] removed {f}")
            removed = True
    if not removed:
        print("[vkflame] nothing to remove")


if __name__ == "__main__":
    if "--undo" in sys.argv:
        uninstall()
    else:
        install()
