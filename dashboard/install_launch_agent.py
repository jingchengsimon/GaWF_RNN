"""Install or remove the dashboard as a persistent macOS LaunchAgent."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys


LABEL = "com.jingchengshi.faw-rnn-dashboard"
DASHBOARD_HOSTNAME = "Jingchengs-Mac-mini.local"
BIND_HOST = "private-lan"
DASHBOARD_DIR = Path(__file__).resolve().parent
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
ANACONDA_PYTHON = Path("/opt/anaconda3/bin/python3")
RUNTIME_ROOT = Path.home() / "Library" / "Application Support" / "FAW_RNN Dashboard"
RUNTIME_PACKAGE = RUNTIME_ROOT / "dashboard"
RUNTIME_REGISTRY = RUNTIME_ROOT / "tasks.json"
RUNTIME_CACHE = RUNTIME_ROOT / "status_cache.json"


def service_python() -> str:
    """Return a background-safe Python executable for the macOS LaunchAgent."""
    if ANACONDA_PYTHON.is_file():
        return str(ANACONDA_PYTHON)
    return sys.executable


def deploy_runtime() -> None:
    """Copy executable dashboard assets outside Desktop for LaunchAgent access."""
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        DASHBOARD_DIR,
        RUNTIME_PACKAGE,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.log", "tasks.json", "status_cache.json"
        ),
    )
    shutil.copy2(DASHBOARD_DIR / "tasks.json", RUNTIME_REGISTRY)


def plist_payload() -> dict[str, object]:
    return {
        "Label": LABEL,
        "ProgramArguments": [
            service_python(),
            "-m",
            "dashboard.server",
            "--host",
            BIND_HOST,
            "--port",
            "8765",
        ],
        "WorkingDirectory": str(RUNTIME_ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(RUNTIME_ROOT / "dashboard.stdout.log"),
        "StandardErrorPath": str(RUNTIME_ROOT / "dashboard.stderr.log"),
        "EnvironmentVariables": {
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "FAW_RNN_DASHBOARD_REGISTRY": str(RUNTIME_REGISTRY),
            "FAW_RNN_DASHBOARD_CACHE": str(RUNTIME_CACHE),
        },
    }


def run_launchctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["launchctl", *args], text=True, capture_output=True, check=check)


def install() -> None:
    deploy_runtime()
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLIST_PATH.open("wb") as handle:
        plistlib.dump(plist_payload(), handle)
    domain = f"gui/{os.getuid()}"
    run_launchctl("bootout", domain, str(PLIST_PATH), check=False)
    run_launchctl("bootstrap", domain, str(PLIST_PATH))
    run_launchctl("kickstart", "-k", f"{domain}/{LABEL}")
    print(f"Installed {LABEL}: http://{DASHBOARD_HOSTNAME}:8765/")


def uninstall() -> None:
    run_launchctl("bootout", f"gui/{os.getuid()}", str(PLIST_PATH), check=False)
    if PLIST_PATH.exists():
        PLIST_PATH.unlink()
    print(f"Removed {LABEL}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["install", "uninstall"])
    args = parser.parse_args()
    install() if args.action == "install" else uninstall()


if __name__ == "__main__":
    main()
