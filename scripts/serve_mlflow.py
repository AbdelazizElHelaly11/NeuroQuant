"""
NeuroQuant — MLflow UI launcher with LAN/public sharing presets.

The launcher wraps ``mlflow ui`` so the same command works whether you
want a private localhost UI, a LAN-visible UI for collaborators on the
same network, or a tunneled public URL via ``cloudflared`` / ``ngrok``.

Examples
--------

  # 1) Default: local-only (127.0.0.1:5000), safest.
  python scripts/serve_mlflow.py

  # 2) LAN: bind on all interfaces so other machines on the same network
  #    can open http://<your-LAN-ip>:5000. Add basic auth at the proxy.
  python scripts/serve_mlflow.py --host 0.0.0.0 --port 5000

  # 3) Public via Cloudflare quick-tunnel (requires cloudflared installed):
  python scripts/serve_mlflow.py --host 127.0.0.1 --tunnel cloudflared

  # 4) Public via ngrok (requires ``ngrok config add-authtoken ...`` once):
  python scripts/serve_mlflow.py --host 127.0.0.1 --tunnel ngrok

The script never publishes to 0.0.0.0 unless ``--host 0.0.0.0`` is
explicitly requested, and it prints a security warning so a public bind
is never accidental.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_backend(backend_uri: str) -> str:
    p = Path(backend_uri).expanduser().resolve()
    return f"file:///{p.as_posix().lstrip('/')}" if p.exists() else backend_uri


def _build_mlflow_command(host: str, port: int, backend_uri: str) -> list:
    """Return the command list. Tries the ``mlflow`` CLI first, falls
    back to ``python -m mlflow`` so it works even when only the package
    is installed without the shim on PATH."""
    if shutil.which("mlflow"):
        base = ["mlflow"]
    else:
        base = [sys.executable, "-m", "mlflow"]
    return base + [
        "ui",
        "--backend-store-uri", _resolve_backend(backend_uri),
        "--host", host,
        "--port", str(port),
    ]


def _spawn_tunnel(provider: str, port: int) -> subprocess.Popen:
    """Spawn a tunnel process (``cloudflared`` or ``ngrok``) pointed at the
    local MLflow port. Returns the running Popen so the caller can keep it
    alive alongside the UI process. Raises ``FileNotFoundError`` if the
    requested tunnel binary is not on PATH.
    """
    provider = provider.lower()
    if provider == "cloudflared":
        if not shutil.which("cloudflared"):
            raise FileNotFoundError(
                "cloudflared not found. Install from "
                "https://developers.cloudflare.com/cloudflare-one/connections/"
                "connect-apps/install-and-setup/installation/"
            )
        return subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"]
        )
    if provider == "ngrok":
        if not shutil.which("ngrok"):
            raise FileNotFoundError(
                "ngrok not found. Install from https://ngrok.com/download "
                "and run `ngrok config add-authtoken <YOUR_TOKEN>`."
            )
        return subprocess.Popen(["ngrok", "http", str(port)])
    raise ValueError(f"Unknown tunnel provider: {provider}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend-store-uri", default="sqlite:///mlflow.db",
        help="MLflow backend (default: sqlite:///mlflow.db).",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 to expose on the LAN. "
             "Default 127.0.0.1 (local only).",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port (default: 5000).",
    )
    parser.add_argument(
        "--tunnel", choices=["none", "cloudflared", "ngrok"], default="none",
        help="Optional tunnel for public sharing.",
    )
    args = parser.parse_args()

    if args.host == "0.0.0.0":
        print("[!] Binding on 0.0.0.0 — anyone on this network can read the "
              "MLflow UI. Put a reverse proxy + auth in front for production.",
              file=sys.stderr)

    cmd = _build_mlflow_command(args.host, args.port, args.backend_store_uri)
    print("[>] Launching:", " ".join(cmd))

    tunnel_proc = None
    if args.tunnel != "none":
        try:
            tunnel_proc = _spawn_tunnel(args.tunnel, args.port)
            print(f"[>] Tunnel started via {args.tunnel}; the public URL is "
                  f"printed in the tunnel's own output above.")
        except (FileNotFoundError, ValueError) as e:
            print(f"[!] Tunnel error: {e}", file=sys.stderr)
            return 2

    try:
        return subprocess.call(cmd)
    finally:
        if tunnel_proc is not None and tunnel_proc.poll() is None:
            tunnel_proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
