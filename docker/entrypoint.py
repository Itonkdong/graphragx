"""Wait for the Compose Qdrant service, then execute the requested command."""

from __future__ import annotations

import os
import socket
import sys
import time
from urllib.parse import urlparse


def _qdrant_address() -> tuple[str, int]:
    parsed = urlparse(os.getenv("QDRANT_URL", "http://qdrant:6333"))
    if not parsed.hostname:
        raise SystemExit("QDRANT_URL must contain a hostname.")
    return parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def _wait_for_qdrant() -> None:
    if os.getenv("GRAPHRAGX_WAIT_FOR_QDRANT", "1").lower() in {
        "0",
        "false",
        "no",
    }:
        return

    host, port = _qdrant_address()
    timeout_seconds = float(os.getenv("GRAPHRAGX_QDRANT_WAIT_SECONDS", "60"))
    deadline = time.monotonic() + timeout_seconds

    print(f"Waiting for Qdrant at {host}:{port} ...", flush=True)
    while True:
        try:
            with socket.create_connection((host, port), timeout=2):
                print("Qdrant is ready.", flush=True)
                return
        except OSError as error:
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"Qdrant at {host}:{port} was not reachable within "
                    f"{timeout_seconds:g} seconds: {error}"
                ) from error
            time.sleep(1)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("No container command was provided.")
    _wait_for_qdrant()
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
