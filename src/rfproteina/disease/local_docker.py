"""Docker endpoints on the local screening host, including Windows-hosted WSL engines."""

import os
import shutil
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path


def local_engines() -> Iterator[list[str]]:
    docker = shutil.which("docker")
    if docker and os.name != "nt":
        sockets = [
            Path("/var/run/docker.sock"),
            Path.home() / ".docker/desktop/docker.sock",
            Path.home() / ".docker/run/docker.sock",
        ]
        host = os.environ.get("DOCKER_HOST", "")
        if host.startswith("unix:///"):
            sockets.insert(0, Path(host.removeprefix("unix://")))
        if runtime := os.environ.get("XDG_RUNTIME_DIR"):
            sockets.append(Path(runtime) / "docker.sock")
        for sock in dict.fromkeys(sockets):
            with suppress(OSError):
                if sock.is_absolute() and sock.is_socket():
                    yield [docker, "--host", f"unix://{sock}"]
    if windows := shutil.which("docker.exe"):
        for pipe in ("dockerDesktopLinuxEngine", "docker_engine"):
            yield [windows, "--host", f"npipe:////./pipe/{pipe}"]
