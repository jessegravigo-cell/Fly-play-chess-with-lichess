"""Streaming, resumable, verified download of the connectome source files.

Each file is streamed to `<name>.partial`, resumed with an HTTP Range request
if a partial file already exists, verified by SHA-256 and then renamed
atomically into place. A file that is already present and verified is skipped.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from openfly.connectome.sources import SOURCES, Source
from openfly.connectome.verify import sha256_file

ProgressCallback = Callable[[int, int], None]

CHUNK = 4 * 1024 * 1024
USER_AGENT = "openfly/0.1 (connectome downloader)"


class DownloadError(RuntimeError):
    pass


def is_verified(src: Source, root: Path | None = None) -> bool:
    """True when the file exists, has the pinned size and the pinned SHA-256."""
    path = src.path(root)
    if not path.exists() or path.stat().st_size != src.bytes:
        return False
    return sha256_file(path) == src.sha256


def download_source(
    src: Source,
    root: Path | None = None,
    progress: ProgressCallback | None = None,
    verify_existing: bool = True,
) -> Path:
    """Download one source if needed. Returns the final path.

    `progress(done_bytes, total_bytes)` is called after every chunk. Resumes a
    `.partial` file with a Range request when the server supports it and
    restarts from zero otherwise.
    """
    path = src.path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not verify_existing or is_verified(src, root):
            if progress is not None:
                progress(src.bytes, src.bytes)
            return path
        path.unlink()

    partial = path.with_name(path.name + ".partial")
    done = partial.stat().st_size if partial.exists() else 0
    if done > src.bytes:
        partial.unlink()
        done = 0

    while done < src.bytes:
        headers = {"User-Agent": USER_AGENT}
        if done > 0:
            headers["Range"] = f"bytes={done}-"
        request = urllib.request.Request(src.url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and done > 0:
                partial.unlink()
                done = 0
                continue
            raise DownloadError(f"{src.url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise DownloadError(f"{src.url}: {exc.reason}") from exc
        with response:
            status = response.status
            if done > 0 and status != 206:
                # Server ignored the range request; start over.
                partial.unlink(missing_ok=True)
                done = 0
                mode = "wb"
            else:
                mode = "ab" if done > 0 else "wb"
            with open(partial, mode) as fh:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, src.bytes)
        if done < src.bytes:
            # Connection closed early; loop resumes from the partial length.
            continue

    size = partial.stat().st_size
    if size != src.bytes:
        partial.unlink()
        raise DownloadError(f"{src.name}: downloaded {size} bytes, expected {src.bytes}")
    digest = sha256_file(partial)
    if digest != src.sha256:
        partial.unlink()
        raise DownloadError(f"{src.name}: sha256 {digest} does not match pinned {src.sha256}")
    os.replace(partial, path)
    return path


def download_all(
    root: Path | None = None,
    progress: Callable[[Source, int, int], None] | None = None,
) -> list[Path]:
    """Download every source that is missing or fails verification."""
    paths = []
    for src in SOURCES:
        cb = (lambda d, t, s=src: progress(s, d, t)) if progress is not None else None
        paths.append(download_source(src, root, cb))
    return paths
