"""Run a function in its own process under a memory limit.

``python -m ocr`` reads every file this way: when a file needs more memory than the limit, its
process is stopped and the file reported, instead of the machine swapping until it freezes (or
the system killing the IDE the run was started from). A fresh process per file also gives back
what the engines keep between images (OpenVINO keeps buffers for the largest image it saw).

The memory used is the proportional set size (Pss) of the process and its children (Tesseract),
read from /proc every ``poll_s``: Linux only. Pss, not the resident set: a child forked to start
Tesseract shares all its parent's memory until it runs Tesseract, and resident sets would count
that memory twice. Elsewhere the function still runs in its own process,
without a limit.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
from pathlib import Path

GB = 1024 ** 3


def _pss(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _children(pid: int) -> list[int]:
    out = []
    try:
        for task in Path(f"/proc/{pid}/task").iterdir():
            out += [int(c) for c in (task / "children").read_text().split()]
    except (OSError, ValueError):
        pass
    return out


def tree(pid: int) -> list[int]:
    """The process and all its descendants."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += _children(p)
    return out


def tree_memory(pid: int) -> int:
    return sum(_pss(p) for p in tree(pid))


def available_bytes() -> int | None:
    """MemAvailable of the machine (None where /proc/meminfo is missing)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def memory_limit(max_gb: float, reserve_gb: float) -> int | None:
    """min(max_gb, available - reserve_gb) in bytes; None: no limit (max_gb 0, or no /proc)."""
    if max_gb <= 0 or not Path("/proc/self/smaps_rollup").exists():
        return None
    limit = int(max_gb * GB)
    avail = available_bytes()
    if avail is not None:
        limit = min(limit, avail - int(reserve_gb * GB))
    return max(limit, int(0.5 * GB))


def _child(conn, fn, args):
    try:
        conn.send(("ok", fn(*args)))
    except BaseException as e:  # reported to the parent, which goes on with the next file
        conn.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()


def run_limited(fn, args: tuple, limit: int | None, poll_s: float = 0.2) -> tuple[str, object]:
    """``fn(*args)`` in a new process (``fn`` and ``args`` picklable). Returns ("ok", result),
    ("error", message) when it raised or died, or ("memory", peak bytes) when it went over
    ``limit`` and was stopped."""
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(child, fn, args), daemon=True)
    # spawn runs the caller's main script again in the new process; a script that is not a
    # file (python - <<EOF in setup.sh, an IDE console) cannot be, and is not needed there
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    hide = main_file is not None and not os.path.exists(main_file)
    if hide:
        del main.__file__
    try:
        proc.start()
    finally:
        if hide:
            main.__file__ = main_file
    child.close()
    peak, result = 0, None
    try:
        while True:
            if parent.poll(poll_s):
                try:
                    result = parent.recv()
                except EOFError:
                    pass
                break
            if limit is not None:
                used = tree_memory(proc.pid)
                peak = max(peak, used)
                if used > limit:
                    for p in reversed(tree(proc.pid)):
                        try:
                            os.kill(p, signal.SIGKILL)
                        except OSError:
                            pass
                    proc.join()
                    return "memory", peak
            if not proc.is_alive():
                if parent.poll(0):  # sent just before it ended
                    try:
                        result = parent.recv()
                    except EOFError:
                        pass
                break
        proc.join()
    finally:
        parent.close()
        if proc.is_alive():
            proc.kill()
            proc.join()
    if result is None:
        return "error", f"the process ended without a result (exit code {proc.exitcode})"
    return result
