import os
import sys
import time

import pytest

from ocr.isolate import GB, memory_limit, run_limited

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="memory read from /proc")


def double(x):
    return {"value": 2 * x, "pid": os.getpid()}


def fail():
    raise ValueError("broken file")


def hog(mb):
    blocks = []
    for _ in range(mb // 50):
        blocks.append(b"\x01" * (50 * 2 ** 20))  # written, so really resident
        time.sleep(0.05)
    return len(blocks)


def hog_and_fork(mb):
    block = b"\x01" * (mb * 2 ** 20)
    pid = os.fork()  # as subprocess does to start Tesseract: the child shares the parent's memory
    if pid == 0:
        time.sleep(1.0)
        os._exit(0)
    os.waitpid(pid, 0)
    return len(block) // 2 ** 20


def die():
    os._exit(3)


def big_result():
    return "x" * 2_000_000  # more than a pipe buffer


def test_the_function_runs_in_another_process_and_its_result_comes_back():
    status, result = run_limited(double, (21,), limit=None)
    assert status == "ok" and result["value"] == 42 and result["pid"] != os.getpid()
    assert run_limited(big_result, (), limit=None) == ("ok", "x" * 2_000_000)


def test_an_exception_or_a_dead_process_is_reported():
    assert run_limited(fail, (), limit=None) == ("error", "ValueError: broken file")
    status, message = run_limited(die, (), limit=None)
    assert status == "error" and "exit code 3" in message


@linux
def test_a_process_over_the_memory_limit_is_stopped():
    t0 = time.perf_counter()
    status, peak = run_limited(hog, (2000,), limit=300 * 2 ** 20, poll_s=0.05)
    assert status == "memory" and 300 * 2 ** 20 < peak < 1 * GB
    assert time.perf_counter() - t0 < 10
    assert run_limited(hog, (100,), limit=1 * GB, poll_s=0.05) == ("ok", 2)


@linux
def test_memory_shared_with_a_forked_child_counts_once():
    assert run_limited(hog_and_fork, (300,), limit=500 * 2 ** 20, poll_s=0.05) == ("ok", 300)


@linux
def test_the_limit_leaves_memory_for_the_rest_of_the_machine(monkeypatch):
    import ocr.isolate as iso

    monkeypatch.setattr(iso, "available_bytes", lambda: 5 * GB)
    assert memory_limit(6.0, 1.0) == 4 * GB
    assert memory_limit(3.0, 1.0) == 3 * GB
    monkeypatch.setattr(iso, "available_bytes", lambda: 1 * GB)
    assert memory_limit(6.0, 1.0) == GB // 2  # never below half a gigabyte
    assert memory_limit(0, 1.0) is None
