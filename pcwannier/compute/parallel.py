from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar


_ACTIVE_EXECUTOR: ContextVar["ParallelExecutor | None"] = ContextVar("pcwannier_parallel_executor", default=None)


class ParallelExecutor:
    def __init__(self, threads: int):
        self.threads = max(1, int(threads))
        self.max_pending = max(1, self.threads * 2)
        self._executor: ThreadPoolExecutor | None = None
        self._token = None

    def __enter__(self):
        if self.threads > 1:
            self._executor = ThreadPoolExecutor(max_workers=self.threads)
        self._token = _ACTIVE_EXECUTOR.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _ACTIVE_EXECUTOR.reset(self._token)
            self._token = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def map(self, items, func):
        iterator = iter(items)
        if self._executor is None:
            for item in iterator:
                yield func(item)
            return

        pending = deque()
        exhausted = False

        def fill_pending():
            nonlocal exhausted
            while not exhausted and len(pending) < self.max_pending:
                try:
                    pending.append(self._executor.submit(func, next(iterator)))
                except StopIteration:
                    exhausted = True

        fill_pending()
        while pending:
            future = pending.popleft()
            yield future.result()
            fill_pending()


def parallel_map(items, func, threads: int):
    active = _ACTIVE_EXECUTOR.get()
    if active is not None and active.threads == max(1, int(threads)):
        yield from active.map(items, func)
        return

    with ParallelExecutor(threads) as executor:
        yield from executor.map(items, func)
