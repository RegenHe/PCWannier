from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import AbstractContextManager
from threading import local
from typing import Callable, Iterable, Iterator, TypeVar

from threadpoolctl import threadpool_limits


_T = TypeVar("_T")
_R = TypeVar("_R")
_THREAD_STATE = local()


def _set_numba_threads(count: int) -> None:
    try:
        import numba

        numba.set_num_threads(max(1, int(count)))
    except (ImportError, RuntimeError, ValueError):
        return


def _get_numba_threads() -> int:
    try:
        import numba

        return max(1, int(numba.get_num_threads()))
    except (ImportError, RuntimeError, ValueError):
        return 1


def _initialize_worker(context: "ExecutionContext") -> None:
    _THREAD_STATE.context = context
    _THREAD_STATE.in_worker = True
    _THREAD_STATE.numba_parallel_allowed = False
    _set_numba_threads(1)


def numba_parallel_allowed() -> bool:
    return bool(getattr(_THREAD_STATE, "numba_parallel_allowed", True))


def set_numba_parallel_allowed(enabled: bool) -> bool:
    previous = numba_parallel_allowed()
    _THREAD_STATE.numba_parallel_allowed = bool(enabled)
    _set_numba_threads(1 if not enabled else max(1, getattr(_THREAD_STATE, "numba_threads", 1)))
    return previous


class ExecutionContext(AbstractContextManager["ExecutionContext"]):
    """Run-scoped owner of the total CPU and temporary-memory budget."""

    def __init__(
        self,
        threads: int,
        *,
        memory_budget_bytes: int = 1 << 30,
    ) -> None:
        self.threads = max(1, int(threads))
        self.memory_budget_bytes = max(1, int(memory_budget_bytes))
        self._executor: ThreadPoolExecutor | None = None
        self._previous_context = None
        self._previous_worker = False
        self._previous_numba = True
        self._previous_numba_threads = 1
        self._previous_thread_numba_threads = 1
        self._threadpool_limit = None

    def __enter__(self) -> "ExecutionContext":
        self._previous_context = getattr(_THREAD_STATE, "context", None)
        self._previous_worker = bool(getattr(_THREAD_STATE, "in_worker", False))
        self._previous_numba = numba_parallel_allowed()
        self._previous_numba_threads = _get_numba_threads()
        self._previous_thread_numba_threads = max(
            1, int(getattr(_THREAD_STATE, "numba_threads", self._previous_numba_threads))
        )
        _THREAD_STATE.context = self
        _THREAD_STATE.in_worker = False
        _THREAD_STATE.numba_parallel_allowed = False
        # Numba-parallel phases run on the caller thread and may consume the
        # whole run budget. Python workers are initialized with one Numba
        # thread, so the two forms of parallelism cannot nest.
        _THREAD_STATE.numba_threads = self.threads
        _set_numba_threads(1)
        self._threadpool_limit = threadpool_limits(limits=1)
        self._threadpool_limit.__enter__()
        if self.threads > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.threads,
                initializer=_initialize_worker,
                initargs=(self,),
                thread_name_prefix="pcwannier",
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=exc_type is not None)
            self._executor = None
        if self._threadpool_limit is not None:
            self._threadpool_limit.__exit__(exc_type, exc, tb)
            self._threadpool_limit = None
        _THREAD_STATE.context = self._previous_context
        _THREAD_STATE.in_worker = self._previous_worker
        _THREAD_STATE.numba_parallel_allowed = self._previous_numba
        _THREAD_STATE.numba_threads = self._previous_thread_numba_threads
        _set_numba_threads(self._previous_numba_threads)

    def _capacity(
        self,
        *,
        max_workers: int | None,
        bytes_per_task: int | None,
        memory_budget_bytes: int | None,
    ) -> int:
        capacity = self.threads
        if max_workers is not None:
            capacity = min(capacity, max(1, int(max_workers)))
        if bytes_per_task is not None:
            task_bytes = max(1, int(bytes_per_task))
            budget = self.memory_budget_bytes if memory_budget_bytes is None else max(
                1, int(memory_budget_bytes)
            )
            capacity = min(capacity, max(1, budget // task_bytes))
        return max(1, capacity)

    @staticmethod
    def _cancel(pending: Iterable[Future]) -> None:
        for future in pending:
            future.cancel()

    def map(
        self,
        items: Iterable[_T],
        func: Callable[[_T], _R],
        *,
        ordered: bool = True,
        max_workers: int | None = None,
        bytes_per_task: int | None = None,
        memory_budget_bytes: int | None = None,
    ) -> Iterator[_R]:
        iterator = iter(items)
        capacity = self._capacity(
            max_workers=max_workers,
            bytes_per_task=bytes_per_task,
            memory_budget_bytes=memory_budget_bytes,
        )
        if (
            self._executor is None
            or capacity == 1
            or bool(getattr(_THREAD_STATE, "in_worker", False))
        ):
            for item in iterator:
                yield func(item)
            return

        if ordered:
            yield from self._map_ordered(iterator, func, capacity)
        else:
            yield from self._map_unordered(iterator, func, capacity)

    def _map_ordered(
        self,
        iterator: Iterator[_T],
        func: Callable[[_T], _R],
        capacity: int,
    ) -> Iterator[_R]:
        assert self._executor is not None
        pending: deque[Future[_R]] = deque()

        def fill() -> None:
            while len(pending) < capacity:
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                pending.append(self._executor.submit(func, item))

        fill()
        try:
            while pending:
                future = pending.popleft()
                yield future.result()
                fill()
        except BaseException:
            self._cancel(pending)
            raise

    def _map_unordered(
        self,
        iterator: Iterator[_T],
        func: Callable[[_T], _R],
        capacity: int,
    ) -> Iterator[_R]:
        assert self._executor is not None
        pending: set[Future[_R]] = set()

        def fill() -> None:
            while len(pending) < capacity:
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                pending.add(self._executor.submit(func, item))

        fill()
        try:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    yield future.result()
                fill()
        except BaseException:
            self._cancel(pending)
            raise


def current_execution_context() -> ExecutionContext | None:
    return getattr(_THREAD_STATE, "context", None)


def parallel_map(
    items: Iterable[_T],
    func: Callable[[_T], _R],
    threads: int,
    *,
    ordered: bool = True,
    bytes_per_task: int | None = None,
    memory_budget_bytes: int | None = None,
) -> Iterator[_R]:
    active = current_execution_context()
    if active is not None:
        yield from active.map(
            items,
            func,
            ordered=ordered,
            max_workers=threads,
            bytes_per_task=bytes_per_task,
            memory_budget_bytes=memory_budget_bytes,
        )
        return
    with ExecutionContext(threads, memory_budget_bytes=memory_budget_bytes or (1 << 30)) as context:
        yield from context.map(
            items,
            func,
            ordered=ordered,
            max_workers=threads,
            bytes_per_task=bytes_per_task,
            memory_budget_bytes=memory_budget_bytes,
        )


def memory_limited_threads(
    requested: int,
    bytes_per_worker: int,
    *,
    budget_bytes: int = 1 << 30,
) -> int:
    """Limit field-level parallelism to a predictable temporary-memory budget."""

    count = max(1, int(requested))
    worker_bytes = max(1, int(bytes_per_worker))
    budget = max(worker_bytes, int(budget_bytes))
    return max(1, min(count, budget // worker_bytes))
