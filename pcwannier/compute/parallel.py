from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


def parallel_map(items, func, threads: int):
    iterator = iter(items)
    if threads <= 1:
        for item in iterator:
            yield func(item)
        return

    max_pending = max(1, int(threads) * 2)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        pending = set()
        exhausted = False

        while not exhausted and len(pending) < max_pending:
            try:
                pending.add(executor.submit(func, next(iterator)))
            except StopIteration:
                exhausted = True

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                if not exhausted:
                    try:
                        pending.add(executor.submit(func, next(iterator)))
                    except StopIteration:
                        exhausted = True
