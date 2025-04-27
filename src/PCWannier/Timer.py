import time
import functools
from .Log import Logger

class Timer:
    def __init__(self, label=""):
        self.label = label
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        elapsed = self.end - self.start
        Logger.info(f"{self.label} takes {elapsed:.4f} s")

def timer(label=""):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            Logger.info(f"{label} takes {elapsed:.4f} s")
            return result
        return wrapper
    return decorator
