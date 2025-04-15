import time
import functools

class Timer:
    def __init__(self, label=""):
        self.label = label
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.perf_counter()
        elapsed = self.end - self.start
        print(f"{self.label} takes {elapsed:.4f} s")

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"func {func.__name__} takes {elapsed:.4f} s")
        return result
    return wrapper
