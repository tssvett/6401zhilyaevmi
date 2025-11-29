import functools
import os

import psutil as psutil


def memory_logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024

        result = func(*args, **kwargs)

        memory_after = process.memory_info().rss / 1024 / 1024
        memory_used = memory_after - memory_before

        print(f"Метод {func.__name__} использовал {memory_used:.2f} МБ памяти")
        return result

    return wrapper
