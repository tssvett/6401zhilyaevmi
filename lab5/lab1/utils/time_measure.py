import time

import cv2
import numpy as np


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} выполнено за {end_time - start_time:.4f} секунд")
        return result

    return wrapper


# Если нужно преобразовать черно-белое в цветное
def ensure_3_channels(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:  # черно-белое (H, W)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:  # одноканальное (H, W, 1)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:  # уже цветное (H, W, 3)
        return image
