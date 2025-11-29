from typing import Generator

import pandas as pd

from lab1.utils.time_measure import measure_time

CHUNK_SIZE = 1000


class BasePipeline:
    """Базовый класс для пайплайнов погодных данных"""

    @staticmethod
    @measure_time
    def read_weather_data(file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Генератор для чтения CSV файла с погодными данными"""
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
            yield chunk
