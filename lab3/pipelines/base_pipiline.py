from typing import Iterator

import pandas as pd

CHUNK_SIZE = 1000


class BasePipeline:
    """Базовый класс для пайплайнов погодных данных"""

    @staticmethod
    def read_weather_data(file_path: str) -> Iterator[pd.DataFrame]:
        """Генератор для чтения CSV файла с погодными данными"""
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
            yield chunk
