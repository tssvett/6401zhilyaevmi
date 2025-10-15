import csv
from typing import Iterator, Dict, Any


class BasePipeline:
    """Базовый класс для пайплайнов погодных данных"""

    @staticmethod
    def read_weather_data(file_path: str) -> Iterator[Dict[str, Any]]:
        """Генератор для чтения CSV файла с погодными данными"""
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                yield row
