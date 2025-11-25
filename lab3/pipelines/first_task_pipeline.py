from typing import Generator, List, Tuple, Any

import matplotlib.pyplot as plt
import pandas as pd

from lab1.utils.time_measure import measure_time
from lab3.pipelines.base_pipiline import BasePipeline
from lab3.utils.utils import memory_logger


class FirstTaskPipeline(BasePipeline):
    """Пайплайн для задачи 1: Агрегация данных по температуре"""

    def __init__(self, file_path: str):
        """
        Инициализация пайплайна

        Args:
            file_path (str): Путь к файлу с данными о погоде
        """
        self.file_path = file_path

    @measure_time
    def get_data(
            self,
            columns: List[str] = ['Station.Location', 'Date.Year', 'Data.Temperature.Avg Temp']
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Метод для загрузки данных о температуре

        Args:
            columns (List[str]): Необходимые для загрузки столбцы
        Returns:
            pd.DataFrame | Generator[pd.DataFrame, None, None]: Либо полный DataFrame, либо генератор чанков DataFrame
        """
        for chunk in self.read_weather_data(self.file_path):
            # Фильтруем только нужные столбцы
            chunk = chunk[columns].copy()

            # Преобразуем типы данных в числовые (данные из CSV приходят как строки)
            chunk['Date.Year'] = pd.to_numeric(chunk['Date.Year'], errors='coerce')
            chunk['Data.Temperature.Avg Temp'] = pd.to_numeric(chunk['Data.Temperature.Avg Temp'], errors='coerce')

            # Удаляем строки с NaN
            chunk = chunk.dropna(subset=['Date.Year', 'Data.Temperature.Avg Temp'])

            # Фильтруем только корректные годы
            chunk = chunk[chunk['Date.Year'] > 0]

            if len(chunk) > 0:
                chunk['count'] = 1
                yield chunk[['Station.Location', 'Data.Temperature.Avg Temp', 'count']]

    @measure_time
    def aggregate_data(
            self,
            data: pd.DataFrame | Generator[pd.DataFrame, None, None]
    ) -> pd.DataFrame:
        """
        Метод для агрегации данных по температуре

        Args:
            data (pd.DataFrame | Generator[pd.DataFrame, None, None]): Данные для агрегирования

        Returns:
            pd.DataFrame: DataFrame с агрегированными данными по локациям
        """
        # Инициализируем пустой DataFrame с нужными колонками
        aggregated = pd.DataFrame(columns=['Station.Location', 'temp_sum', 'count'])

        for chunk in data:
            # Конкатенируем с общим DataFrame
            aggregated = pd.concat([aggregated, chunk], ignore_index=True)

            # Группируем по локациям для суммирования значений
            aggregated = aggregated.groupby('Station.Location').agg({
                'Data.Temperature.Avg Temp': 'sum',
                'count': 'sum'
            }).reset_index()

        # Переименовываем колонку для консистентности
        aggregated.columns = ['Station.Location', 'temp_sum', 'count']

        if len(aggregated) <= 0:
            return pd.DataFrame(columns=['Station.Location', 'avg_temperature'])
        # Вычисляем среднюю температуру
        aggregated['avg_temperature'] = aggregated['temp_sum'] / aggregated['count']
        return aggregated[['Station.Location', 'avg_temperature']]

    @measure_time
    def task_job(self, data: pd.DataFrame) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Метод для выполнения манипуляций над агрегированными данными

        Args:
            data (pd.DataFrame): Агрегированные данные по температуре

        Returns:
            Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
                (самые теплые локации, самые холодные локации) в формате [(локация, температура), ...]
        """
        if data.empty:
            return [], []

        # Сортируем по температуре
        sorted_data = data.sort_values('avg_temperature', ascending=False)

        # Берем топ-3 самых теплых и холодных локаций
        warmest = list(sorted_data.head(3).itertuples(index=False, name=None))
        coldest = list(sorted_data.tail(3).itertuples(index=False, name=None))

        return warmest, coldest

    @staticmethod
    def plot_results(data: Any):
        """
        Метод для отрисовки результатов работы

        Args:
            data (Any): Результат работы в формате (теплые, холодные)
        """
        warmest, coldest = data

        if not warmest and not coldest:
            print("Нет данных для построения графика")
            return

        # Подготавливаем данные для визуализации
        all_locations = warmest + coldest
        locations = [x[0] for x in all_locations]
        temperatures = [x[1] for x in all_locations]

        plt.figure(figsize=(12, 6))

        # Создаем бар-чарт с разными цветами для теплых и холодных локаций
        plt.bar(locations, temperatures, color=['red'] * len(warmest) + ['blue'] * len(coldest), alpha=0.7)

        plt.title('Самые теплые и холодные локации по среднегодовой температуре', fontsize=14)
        plt.xlabel('Локации')
        plt.ylabel('Среднегодовая температура (°F)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # Добавляем подписи значений
        for i, (location, temp) in enumerate(all_locations):
            plt.text(i, temp + 0.1, f'{temp:.1f}°F', ha='center', va='bottom')

        # Добавляем легенду
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Теплые локации'),
            Patch(facecolor='blue', alpha=0.7, label='Холодные локации')
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.show()

        print(f"Теплые локации: {warmest}")
        print(f"Холодные локации: {coldest}")

    @memory_logger
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        print("=== ЗАДАЧА 1: Топ-3 самых теплых и холодных локаций ===")
        self.plot_results(self.task_job(self.aggregate_data(self.get_data())))
