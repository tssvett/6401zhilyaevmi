import matplotlib.pyplot as plt
from typing import Iterator, Tuple, List
from collections import defaultdict
import statistics
import pandas as pd

from lab3.pipelines.base_pipiline import BasePipeline


class ThirdTaskPipeline(BasePipeline):
    """Пайплайн для задачи 3: Изменение во времени и скользящее среднее"""

    @staticmethod
    def extract_wind_data(weather_chunks: Iterator[pd.DataFrame]) -> Iterator[Tuple[str, str, float]]:
        """Генератор для извлечения данных о скорости ветра по штатам и датам"""
        for chunk in weather_chunks:
            for _, row in chunk.iterrows():
                try:
                    state = row['Station.State']
                    date = row['Date.Full']
                    wind_speed = float(row['Data.Wind.Speed'])
                    yield state, date, wind_speed
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def find_windiest_state(wind_data: Iterator[Tuple[str, str, float]]) -> str:
        """Найти самый ветреный штат по средней скорости ветра"""
        state_wind_speeds = defaultdict(list)

        for state, date, wind_speed in wind_data:
            state_wind_speeds[state].append(wind_speed)

        state_avg_speeds = {}
        for state, speeds in state_wind_speeds.items():
            state_avg_speeds[state] = statistics.mean(speeds)

        windiest_state = max(state_avg_speeds.items(), key=lambda x: x[1])
        return windiest_state[0]

    @staticmethod
    def extract_wind_data_for_state(weather_chunks: Iterator[pd.DataFrame], target_state: str) -> Iterator[
           Tuple[str, float]]:
        """Генератор для извлечения данных о скорости ветра для конкретного штата"""
        for chunk in weather_chunks:
            for _, row in chunk.iterrows():
                try:
                    state = row['Station.State']
                    if state == target_state:
                        date = row['Date.Full']
                        wind_speed = float(row['Data.Wind.Speed'])
                        yield date, wind_speed
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def calculate_moving_average(wind_data: Iterator[Tuple[str, float]], window_size: int = 30) -> Iterator[
           Tuple[str, float]]:
        """Генератор для расчета скользящего среднего скорости ветра"""
        dates = []
        wind_speeds = []

        for date, wind_speed in wind_data:
            dates.append(date)
            wind_speeds.append(wind_speed)

        # Сортируем по дате
        sorted_data = sorted(zip(dates, wind_speeds), key=lambda x: x[0])
        dates_sorted, wind_speeds_sorted = zip(*sorted_data)

        # Вычисляем скользящее среднее
        for i in range(len(wind_speeds_sorted)):
            if i < window_size - 1:
                continue
            window = wind_speeds_sorted[i - window_size + 1:i + 1]
            moving_avg = sum(window) / len(window)
            yield dates_sorted[i], moving_avg

    @staticmethod
    def plot_wind_data_with_moving_average(dates: List[str], wind_speeds: List[float], moving_avgs: List[float],
                                           state: str):
        """Построить line plot скорости ветра со скользящим средним"""
        plt.figure(figsize=(14, 7))

        # Исходные данные
        plt.plot(dates, wind_speeds, alpha=0.3, color='blue', label='Скорость ветра')

        # Скользящее среднее
        plt.plot(dates[len(dates) - len(moving_avgs):], moving_avgs, color='red',
                 linewidth=2, label=f'Скользящее среднее ({30} дней)')

        plt.title(f'Скорость ветра в самом ветреном штате ({state})', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Скорость ветра')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self, file_path: str):
        """Запуск пайплайна для задачи 3"""
        print("\n=== ЗАДАЧА 3: Скорость ветра в самом ветреном штате ===")

        # Находим самый ветреный штат
        raw_data_1 = self.read_weather_data(file_path)
        wind_data = self.extract_wind_data(raw_data_1)
        windiest_state = self.find_windiest_state(wind_data)
        print(f"Самый ветреный штат: {windiest_state}")

        # Собираем данные для графика
        raw_data_2 = self.read_weather_data(file_path)
        state_wind_data = self.extract_wind_data_for_state(raw_data_2, windiest_state)

        all_dates = []
        all_wind_speeds = []
        for date, wind_speed in state_wind_data:
            all_dates.append(date)
            all_wind_speeds.append(wind_speed)

        # Вычисляем скользящее среднее
        raw_data_3 = self.read_weather_data(file_path)
        state_wind_data_2 = self.extract_wind_data_for_state(raw_data_3, windiest_state)
        moving_avg_data = self.calculate_moving_average(state_wind_data_2)

        moving_avg_dates = []
        moving_avg_values = []
        for date, avg in moving_avg_data:
            moving_avg_dates.append(date)
            moving_avg_values.append(avg)

        self.plot_wind_data_with_moving_average(all_dates, all_wind_speeds, moving_avg_values, windiest_state)
