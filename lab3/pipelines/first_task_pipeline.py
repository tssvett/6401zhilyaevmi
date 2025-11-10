from collections import defaultdict
from typing import Iterator, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd

from lab3.pipelines.base_pipiline import BasePipeline


class FirstTaskPipeline(BasePipeline):
    """Пайплайн для задачи 1: Агрегация данных по температуре"""

    @staticmethod
    def extract_temperature_data(weather_chunks: Iterator[pd.DataFrame]) -> Iterator[Tuple[str, int, float]]:
        """Генератор для извлечения данных о температуре по локациям и годам"""
        for chunk in weather_chunks:
            for _, row in chunk.iterrows():
                try:
                    location = row['Station.Location']
                    year = int(row['Date.Year'])
                    avg_temp = float(row['Data.Temperature.Avg Temp'])
                    yield location, year, avg_temp
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def aggregate_temperature_data(temp_data: Iterator[Tuple[str, int, float]]) -> Iterator[Tuple[str, float]]:
        """Генератор для агрегации среднегодовых температур по локациям"""
        location_year_data = defaultdict(list)

        for location, year, temp in temp_data:
            location_year_data[(location, year)].append(temp)

        location_annual_avg = defaultdict(list)
        for (location, year), temps in location_year_data.items():
            annual_avg = sum(temps) / len(temps)
            location_annual_avg[location].append(annual_avg)

        for location, annual_temps in location_annual_avg.items():
            overall_avg = sum(annual_temps) / len(annual_temps)
            yield location, overall_avg

    @staticmethod
    def get_top_bottom_locations(avg_temperatures: Iterator[Tuple[str, float]], n: int = 3) -> Tuple[
        List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Получить топ-N самых теплых и холодных локаций"""
        all_locations = list(avg_temperatures)

        if not all_locations:
            return [], []

        sorted_locations = sorted(all_locations, key=lambda x: x[1], reverse=True)
        top_locations = sorted_locations[:n]
        bottom_locations = sorted_locations[-n:]

        return top_locations, bottom_locations

    @staticmethod
    def plot_temperature_comparison(top_locations: List[Tuple[str, float]], bottom_locations: List[Tuple[str, float]]):
        """Построить bar plot для сравнения температур"""
        locations = [loc[0] for loc in top_locations + bottom_locations]
        temperatures = [loc[1] for loc in top_locations + bottom_locations]
        colors = ['red'] * len(top_locations) + ['blue'] * len(bottom_locations)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(locations, temperatures, color=colors, alpha=0.7)
        plt.title('Самые теплые и холодные локации по среднегодовой температуре', fontsize=14)
        plt.xlabel('Локации')
        plt.ylabel('Среднегодовая температура (°F)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        for bar, temp in zip(bars, temperatures):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{temp: .1f}°F', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def run(self, file_path: str):
        """Запуск пайплайна для задачи 1"""
        print("=== ЗАДАЧА 1: Топ-3 самых теплых и холодных локаций ===")

        raw_data = self.read_weather_data(file_path)
        temp_data = self.extract_temperature_data(raw_data)
        aggregated_data = self.aggregate_temperature_data(temp_data)
        top_locations, bottom_locations = self.get_top_bottom_locations(aggregated_data)

        print("Топ-3 самых теплых локаций:")
        for location, temp in top_locations:
            print(f"  {location}: {temp: .2f}°F")

        print("\nТоп-3 самых холодных локаций:")
        for location, temp in bottom_locations:
            print(f"  {location}: {temp: .2f}°F")

        self.plot_temperature_comparison(top_locations, bottom_locations)
