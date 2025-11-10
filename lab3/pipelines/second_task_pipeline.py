import matplotlib.pyplot as plt
from typing import Iterator, Tuple, List
from collections import defaultdict
import statistics
import math
import pandas as pd

from lab3.pipelines.base_pipiline import BasePipeline


class SecondTaskPipeline(BasePipeline):
    """Пайплайн для задачи 2: Дисперсия и доверительный интервал"""

    @staticmethod
    def extract_state_monthly_temperature_data(weather_chunks: Iterator[pd.DataFrame]) -> Iterator[
           Tuple[str, int, int, float]]:
        """Генератор для извлечения данных о температуре по штатам, годам и месяцам"""
        for chunk in weather_chunks:
            for _, row in chunk.iterrows():
                try:
                    state = row['Station.State']
                    year = int(row['Date.Year'])
                    month = int(row['Date.Month'])
                    avg_temp = float(row['Data.Temperature.Avg Temp'])
                    yield state, year, month, avg_temp
                except (ValueError, KeyError):
                    continue

    @staticmethod
    def calculate_monthly_avg_temperatures(temp_data: Iterator[Tuple[str, int, int, float]]) -> Iterator[
           Tuple[str, int, int, float]]:
        """Генератор для расчета среднемесячных температур по штатам"""
        state_year_month_data = defaultdict(list)

        for state, year, month, temp in temp_data:
            state_year_month_data[(state, year, month)].append(temp)

        for (state, year, month), temps in state_year_month_data.items():
            monthly_avg = statistics.mean(temps)
            yield state, year, month, monthly_avg

    @staticmethod
    def calculate_temperature_variance_by_state(monthly_avg_data: Iterator[Tuple[str, int, int, float]]) -> Iterator[
           Tuple[str, float, List[float]]]:
        """Генератор для вычисления дисперсии среднемесячных температур по штатам"""
        state_year_temps = defaultdict(lambda: defaultdict(list))

        for state, year, month, temp in monthly_avg_data:
            state_year_temps[state][year].append(temp)

        for state, year_data in state_year_temps.items():
            annual_variances = []
            for year, monthly_temps in year_data.items():
                if len(monthly_temps) >= 2:
                    variance = statistics.variance(monthly_temps)
                    annual_variances.append(variance)

            if annual_variances:
                avg_variance = statistics.mean(annual_variances)
                yield state, avg_variance, annual_variances

    @staticmethod
    def calculate_confidence_interval(variances: List[float]) -> Tuple[float, float]:
        """Вычисление доверительного интервала для дисперсий"""
        if len(variances) < 2:
            return variances[0], variances[0] if variances else (0, 0)

        mean_var = statistics.mean(variances)
        stderr = statistics.stdev(variances) / math.sqrt(len(variances))

        z_score = 1.96
        margin_of_error = z_score * stderr
        return mean_var - margin_of_error, mean_var + margin_of_error

    @staticmethod
    def get_top_bottom_states_by_variance(state_variances: Iterator[Tuple[str, float, List[float]]], n: int = 3) -> \
            Tuple[List[Tuple[str, float, Tuple[float, float]]], List[Tuple[str, float, Tuple[float, float]]]]:
        """Получить топ-N штатов с наибольшей и наименьшей дисперсией температур"""
        all_states = list(state_variances)

        if not all_states:
            return [], []

        sorted_states = sorted(all_states, key=lambda x: x[1], reverse=True)

        top_states = []
        bottom_states = []

        for state, avg_variance, variances in sorted_states[:n]:
            ci_low, ci_high = SecondTaskPipeline.calculate_confidence_interval(variances)
            top_states.append((state, avg_variance, (ci_low, ci_high)))

        for state, avg_variance, variances in sorted_states[-n:]:
            ci_low, ci_high = SecondTaskPipeline.calculate_confidence_interval(variances)
            bottom_states.append((state, avg_variance, (ci_low, ci_high)))

        return top_states, bottom_states

    @staticmethod
    def plot_variance_with_confidence_intervals(top_states: List[Tuple[str, float, Tuple[float, float]]],
                                                bottom_states: List[Tuple[str, float, Tuple[float, float]]]):
        """Построить bar plot с доверительными интервалами"""
        all_states = top_states + bottom_states
        states = [state[0] for state in all_states]
        variances = [state[1] for state in all_states]
        ci_errors = [[state[1] - state[2][0] for state in all_states],
                     [state[2][1] - state[1] for state in all_states]]

        colors = ['red'] * len(top_states) + ['blue'] * len(bottom_states)

        plt.figure(figsize=(14, 7))
        bars = plt.bar(states, variances, color=colors, alpha=0.7,
                       yerr=ci_errors, capsize=5, error_kw={'elinewidth': 2, 'capthick': 2})

        plt.title('Штаты с наибольшим и наименьшим разбросом среднемесячных температур', fontsize=14)
        plt.xlabel('Штаты')
        plt.ylabel('Дисперсия среднемесячных температур')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        for bar, var in zip(bars, variances):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{var: .2f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def run(self, file_path: str):
        """Запуск пайплайна для задачи 2"""
        print("\n=== ЗАДАЧА 2: Штаты с наибольшим и наименьшим разбросом температур ===")

        raw_data = self.read_weather_data(file_path)
        state_temp_data = self.extract_state_monthly_temperature_data(raw_data)
        monthly_avg_data = self.calculate_monthly_avg_temperatures(state_temp_data)
        state_variances = self.calculate_temperature_variance_by_state(monthly_avg_data)
        top_states, bottom_states = self.get_top_bottom_states_by_variance(state_variances)

        print("Топ-3 штата с наибольшим разбросом среднемесячных температур:")
        for state, variance, (ci_low, ci_high) in top_states:
            print(f"  {state}: дисперсия = {variance: .2f}, 95% ДИ = [{ci_low: .2f}, {ci_high: .2f}]")

        print("\nТоп-3 штата с наименьшим разбросом среднемесячных температур:")
        for state, variance, (ci_low, ci_high) in bottom_states:
            print(f"  {state}: дисперсия = {variance: .2f}, 95% ДИ = [{ci_low: .2f}, {ci_high: .2f}]")

        self.plot_variance_with_confidence_intervals(top_states, bottom_states)
