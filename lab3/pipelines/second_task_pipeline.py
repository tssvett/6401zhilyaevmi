from typing import Generator, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from lab1.utils.time_measure import measure_time
from lab3.pipelines.base_pipiline import BasePipeline
from lab3.utils.utils import memory_logger
from matplotlib.patches import Patch


class SecondTaskPipeline(BasePipeline):
    """Пайплайн для задачи 2: Анализ разброса среднемесячных температур по штатам"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    @measure_time
    def get_data(
            self,
            columns: List[str] = ['Station.State', 'Date.Month', 'Date.Year', 'Data.Temperature.Avg Temp']
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Метод для загрузки данных о среднемесячных температурах по штатам
        """
        for chunk in self.read_weather_data(self.file_path):
            chunk = chunk[columns].copy()

            # Преобразуем в числовые типы
            chunk['Date.Month'] = pd.to_numeric(chunk['Date.Month'], errors='coerce')
            chunk['Date.Year'] = pd.to_numeric(chunk['Date.Year'], errors='coerce')
            chunk['Data.Temperature.Avg Temp'] = pd.to_numeric(chunk['Data.Temperature.Avg Temp'], errors='coerce')

            # Удаляем строки с NaN
            chunk = chunk.dropna(subset=['Date.Month', 'Date.Year', 'Data.Temperature.Avg Temp'])

            # Фильтруем корректные месяцы и годы
            chunk = chunk[(chunk['Date.Month'] >= 1) & (chunk['Date.Month'] <= 12)]
            chunk = chunk[chunk['Date.Year'] > 0]

            if len(chunk) > 0:
                yield chunk

    @measure_time
    def aggregate_data(self, data):
        monthly_agg = None

        for chunk in data:
            chunk_agg = chunk.groupby(['Station.State', 'Date.Year', 'Date.Month']).agg(
                sum_temp=('Data.Temperature.Avg Temp', 'sum'),
                count=('Data.Temperature.Avg Temp', 'count')
            ).reset_index().rename(columns={
                'Station.State': 'State',
                'Date.Year': 'Year',
                'Date.Month': 'Month'
            }).set_index(['State', 'Year', 'Month']).astype({
                'sum_temp': float,
                'count': int
            })
            if monthly_agg is None:
                monthly_agg = chunk_agg

            monthly_agg = monthly_agg.add(chunk_agg, fill_value=0)

        if monthly_agg.empty:
            return pd.DataFrame(columns=['State', 'mean', 'std', 'count', 'variance'])

        # Вычисляем среднемесячные температуры
        monthly_agg['monthly_avg'] = monthly_agg['sum_temp'] / monthly_agg['count']

        result = monthly_agg.groupby('State').agg(
            mean=('monthly_avg', 'mean'),
            variance=('monthly_avg', 'var'),
            count=('monthly_avg', 'count')
        ).reset_index()

        result['std'] = np.sqrt(result['variance'].fillna(0))
        return result[['State', 'mean', 'std', 'count', 'variance']]

    @measure_time
    def task_job(self, data: pd.DataFrame) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], dict]:
        """
        Находим штаты с самым высоким и низким разбросом температур + доверительные интервалы
        """
        if data.empty:
            return [], [], {}

        # Сортируем по дисперсии (разбросу)
        sorted_var = data.sort_values('variance')

        # Берем топ-3 с самой низкой и высокой дисперсией
        lowest_var = list(sorted_var.head(3).itertuples(index=False, name=None))
        highest_var = list(sorted_var.tail(3).itertuples(index=False, name=None))

        # Вычисляем доверительные интервалы для выбранных штатов
        selected_states = [state for state, _, _, _, _ in lowest_var + highest_var]
        ci_data = {}

        for state in selected_states:
            state_stats = data[data['State'] == state].iloc[0]
            mean = state_stats['mean']
            std = state_stats['std']
            n = state_stats['count']

            if n > 1:
                # Стандартная ошибка среднего
                sem = std / np.sqrt(n)
                # 95% доверительный интервал (z-score для 95% = 1.96)
                ci = norm.ppf(0.975) * sem
                ci_data[state] = (mean - ci, mean + ci)
            else:
                ci_data[state] = (mean, mean)

        return lowest_var, highest_var, ci_data

    @staticmethod
    def plot_results(data: Any):
        """
        Метод для отрисовки результатов с доверительными интервалами
        """
        lowest_var, highest_var, ci_data = data

        if not lowest_var and not highest_var:
            print("Нет данных для построения графика")
            return

        # Подготавливаем данные для визуализации
        all_states = [x[0] for x in lowest_var] + [x[0] for x in highest_var]

        # Вычисляем средние значения и ошибки для доверительных интервалов
        means = [np.mean(ci_data[state]) for state in all_states]
        errors = [
            [means[i] - ci_data[state][0] for i, state in enumerate(all_states)],
            [ci_data[state][1] - means[i] for i, state in enumerate(all_states)]
        ]

        plt.figure(figsize=(12, 8))

        # Создаем бар-чарт с доверительными интервалами
        bars = plt.bar(all_states, means, yerr=errors, capsize=5,
                       color=['green'] * len(lowest_var) + ['red'] * len(highest_var),
                       alpha=0.7)

        plt.title('Штаты с наименьшим и наибольшим разбросом среднемесячных температур\n(95% доверительные интервалы)',
                  fontsize=14)
        plt.xlabel('Штаты')
        plt.ylabel('Средняя температура (°F)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        # Добавляем легенду
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Маленький разброс'),
            Patch(facecolor='red', alpha=0.7, label='Большой разброс')
        ]
        plt.legend(handles=legend_elements)

        plt.tight_layout()
        plt.show()

        print(f"Штаты с маленьким разбросом: {lowest_var}")
        print(f"Штаты с большим разбросом: {highest_var}")
        print(f"Доверительные интервалы: {ci_data}")

    @memory_logger
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        print("=== ЗАДАЧА 2: Топ-3 штатов с самым высоким и низким разбросом среднемесячных температур ===")
        self.plot_results(self.task_job(self.aggregate_data(self.get_data())))
