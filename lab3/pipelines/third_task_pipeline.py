from typing import Generator, List, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt

from lab1.utils.time_measure import measure_time
from lab3.pipelines.base_pipiline import BasePipeline
from lab3.utils.utils import memory_logger


class ThirdTaskPipeline(BasePipeline):
    """Пайплайн для задачи 3: Скорость ветра в самом ветреном штате"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    @measure_time
    def get_data(
            self,
            columns: List[str] = ['Station.State', 'Date.Full', 'Data.Wind.Speed']
    ) -> pd.DataFrame | Generator[pd.DataFrame, None, None]:
        """
        Метод для загрузки данных о скорости ветра
        """
        for chunk in self.read_weather_data(self.file_path):
            chunk = chunk[columns].copy()

            # Преобразуем в числовые типы
            chunk['Data.Wind.Speed'] = pd.to_numeric(chunk['Data.Wind.Speed'], errors='coerce')

            # Удаляем строки с NaN
            chunk = chunk.dropna(subset=['Data.Wind.Speed'])

            if len(chunk) > 0:
                yield chunk

    @measure_time
    def aggregate_data(
            self,
            data: pd.DataFrame | Generator[pd.DataFrame, None, None]
    ) -> Tuple[str, pd.DataFrame]:
        """
        Метод для нахождения самого ветреного штата и его данных
        """
        all_data = pd.DataFrame()
        state_stats = pd.DataFrame(columns=['State', 'wind_sum', 'count'])

        for chunk in data:
            # Сохраняем все данные для последующей обработки
            all_data = pd.concat([all_data, chunk], ignore_index=True)

            # Одновременно собираем статистику по штатам
            chunk_agg = chunk.groupby('Station.State').agg({
                'Data.Wind.Speed': ['sum', 'count']
            }).reset_index()

            chunk_agg.columns = ['State', 'wind_sum', 'count']

            state_stats = pd.concat([state_stats, chunk_agg], ignore_index=True)
            state_stats = state_stats.groupby('State').agg({
                'wind_sum': 'sum',
                'count': 'sum'
            }).reset_index()

        if state_stats.empty:
            return "", pd.DataFrame()

        # Находим самый ветреный штат
        state_stats['avg_wind'] = state_stats['wind_sum'] / state_stats['count']
        windiest_state = state_stats.loc[state_stats['avg_wind'].idxmax(), 'State']

        # Фильтруем данные только для ветреного штата
        wind_data = all_data[all_data['Station.State'] == windiest_state].copy()
        wind_data = wind_data[['Date.Full', 'Data.Wind.Speed']].copy()
        wind_data.columns = ['Date', 'Wind_Speed']
        wind_data['Date'] = pd.to_datetime(wind_data['Date'])
        wind_data = wind_data.sort_values('Date')

        return windiest_state, wind_data

    @measure_time
    def task_job(self, data: Tuple[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        """
        Метод для вычисления скользящего среднего
        """
        windiest_state, wind_data = data

        if wind_data.empty:
            return "", pd.DataFrame(), pd.DataFrame()

        # Вычисляем скользящее среднее с окном 30 дней
        wind_data['Moving_Avg_30'] = wind_data['Wind_Speed'].rolling(
            window=30, min_periods=1, center=True
        ).mean()

        # Разделяем на исходные данные и скользящее среднее
        original_data = wind_data[['Date', 'Wind_Speed']].copy()
        moving_avg_data = wind_data[['Date', 'Moving_Avg_30']].copy()
        moving_avg_data = moving_avg_data.dropna()  # Убираем NaN по краям

        return windiest_state, original_data, moving_avg_data

    @staticmethod
    def plot_results(data: Any):
        """
        Метод для отрисовки результатов работы
        """
        windiest_state, original_data, moving_avg_data = data

        if original_data.empty:
            print("Нет данных для построения графика")
            return

        plt.figure(figsize=(14, 7))

        # Исходные данные
        plt.plot(original_data['Date'], original_data['Wind_Speed'],
                 alpha=0.3, color='blue', label='Скорость ветра', linewidth=0.5)

        # Скользящее среднее
        plt.plot(moving_avg_data['Date'], moving_avg_data['Moving_Avg_30'],
                 color='red', linewidth=2, label='Скользящее среднее (30 дней)')

        plt.title(f'Скорость ветра в самом ветреном штате ({windiest_state})', fontsize=14)
        plt.xlabel('Дата')
        plt.ylabel('Скорость ветра')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Выводим статистику
        print(f"Самый ветреный штат: {windiest_state}")
        print(f"Средняя скорость ветра: {original_data['Wind_Speed'].mean():.2f}")
        print(f"Максимальная скорость ветра: {original_data['Wind_Speed'].max():.2f}")
        print(f"Минимальная скорость ветра: {original_data['Wind_Speed'].min():.2f}")

    @memory_logger
    def run(self):
        """
        Обертка-метод для вызова выполнения задания
        """
        print("\n=== ЗАДАЧА 3: Скорость ветра в самом ветреном штате ===")
        self.plot_results(self.task_job(self.aggregate_data(self.get_data())))