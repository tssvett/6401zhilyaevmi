import asyncio
import os
import time
from typing import List, Dict, Any

import aiohttp
from dotenv import load_dotenv

from lab4.AsyncPipelineManager import AsyncPipelineManager


class AsyncCatImageProcessor:
    """
    Главный класс для асинхронной обработки изображений кошек.
    Координирует получение данных из API и запуск пайплайна обработки.
    """

    _BASE_URL = "https://api.thecatapi.com/v1/images/search"
    _DEFAULT_OUTPUT_DIR = "cat_images_async"
    _ENV_PATH = "D:/chromedriver/6401zhilyaevmi/lab2/env/.env"

    def __init__(self, max_download_workers: int = 5, max_process_workers: int = None, max_save_workers: int = 3):
        self.api_key = self._get_api_key()
        self.pipeline_manager = AsyncPipelineManager(
            max_download_workers=max_download_workers,
            max_process_workers=max_process_workers,
            max_save_workers=max_save_workers,
            output_dir=self._DEFAULT_OUTPUT_DIR
        )

    def _get_api_key(self) -> str:
        """
        Загружает API ключ из .env файла.
        """
        load_dotenv(self._ENV_PATH)
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("API_KEY не найден в файле .env")
        return api_key

    async def get_image_urls_from_api(self, limit: int = 5) -> List[str]:
        """
        Получает список URL изображений из API.
        """
        print(f"Получение {limit} URL изображений из API...")

        params = {'limit': limit, 'has_breeds': 1, 'api_key': self.api_key}
        headers = {"x-api-key": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(self._BASE_URL, params=params, headers=headers) as response:
                response.raise_for_status()
                images_data = await response.json()

                urls = [data["url"] for data in images_data]
                print(f"Получено {len(urls)} URL: {urls}")
                return urls

    async def run_pipeline(self, limit: int = 5) -> Dict[str, Any]:
        """
        Запускает полный пайплайн обработки изображений.
        """
        print("Запуск асинхронного пайплайна обработки изображений...")
        start_time = time.time()

        try:
            # 1. Получаем URL из API
            image_urls = await self.get_image_urls_from_api(limit)

            if not image_urls:
                print("Нет URL для обработки")
                return {"error": "No URLs received from API"}

            # 2. Инициализируем пайплайн с полученными URL
            await self.pipeline_manager.initialize_from_api(image_urls)

            # 3. Запускаем всех воркеров
            await self.pipeline_manager.start_workers()

            # 4. Ждем завершения обработки
            stats = await self.pipeline_manager.wait_for_completion()

            total_time = time.time() - start_time

            # 5. Формируем итоговую статистику
            result = {
                "total_time": total_time,
                "images_requested": limit,
                "images_processed": stats.total_images,
                "successfully_downloaded": stats.downloaded,
                "successfully_processed": stats.processed,
                "successfully_saved": stats.saved,
                "errors": stats.errors,
                "throughput": stats.total_images / total_time if total_time > 0 else 0
            }

            print("\n" + "=" * 50)
            print("ИТОГОВАЯ СТАТИСТИКА:")
            print(f"Общее время: {total_time: .2f} секунд")
            print(f"Запрошено изображений: {limit}")
            print(f"Обработано изображений: {stats.total_images}")
            print(f"Успешно скачано: {stats.downloaded}")
            print(f"Успешно обработано: {stats.processed}")
            print(f"Успешно сохранено: {stats.saved}")
            print(f"Ошибок: {stats.errors}")
            print(f"Пропускная способность: {result['throughput']: .2f} изображений/сек")
            print("=" * 50)

            return result

        except Exception as e:
            print(f"Ошибка в пайплайне: {e}")
            return {"error": str(e)}

    async def monitor_progress(self, interval: float = 2.0) -> None:
        """
        Периодически выводит статистику обработки.
        """
        while self.pipeline_manager.is_running:
            stats = self.pipeline_manager.get_current_stats()
            print(f"\n📈 Текущая статистика: "
                  f"Загружено: {stats['downloaded']}/{stats['total']}, "
                  f"Обработано: {stats['processed']}, "
                  f"Сохранено: {stats['saved']}, "
                  f"Ошибок: {stats['errors']}, "
                  f"Очереди: D[{stats['download_queue_size']}] P[{stats['process_queue_size']}]"
                  f" S[{stats['save_queue_size']}]")
            await asyncio.sleep(interval)
