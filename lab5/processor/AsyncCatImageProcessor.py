import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import aiohttp
from dotenv import load_dotenv

from lab5.manager import AsyncPipelineManager

import lab5.my_logging
logger = logging.getLogger(__name__)


class AsyncCatImageProcessor:
    """
    Главный класс для асинхронной обработки изображений кошек.
    Координирует получение данных из API и запуск пайплайна обработки.
    """

    _BASE_URL = "https://api.thecatapi.com/v1/images/search"
    _DEFAULT_OUTPUT_DIR = "cat_images_async"
    _ENV_PATH = "D:/chromedriver/6401zhilyaevmi/lab5/processor/.env"

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
        logger.info(f"Поиск .env файла по пути: {self._ENV_PATH}")

        # Проверяем, существует ли файл
        if not Path(self._ENV_PATH).exists():
            logger.error(f"Файл .env не найден по пути: {self._ENV_PATH}")
            raise FileNotFoundError(f"Файл .env не найден по пути: {self._ENV_PATH}")

        load_dotenv(self._ENV_PATH)
        api_key = os.getenv('API_KEY')

        if not api_key:
            # Если ключ не найден, попробуем посмотреть, какие переменные загрузились
            logger.error(f"API_KEY не найден. Проверьте, что в файле {self._ENV_PATH} есть строка API_KEY=ваш_ключ")
            raise ValueError(f"API_KEY не найден в файле .env")

        logger.info("API ключ успешно загружен")
        return api_key

    async def get_image_urls_from_api(self, limit: int = 5) -> List[str]:
        """
        Получает список URL изображений из API.
        """
        logger.info(f"Получение {limit} URL изображений из API...")

        params = {'limit': limit, 'has_breeds': 1, 'api_key': self.api_key}
        headers = {"x-api-key": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(self._BASE_URL, params=params, headers=headers) as response:
                response.raise_for_status()
                images_data = await response.json()

                urls = [data["url"] for data in images_data]
                logger.info(f"Получено {len(urls)} URL: {urls}")
                return urls

    async def run_pipeline(self, limit: int = 5) -> Dict[str, Any]:
        """
        Запускает полный пайплайн обработки изображений.
        """
        logger.info("Запуск асинхронного пайплайна обработки изображений...")
        start_time = time.time()

        try:
            # 1. Получаем URL из API
            image_urls = await self.get_image_urls_from_api(limit)

            if not image_urls:
                logger.warning("Нет URL для обработки")
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

            logger.info("\n" + "=" * 50)
            logger.info("ИТОГОВАЯ СТАТИСТИКА:")
            logger.info(f"Общее время: {total_time: .2f} секунд")
            logger.info(f"Запрошено изображений: {limit}")
            logger.info(f"Обработано изображений: {stats.total_images}")
            logger.info(f"Успешно скачано: {stats.downloaded}")
            logger.info(f"Успешно обработано: {stats.processed}")
            logger.info(f"Успешно сохранено: {stats.saved}")
            logger.info(f"Ошибок: {stats.errors}")
            logger.info(f"Пропускная способность: {result['throughput']: .2f} изображений/сек")
            logger.info("=" * 50)

            return result

        except Exception as e:
            logger.error(f"Ошибка в пайплайне: {e}")
            return {"error": str(e)}
