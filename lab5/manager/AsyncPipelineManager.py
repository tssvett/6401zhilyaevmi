import asyncio
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Any

import aiohttp

from lab5.stats import ProcessingStats
from lab5.workers import DownloadWorker
from lab5.workers import ProcessWorker
from lab5.workers import SaveWorker

import lab5.my_logging
logger = logging.getLogger(__name__)


class AsyncPipelineManager:
    """
    Главный класс управления асинхронным пайплайном обработки изображений.
    Координирует работу воркеров через очереди.
    """

    def __init__(self, max_download_workers: int = 5, max_process_workers: int = None, max_save_workers: int = 3,
                 output_dir: str = "cat_images_async"):
        self.download_queue: asyncio.Queue = asyncio.Queue()
        self.process_queue: asyncio.Queue = asyncio.Queue()
        self.save_queue: asyncio.Queue = asyncio.Queue()

        self.max_download_workers = max_download_workers
        self.max_process_workers = max_process_workers or os.cpu_count()
        self.max_save_workers = max_save_workers
        self.output_dir = output_dir

        self.stats = ProcessingStats()
        self.is_running = False

        self.download_tasks: List[asyncio.Task] = []
        self.process_tasks: List[asyncio.Task] = []
        self.save_tasks: List[asyncio.Task] = []

        self.process_executor = ProcessPoolExecutor(max_workers=self.max_process_workers)

    async def initialize_from_api(self, api_urls: List[str]) -> None:
        """
        Инициализирует пайплайн списком URL из API.
        Фиксирует индексы и помещает задачи в очередь загрузки.
        """
        self.stats.total_images = len(api_urls)
        self.stats.start_time = time.time()

        for index, url in enumerate(api_urls):
            await self.download_queue.put((index, url))
            logger.debug(f"Добавлена задача загрузки: индекс={index}, url={url}")

    async def start_workers(self) -> None:
        """
        Запускает все воркеры пайплайна.
        """
        self.is_running = True
        logger.info(f"Запуск воркеров: загрузка={self.max_download_workers}, "
                    f"обработка={self.max_process_workers}, сохранение={self.max_save_workers}")

        for i in range(self.max_download_workers):
            task = asyncio.create_task(self._download_worker(f"DownloadWorker-{i}"))
            self.download_tasks.append(task)

        for i in range(self.max_process_workers):
            task = asyncio.create_task(self._process_worker(f"ProcessWorker-{i}"))
            self.process_tasks.append(task)

        for i in range(self.max_save_workers):
            task = asyncio.create_task(self._save_worker(f"SaveWorker-{i}"))
            self.save_tasks.append(task)

    async def _download_worker(self, worker_name: str) -> None:
        """
        Запускает DownloadWorker в отдельной задаче.
        """
        async with aiohttp.ClientSession() as session:
            worker = DownloadWorker(self, session, worker_name)
            await worker.run()

    async def _process_worker(self, worker_name: str) -> None:
        """
        Запускает ProcessWorker в отдельной задаче.
        """
        worker = ProcessWorker(self, worker_name)
        await worker.run()

    async def _save_worker(self, worker_name: str) -> None:
        """
        Запускает SaveWorker в отдельной задаче.
        """
        worker = SaveWorker(self, self.output_dir, worker_name)
        await worker.run()

    async def wait_for_completion(self) -> ProcessingStats:
        """
        Ожидает завершения всех задач и возвращает статистику.
        """
        logger.info("Ожидание завершения обработки...")

        # Ждем пока все URL будут обработаны
        await self.download_queue.join()

        # Останавливаем воркеры
        self.is_running = False

        # Ждем завершения всех задач
        await asyncio.gather(*self.download_tasks, *self.process_tasks, *self.save_tasks)

        # Закрываем ProcessPool
        self.process_executor.shutdown()

        self.stats.end_time = time.time()
        logger.info("Обработка завершена")
        return self.stats

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Возвращает текущую статистику обработки.
        """
        return {
            'total': self.stats.total_images,
            'downloaded': self.stats.downloaded,
            'processed': self.stats.processed,
            'saved': self.stats.saved,
            'errors': self.stats.errors,
            'download_queue_size': self.download_queue.qsize(),
            'process_queue_size': self.process_queue.qsize(),
            'save_queue_size': self.save_queue.qsize()
        }
