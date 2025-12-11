import asyncio
import logging
import os
import time

import aiofiles
import cv2
import numpy as np

import lab5.my_logging
logger = logging.getLogger(__name__)


class SaveWorker:
    """
    Воркер для асинхронного сохранения изображений через aiofiles.
    Получает задачи из save_queue и сохраняет три варианта изображения.
    """

    def __init__(self, pipeline_manager, output_dir: str, worker_name: str):
        self.pipeline_manager = pipeline_manager
        self.output_dir = output_dir
        self.worker_name = worker_name
        self.is_running = True

    async def run(self) -> None:
        """
        Основной цикл воркера сохранения.
        """
        while self.is_running or not self.pipeline_manager.save_queue.empty():
            try:
                # Ждем задачу с таймаутом чтобы проверять is_running
                try:
                    task = await asyncio.wait_for(
                        self.pipeline_manager.save_queue.get(),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.debug(f"{self.worker_name}: Timeout waiting for task")
                    break

                index, _, original_image, lib_edges, custom_edges = task
                logger.debug(f"{self.worker_name}: Saving image {index} started")
                start_time = time.time()

                try:
                    # Создаем директорию для породы (в данном случае используем индекс, но можно и породу)
                    # Так как у нас нет породы, сохраним в папке по индексу или общую папку
                    breed_dir = os.path.join(self.output_dir, f"image_{index}")
                    os.makedirs(breed_dir, exist_ok=True)

                    # Сохраняем три изображения
                    await self._save_single_image(original_image, os.path.join(breed_dir, f"{index}_original.jpg"))
                    await self._save_single_image(lib_edges, os.path.join(breed_dir, f"{index}_lib_edges.jpg"))
                    await self._save_single_image(custom_edges, os.path.join(breed_dir, f"{index}_custom_edges.jpg"))

                    self.pipeline_manager.stats.saved += 1
                    logger.debug(f"{self.worker_name}: Saving image {index} finished - {time.time() - start_time:.2f}s")

                except Exception as e:
                    self.pipeline_manager.stats.errors += 1
                    logger.error(f"{self.worker_name}: Error saving image {index}: {e}")

                finally:
                    self.pipeline_manager.save_queue.task_done()

            except Exception as e:
                logger.error(f"{self.worker_name}: Unexpected error: {e}")
                await asyncio.sleep(0.1)

    async def _save_single_image(self, image: np.ndarray, file_path: str) -> None:
        """
        Сохраняет одно изображение в файл асинхронно.
        """
        try:
            # Кодируем изображение в JPEG
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(encoded_image.tobytes())
            else:
                logger.error(f"Failed to encode image for {file_path}")
        except Exception as e:
            logger.error(f"Save error for {file_path}: {e}")

    def stop(self) -> None:
        """
        Останавливает воркер.
        """
        self.is_running = False
