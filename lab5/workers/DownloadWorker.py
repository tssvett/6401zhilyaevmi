import asyncio
import logging
import time
from typing import Optional

import aiohttp
import cv2
import numpy as np

import lab5.my_logging
logger = logging.getLogger(__name__)


class DownloadWorker:
    """
    Воркер для асинхронной загрузки изображений через aiohttp.
    Получает задачи из download_queue, загружает и передает в process_queue.
    """

    def __init__(self, pipeline_manager, session: aiohttp.ClientSession, worker_name: str):
        self.pipeline_manager = pipeline_manager
        self.session = session
        self.worker_name = worker_name
        self.is_running = True

    async def run(self) -> None:
        """
        Основной цикл воркера загрузки.
        """
        while self.is_running or not self.pipeline_manager.download_queue.empty():
            try:
                # Ждем задачу с таймаутом чтобы проверять is_running
                try:
                    index, url = await asyncio.wait_for(
                        self.pipeline_manager.download_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    logger.debug(f"{self.worker_name}: Timeout waiting for task")
                    break

                logger.debug(f"{self.worker_name}: Downloading image {index} started")
                start_time = time.time()

                try:
                    image_data = await self._download_single_image(url)

                    if image_data is not None:
                        process_task = (index, url, image_data)
                        await self.pipeline_manager.process_queue.put(process_task)

                        self.pipeline_manager.stats.downloaded += 1
                        logger.debug(
                            f"{self.worker_name}: Downloading image {index} finished - {time.time() - start_time:.2f}s")
                    else:
                        self.pipeline_manager.stats.errors += 1
                        logger.error(f"{self.worker_name}: Downloading image {index} failed")

                except Exception as e:
                    self.pipeline_manager.stats.errors += 1
                    logger.error(f"{self.worker_name}: Error downloading image {index}: {e}")

                finally:
                    self.pipeline_manager.download_queue.task_done()

            except Exception as e:
                logger.error(f"{self.worker_name}: Unexpected error: {e}")
                await asyncio.sleep(0.1)

    async def _download_single_image(self, url: str) -> Optional[np.ndarray]:
        """
        Загружает одно изображение по URL и преобразует в numpy array.
        """
        try:
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.read()

                    img_array = np.frombuffer(content, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                    if image is not None:
                        return image
                    else:
                        logger.error(f"Failed to decode image from {url}")
                        return None
                else:
                    logger.error(f"HTTP {response.status} for URL: {url}")
                    return None

        except Exception as e:
            logger.error(f"Download error for {url}: {e}")
            return None

    def stop(self) -> None:
        """
        Останавливает воркер.
        """
        self.is_running = False
