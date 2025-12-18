"""
Модуль для обработки изображений кошек.
"""
import asyncio
import logging
import multiprocessing
import os
import time
from multiprocessing import Pool, current_process
from typing import List, Optional

import aiofiles
import aiohttp
import cv2
import numpy as np

from .CatClient import CatClient
from .CatImage import CatImage

logger = logging.getLogger(__name__)


class CatImageProcessor:
    def __init__(self) -> None:
        self._cat_client = CatClient()
        logger.debug("Инициализирован CatImageProcessor")

    async def get_cat_images(self, limit: int) -> List[CatImage]:
        start_time = time.time()
        logger.info(f"Начало получения {limit} изображений...")

        # Синхронное получение данных
        cats_response = self._cat_client.get_cats(limit)

        if not cats_response.images:
            logger.warning("API не вернуло изображений")
            return []

        # Асинхронная загрузка изображений
        async with aiohttp.ClientSession() as session:
            tasks = []
            for index, cat_dto in enumerate(cats_response.images):
                breed_name = cat_dto.breeds[0].name if cat_dto.breeds else "Unknown"
                tasks.append(self._download_and_create_image(
                    session, cat_dto.url, breed_name, index + 1
                ))

            cat_images = await asyncio.gather(*tasks)

        download_time = time.time() - start_time
        logger.info(f"Загрузка завершена за {download_time:.2f} секунд")

        return [img for img in cat_images if img is not None]

    async def _download_and_create_image(self, session: aiohttp.ClientSession,
                                         url: str, breed: str, index: int) -> Optional[CatImage]:
        logger.debug(f"Загрузка изображения {index} начата")
        start_time = time.time()

        image = await self._cat_client.download_image_async(session, url)

        download_time = time.time() - start_time
        logger.debug(f"Загрузка изображения {index} завершена ({download_time:.2f} секунд)")

        if image is not None:
            logger.debug(f"Создан CatImage для {breed}")
            return CatImage(image, url, breed)

        logger.warning(f"Не удалось загрузить изображение {index}")
        return None

    async def process_images(self, cat_images: List[CatImage]) -> List[CatImage]:
        """Многопроцессорная обработка изображений"""
        start_time = time.time()
        logger.info(f"Начало многопроцессорной обработки {len(cat_images)} изображений...")

        with Pool(multiprocessing.cpu_count()) as pool:
            processed_images = list(pool.map(self._process_single_image_wrapper,
                                             enumerate(cat_images, 1)))

        process_time = time.time() - start_time
        logger.info(f"Обработка завершена за {process_time:.2f} секунд")

        return processed_images

    @staticmethod
    def _process_single_image_wrapper(args) -> CatImage:
        """Wrapper для передачи индекса вместе с изображением"""
        index, cat_image = args
        return CatImageProcessor._process_single_image(cat_image, index)

    @staticmethod
    def _process_single_image(cat_image: CatImage, index: int) -> CatImage:
        """Обработка одного изображения в отдельном процессе"""
        pid = current_process().pid
        logger.debug(f"Свертка для изображения {index} начата (PID {pid})")
        start_time = time.time()

        cat_image.process_edges()

        process_time = time.time() - start_time
        logger.debug(f"Свертка для изображения {index} завершена (PID {pid}) - {process_time:.2f} секунд")

        return cat_image

    async def save_images(self, cat_images: List[CatImage], output_dir: str = "cat_images") -> None:
        """Асинхронное сохранение изображений с использованием aiofiles"""
        if not cat_images:
            logger.warning("Нет изображений для сохранения")
            return

        start_time = time.time()
        logger.info(f"Начало асинхронного сохранения {len(cat_images)} изображений...")

        os.makedirs(output_dir, exist_ok=True)

        tasks = []
        for index, cat_image in enumerate(cat_images, 1):
            safe_breed = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
            breed_dir = os.path.join(output_dir, safe_breed)
            os.makedirs(breed_dir, exist_ok=True)

            tasks.append(self._save_single_image_async(cat_image, breed_dir, safe_breed, index))

        await asyncio.gather(*tasks)

        save_time = time.time() - start_time
        logger.info(f"Сохранение завершено за {save_time:.2f} секунд. Результаты в директории: {output_dir}")

    async def _save_single_image_async(self, cat_image: CatImage, breed_dir: str,
                                       safe_breed: str, index: int) -> None:
        """Сохраняет одно изображение в трёх форматах асинхронно через корутины"""
        logger.debug(f"Сохранение изображения {index} начато")
        start_time = time.time()

        try:
            # Генерируем все варианты изображений
            edged_cat = cat_image + cat_image.lib_image

            # Создаем задачи для асинхронного сохранения каждого изображения
            save_tasks = [
                self._async_save_image(
                    os.path.join(breed_dir, f"{index}_{safe_breed}_original.jpg"),
                    cat_image.image
                ),
                self._async_save_image(
                    os.path.join(breed_dir, f"{index}_{safe_breed}_lib_edges.jpg"),
                    cat_image.lib_image
                ),
                self._async_save_image(
                    os.path.join(breed_dir, f"{index}_{safe_breed}_custom_edges.jpg"),
                    cat_image.custom_image
                ),
                self._async_save_image(
                    os.path.join(breed_dir, f"{index}_{safe_breed}_sum_edges.jpg"),
                    edged_cat.image
                )
            ]

            # Запускаем все задачи параллельно
            await asyncio.gather(*save_tasks)

            save_time = time.time() - start_time
            logger.debug(f"Сохранение изображения {index} завершено - {save_time:.2f} секунд")

        except Exception as e:
            logger.error(f"Ошибка при сохранении изображения {index}: {e}")

    async def _async_save_image(self, filepath: str, image: np.ndarray) -> None:
        """Асинхронно сохраняет одно изображение с помощью aiofiles"""
        try:
            # Кодируем изображение в байты
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError(f"Не удалось закодировать изображение: {filepath}")

            # Получаем байты из закодированного изображения
            image_bytes = encoded_image.tobytes()

            # Асинхронно записываем байты в файл
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(image_bytes)

            logger.debug(f"Изображение сохранено: {filepath}")

        except Exception as e:
            logger.error(f"Ошибка при сохранении {filepath}: {e}")
            raise
