import asyncio
import os
import time
import functools
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple

import aiofiles
import aiohttp
import cv2
import numpy as np
from dotenv import load_dotenv

from async_cat_image import AsyncCatImage


def time_logger_async(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper


def time_logger_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} выполнен за {end_time - start_time:.4f} секунд")
        return result
    return wrapper


class AsyncCatImageProcessor:
    _BASE_URL = "https://api.thecatapi.com/v1/images/search"
    _DEFAULT_OUTPUT_DIR = "cat_images_async"
    _ENV_PATH = "D:/chromedriver/6401zhilyaevmi/lab2/env/.env"

    def __init__(self) -> None:
        self._api_key: str = self._get_api_key()

    @time_logger_sync
    def _get_api_key(self) -> str:
        load_dotenv(self._ENV_PATH)
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("API_KEY не найден в файле .env")
        return api_key

    @time_logger_async
    async def download_single(self, session, data_item, index):
        print(f"Загрузка изображения {index} начата")
        url = data_item["url"]
        async with session.get(url) as img_response:
            if img_response.status == 200:
                content = await img_response.read()
                img_array = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                print(f"Загрузка изображения {index} завершена")
                return data_item, index, image
            else:
                raise Exception(f"Ошибка загрузки изображения {index}: {img_response.status}")

    @time_logger_async
    async def download_images(self, limit: int = 1) -> List[Tuple[Dict[str, Any], int, np.ndarray]]:
        print(f"Получение {limit} изображений из API...")

        params = {'limit': limit, 'has_breeds': 1, 'api_key': self._api_key}
        headers = {"x-api-key": self._api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(self._BASE_URL, params=params, headers=headers) as response:
                response.raise_for_status()
                images_data = await response.json()

            tasks = []
            for i, data in enumerate(images_data):
                tasks.append(self.download_single(session, data, i))

            results = await asyncio.gather(*tasks)
            return results

    @staticmethod
    @time_logger_sync
    def process_single_image(args: Tuple[Dict[str, Any], int, np.ndarray]) -> Tuple[int, np.ndarray, np.ndarray]:
        data, index, image_array = args
        breeds = data.get("breeds", [])
        breed_name = breeds[0]["name"] if breeds else "Неизвестно"

        pid = os.getpid()
        print(f"Свертка для изображения {index} начата (PID {pid})")

        cat_image = AsyncCatImage(image_array, data["url"], breed_name, index)

        lib_edges = cat_image.detect_edges_using_library()
        custom_edges = cat_image.detect_edges_using_custom_method()

        print(f"Свертка для изображения {index} завершена (PID {pid})")
        return index, lib_edges, custom_edges

    @time_logger_async
    async def process_images_parallel(self, downloaded_data: List[Tuple[Dict[str, Any], int, np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        print(f"Обработка {len(downloaded_data)} изображений в параллельных процессах...")

        images_count = len(downloaded_data)
        original_images = [image for _, _, image in downloaded_data]
        lib_edges_images = [None] * images_count
        custom_edges_images = [None] * images_count

        with ProcessPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            for args in downloaded_data:
                future = loop.run_in_executor(executor, self.process_single_image, args)
                tasks.append(future)
            results = await asyncio.gather(*tasks)

            for index, lib_edges, custom_edges in results:
                lib_edges_images[index] = lib_edges
                custom_edges_images[index] = custom_edges

        print(f"Обработка {images_count} изображений завершена")
        return {
            'original': original_images,
            'lib_edges': lib_edges_images,
            'custom_edges': custom_edges_images
        }

    @time_logger_async
    async def save_image(self, file_path: str, image: np.ndarray) -> None:
        try:
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(encoded_image.tobytes())
        except Exception as e:
            print(f"Ошибка при сохранении {file_path}: {e}")

    @time_logger_async
    async def save_images_async(self,
                                downloaded_data: List[Tuple[Dict[str, Any], int, np.ndarray]],
                                processed_data: Dict[str, List[np.ndarray]],
                                output_dir: str = _DEFAULT_OUTPUT_DIR) -> None:
        if not downloaded_data:
            return

        print(f"Сохранение {len(downloaded_data)} изображений...")
        os.makedirs(output_dir, exist_ok=True)

        for i, (data, index, _) in enumerate(downloaded_data):
            breeds = data.get("breeds", [])
            breed_name = breeds[0]["name"] if breeds else "Неизвестно"
            safe_breed = "".join(c if c.isalnum() else "_" for c in breed_name)
            breed_dir = os.path.join(output_dir, safe_breed)
            os.makedirs(breed_dir, exist_ok=True)

        tasks = []
        for i, (data, index, _) in enumerate(downloaded_data):
            breeds = data.get("breeds", [])
            breed_name = breeds[0]["name"] if breeds else "Неизвестно"
            safe_breed = "".join(c if c.isalnum() else "_" for c in breed_name)
            breed_dir = os.path.join(output_dir, safe_breed)

            original_path = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_original.jpg")
            lib_edges_path = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_lib_edges.jpg")
            custom_edges_path = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_custom_edges.jpg")

            tasks.append(self.save_image(original_path, processed_data['original'][i]))
            tasks.append(self.save_image(lib_edges_path, processed_data['lib_edges'][i]))
            tasks.append(self.save_image(custom_edges_path, processed_data['custom_edges'][i]))

        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)

        print(f"Сохранение завершено в {output_dir}")

    @time_logger_async
    async def run_async_pipeline(self, limit: int = 5) -> float:
        print("Запуск асинхронного пайплайна...")
        start_time = time.time()

        try:
            downloaded_data = await self.download_images(limit)

            if not downloaded_data:
                print("Нет данных для обработки")
                return 0

            processed_data = await self.process_images_parallel(downloaded_data)

            await self.save_images_async(downloaded_data, processed_data)

            total_time = time.time() - start_time
            print(f"Пайплайн завершен за {total_time:.2f} секунд")
            return total_time

        except Exception as e:
            print(f"Ошибка в пайплайне: {e}")
            return 0