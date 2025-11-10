import asyncio
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple, Final

import aiofiles
import aiohttp
import cv2
import numpy as np
from dotenv import load_dotenv

from async_cat_image import AsyncCatImage


class AsyncCatImageProcessor:
    """
    Асинхронный класс для обработки изображений кошек через API.
    С использованием aiohttp, aiofiles и многопроцессорной обработки.
    """

    _BASE_URL: Final[str] = "https://api.thecatapi.com/v1/images/search"
    _DEFAULT_OUTPUT_DIR: Final[str] = "cat_images_async"
    _ENV_PATH: Final[str] = "D:/chromedriver/6401zhilyaevmi/lab2/env/.env"

    def __init__(self) -> None:
        """
        Инициализация асинхронного процессора.
        """
        self._api_key: str = self._get_api_key()

    @staticmethod
    def _create_breed_directory(safe_breed: str, output_dir: str) -> str:
        """
        Создает безопасную директорию для породы.

        Args:
            safe_breed: название породы
            output_dir: основная директория для сохранения

        Returns:
            Путь к созданной директории
        """
        breed_dir = os.path.join(output_dir, safe_breed)
        os.makedirs(breed_dir, exist_ok=True)
        return breed_dir

    @staticmethod
    def _generate_file_paths(breed_dir: str, safe_breed: str, index: int) -> Tuple[str, str, str]:
        """
        Генерирует пути для файлов изображений.

        Args:
            breed_dir: директория породы
            safe_breed: безопасное название породы
            index: индекс изображения

        Returns:
            Кортеж путей (original, lib_edges, custom_edges)
        """
        original_path = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_original.jpg")
        lib_edges_path = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_lib_edges.jpg")
        custom_edges_path = os.path.join(breed_dir, f"{index + 1}_{safe_breed}_custom_edges.jpg")

        return original_path, lib_edges_path, custom_edges_path

    async def _build_cat_image_async(self, image_url: str, breed: str, index: int) -> AsyncCatImage | None:
        """
        Асинхронно собирает объект кошки из урла и породы.

        Args:
            breed: Порода кошки
            image_url: Ссылка на изображение кошки
            index: Порядковый номер изображения

        Returns:
            Объект AsyncCatImage или None при ошибке
        """
        try:
            print(f"Downloading image {index} started")
            image = await self.download_image_async(image_url)
            if image is None:
                print(f"Не удалось загрузить изображение с URL: {image_url}")
                return None

            cat_image = AsyncCatImage(image, image_url, breed, index)
            print(f"Downloading image {index} finished")
            print(f"Изображение кота смапплено успешно: {cat_image}")

            return cat_image

        except (KeyError, IndexError) as exception:
            print(f"Ошибка при обработке данных изображения: {exception}")
            return None

    @property
    def api_key(self) -> str:
        """Property для получения API ключа (только чтение)."""
        return self._api_key

    def _get_api_key(self) -> str:
        """
        Получает API ключ из переменных окружения.

        Returns:
            API ключ

        Raises:
            ValueError: если ключ не найден
        """
        load_dotenv(self._ENV_PATH)
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("API_KEY не найден. Добавьте API_KEY в файл .env")
        return api_key

    async def get_json_images_async(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Асинхронное получение данных из API.

        Args:
            limit: количество изображений для получения

        Returns:
            Список jsonов для изображений
        """
        print(f"Получение {limit} изображений из API...")

        params = {
            'limit': limit,
            'has_breeds': 1,
            'api_key': self._api_key
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self._BASE_URL, params=params) as response:
                    response.raise_for_status()
                    json_response = await response.json()
                    return json_response
        except aiohttp.ClientError as e:
            print(f"Ошибка при запросе к API: {e}")
            return []

    async def download_image_async(self, image_url: str) -> np.ndarray | None:
        """
        Асинхронно загружает изображение по URL.

        Args:
            image_url: URL изображения для загрузки

        Returns:
            numpy-массив с изображением или None при ошибке
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status == 200:
                        content = await response.read()
                        img_array = np.frombuffer(content, np.uint8)
                        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        return image
                    else:
                        print(f"Ошибка при загрузке изображения {image_url}: {response.status}")
                        return None

        except Exception as e:
            print(f"Ошибка при загрузке изображения {image_url}: {e}")
            return None

    async def json_to_cat_images_async(self, api_data: List[Dict[str, Any]]) -> List[AsyncCatImage]:
        """
        Асинхронно преобразует данные API в объекты AsyncCatImage.

        Args:
            api_data: данные из API

        Returns:
            Список объектов AsyncCatImage
        """
        print("Старт маппинга изображений из API в изображения котов")
        cat_images = []

        # Создаем задачи для асинхронной обработки
        tasks = []
        for index, item in enumerate(api_data):
            image_url = item['url']
            breed = item['breeds'][0]['name'] if item['breeds'] else 'Unknown'
            task = self._build_cat_image_async(image_url, breed, index)
            tasks.append(task)

        # Ждем завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обрабатываем результаты
        for result in results:
            if isinstance(result, Exception):
                print(f"Ошибка при создании AsyncCatImage: {result}")
            elif result is not None:
                cat_images.append(result)

        print(f"Успешно создано {len(cat_images)} объектов котов")
        return cat_images

    @staticmethod
    def _process_single_image(cat_image: AsyncCatImage) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Обрабатывает одно изображение в отдельном процессе.

        Args:
            cat_image: объект AsyncCatImage для обработки

        Returns:
            Кортеж (index, lib_edges, custom_edges)
        """
        pid = os.getpid()
        print(f"Convolution for image {cat_image.index} started (PID {pid})")

        lib_edges = cat_image.detect_edges_using_library()
        custom_edges = cat_image.detect_edges_using_custom_method()

        print(f"Convolution for image {cat_image.index} finished (PID {pid})")
        return cat_image.index, lib_edges, custom_edges

    async def process_images_parallel(self, cat_images: List[AsyncCatImage]) -> Dict[str, List[np.ndarray]]:
        """
        Обрабатывает изображения в параллельных процессах.

        Args:
            cat_images: список объектов AsyncCatImage для обработки

        Returns:
            Словарь с тремя списками изображений
        """
        cat_images_number = len(cat_images)
        print(f"Обработка {cat_images_number} изображений в параллельных процессах...")

        # Создаем списки для результатов
        original_images = [cat_image.image.copy() for cat_image in cat_images]
        lib_edges_images = [None] * cat_images_number
        custom_edges_images = [None] * cat_images_number

        # Используем ProcessPoolExecutor для параллельной обработки
        with ProcessPoolExecutor() as executor:
            # Запускаем обработку в процессах
            loop = asyncio.get_event_loop()
            futures = [
                loop.run_in_executor(executor, self._process_single_image, cat_image)
                for cat_image in cat_images
            ]

            # Ждем завершения всех процессов
            results = await asyncio.gather(*futures)

            # Распределяем результаты по соответствующим индексам
            for index, lib_edges, custom_edges in results:
                lib_edges_images[index] = lib_edges
                custom_edges_images[index] = custom_edges

        print(f"Обработка {cat_images_number} завершена успешно")
        return {
            'original': original_images,
            'lib_edges': lib_edges_images,
            'custom_edges': custom_edges_images
        }

    async def save_images_async(self,
                                cat_images: List[AsyncCatImage],
                                processed_data: Dict[str, List[np.ndarray]],
                                output_dir: str = _DEFAULT_OUTPUT_DIR) -> None:
        """
        Асинхронно сохраняет изображения в файлы.

        Args:
            cat_images: список объектов AsyncCatImage (для получения метаданных)
            processed_data: словарь с обработанными изображениями
            output_dir: директория для сохранения результатов
        """
        if not cat_images:
            print("Нет изображений для сохранения")
            return

        print(f"Сохранение {len(cat_images)} изображений...")
        os.makedirs(output_dir, exist_ok=True)

        original_images = processed_data['original']
        lib_edges_images = processed_data['lib_edges']
        custom_edges_images = processed_data['custom_edges']

        # Создаем задачи для асинхронного сохранения
        tasks = []
        for index, cat_image in enumerate(cat_images):
            safe_breed = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
            breed_dir = self._create_breed_directory(safe_breed, output_dir)
            original_path, lib_edges_path, custom_edges_path = self._generate_file_paths(
                breed_dir, safe_breed, index
            )

            # Добавляем задачи сохранения
            tasks.append(self._save_single_image_async(original_path, original_images[index]))
            tasks.append(self._save_single_image_async(lib_edges_path, lib_edges_images[index]))
            tasks.append(self._save_single_image_async(custom_edges_path, custom_edges_images[index]))

        # Ждем завершения всех задач сохранения
        await asyncio.gather(*tasks)
        print(f"Сохранение завершено. Результаты в директории: {output_dir}")

    async def _save_single_image_async(self, file_path: str, image: np.ndarray) -> None:
        """
        Асинхронно сохраняет одно изображение.

        Args:
            file_path: путь для сохранения
            image: изображение для сохранения
        """
        try:
            # Кодируем изображение в JPEG
            success, encoded_image = cv2.imencode('.jpg', image)
            if success:
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(encoded_image.tobytes())
            else:
                print(f"Ошибка кодирования изображения: {file_path}")
        except Exception as e:
            print(f"Ошибка при сохранении изображения {file_path}: {e}")
