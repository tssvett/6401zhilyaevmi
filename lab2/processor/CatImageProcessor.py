import os
from typing import List, Dict, Any, Tuple, Final

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from lab1.utils.time_measure import measure_time
from lab2.CatImage import CatImage


class CatImageProcessor:
    """
    Класс для обработки изображений кошек через API.
    Чистая архитектура с разделением ответственности.
    """

    _BASE_URL: Final[str] = "https://api.thecatapi.com/v1/images/search"
    _DEFAULT_OUTPUT_DIR: Final[str] = "../cat_images"
    _ENV_PATH: Final[str] = "/lab2/env/.env"

    def __init__(self) -> None:
        """
        Инициализация процессора.
        """
        self._api_key: str = self._get_api_key()

    @staticmethod
    @measure_time
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

    @measure_time
    def _build_cat_image(self, image_url: str, breed: str) -> CatImage | None:
        """
        Собирает объект кошки из урла и породы.

        Args:
            breed: Порода кошки
            image_url: Ссылка на изображение кошки

        Returns:
            Объект CatImage или None при ошибке
        """
        try:
            image = self.download_image(image_url)
            if image is None:
                print(f"Не удалось загрузить изображение с URL: {image_url}")
                return None

            cat_image = CatImage(image, image_url, breed)
            print(f"Изображение кота смапплено успешно: {cat_image}")

            return cat_image

        except (KeyError, IndexError) as exception:
            print(f"Ошибка при обработке данных изображения: {exception}")
            return None

    @property
    def api_key(self) -> str:
        """Property для получения API ключа (только чтение)."""
        return self._api_key

    @measure_time
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

    @measure_time
    def get_json_images(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Получение данных из API.

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
            response = requests.get(self._BASE_URL, params=params)
            response.raise_for_status()
            json_response = response.json()
            return json_response
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к API: {e}")
            return []

    @measure_time
    def download_image(self, image_url: str) -> np.ndarray | None:
        """
        Загружает изображение по URL.

        Args:
            image_url: URL изображения для загрузки

        Returns:
            numpy-массив с изображением или None при ошибке
        """
        try:
            img_response = requests.get(image_url)
            img_array = np.frombuffer(img_response.content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            return image

        except Exception as e:
            print(f"Ошибка при загрузке изображения {image_url}: {e}")
            return None

    @measure_time
    def json_to_cat_images(self, api_data: List[Dict[str, Any]]) -> List[CatImage]:
        """
        Преобразует данные API в объекты CatImage.

        Args:
            api_data: данные из API

        Returns:
            Список объектов CatImage
        """
        print("Старт маппинга изображений из API в изображения котов из ЛР2")
        cat_images = []
        api_data_images_number = len(api_data)

        for index, item in enumerate(api_data):
            image_url = item['url']
            breed = item['breeds'][0]['name'] if item['breeds'] else 'Unknown'
            cat_image = self._build_cat_image(image_url, breed)
            if cat_image is not None:
                cat_images.append(cat_image)
            print(f"Смаплено {index + 1}/{api_data_images_number} изображений")

        print(f"Успешно создано {len(cat_images)} объектов котов")
        return cat_images

    @measure_time
    def process_images(self, cat_images: List[CatImage]) -> Dict[str, List[np.ndarray]]:
        """
        Обрабатывает изображения (выделение контуров) используя методы CatImage.

        Args:
            cat_images: список объектов CatImage для обработки

        Returns:
            Словарь с тремя списками изображений:
            - 'original': исходные изображения
            - 'lib_edges': библиотечная обработка
            - 'custom_edges': пользовательская обработка
        """
        cat_images_number = len(cat_images)
        print(f"Обработка {cat_images_number} изображений...")

        original_images = []
        lib_edges_images = []
        custom_edges_images = []

        for index, cat_image in enumerate(cat_images):
            original_images.append(cat_image.image.copy())
            lib_edges_images.append(cat_image.detect_edges_using_library())
            custom_edges_images.append(cat_image.detect_edges_using_custom_method())
            print(f"Обработано {index + 1}/{cat_images_number} изображений")

        print(f"Обработка {cat_images_number} завершена успешно")
        return {
            'original': original_images,
            'lib_edges': lib_edges_images,
            'custom_edges': custom_edges_images
        }

    @measure_time
    def save_images(self,
                    cat_images: List[CatImage],
                    processed_data: Dict[str, List[np.ndarray]],
                    output_dir: str = _DEFAULT_OUTPUT_DIR) -> None:
        """
        Сохраняет изображения в файлы.

        Args:
            cat_images: список объектов CatImage (для получения метаданных)
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

        for index, cat_image in enumerate(cat_images):
            safe_breed = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
            breed_dir = self._create_breed_directory(safe_breed, output_dir)
            original_path, lib_edges_path, custom_edges_path = self._generate_file_paths(
                breed_dir, safe_breed, index
            )

            try:
                cv2.imwrite(original_path, original_images[index])
                cv2.imwrite(lib_edges_path, lib_edges_images[index])
                cv2.imwrite(custom_edges_path, custom_edges_images[index])
                print(f"Сохранено изображение {index + 1}: {cat_image.breed}")

            except Exception as e:
                print(f"Ошибка при сохранении изображения {index + 1}: {e}")

        print(f"Сохранение завершено. Результаты в директории: {output_dir}")