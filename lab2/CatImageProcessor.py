import os
from typing import List, Dict, Any

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

    def __init__(self):
        """
        Инициализация процессора.
        """
        self._base_url = "https://api.thecatapi.com/v1/images/search"
        self._api_key = self.get_api_key()

    @measure_time
    def get_api_key(self) -> str:
        """
        Получает API ключ из переменных окружения.

        Returns:
            API ключ

        Raises:
            ValueError: если ключ не найден
        """
        load_dotenv("D:/chromedriver/6401zhilyaevmi/lab2/env/.env")
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError(
                "API_KEY не найден. Добавьте API_KEY в файл .env"
            )
        return api_key

    @measure_time
    def get_images_from_api(self, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Получает только данные изображений из API.

        Args:
            limit: количество изображений для получения

        Returns:
            Список словарей с данными изображений
        """
        print(f"Получение {limit} изображений из API...")

        params = {
            'limit': limit,
            'has_breeds': 1,
            'api_key': self._api_key
        }

        try:
            response = requests.get(self._base_url, params=params)
            response.raise_for_status()
            json_response = response.json()
            return json_response
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к API: {e}")
            return []

    @measure_time
    def map_to_cat_images(self, api_data: List[Dict[str, Any]]) -> List[CatImage]:
        """
        Преобразует данные API в объекты CatImage.

        Args:
            api_data: данные из API

        Returns:
            Список объектов CatImage
        """
        cat_images = []

        for item in api_data:
            try:
                image_url = item['url']
                breed = item['breeds'][0]['name'] if item['breeds'] else 'Unknown'

                # Загрузка изображения
                img_response = requests.get(image_url)
                img_array = np.frombuffer(img_response.content, np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if image is not None:
                    cat_image = CatImage(image, image_url, breed)
                    cat_images.append(cat_image)
                    print(f"Создан объект CatImage: {cat_image}")
                else:
                    print(f"Не удалось декодировать изображение с URL: {image_url}")

            except (KeyError, IndexError) as e:
                print(f"Ошибка при обработке данных изображения: {e}")
                continue

        print(f"Успешно создано {len(cat_images)} объектов CatImage")
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
        print(f"Обработка {len(cat_images)} изображений...")

        original_images = []
        lib_edges_images = []
        custom_edges_images = []

        for cat_image in cat_images:
            # Сохраняем оригинал
            original_images.append(cat_image.image.copy())

            # Библиотечная обработка (используем ваш метод)
            lib_edges = cat_image.detect_edges_using_library()
            lib_edges_images.append(lib_edges)

            # Пользовательская обработка (используем ваш метод)
            custom_edges = cat_image.detect_edges_using_custom_method()
            custom_edges_images.append(custom_edges)

        print("Обработка завершена")

        return {
            'original': original_images,
            'lib_edges': lib_edges_images,
            'custom_edges': custom_edges_images
        }

    @measure_time
    def save_images(self,
                    cat_images: List[CatImage],
                    processed_data: Dict[str, List[np.ndarray]],
                    output_dir: str = "cat_images") -> None:
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

        # Создание основной директории
        os.makedirs(output_dir, exist_ok=True)

        original_images = processed_data['original']
        lib_edges_images = processed_data['lib_edges']
        custom_edges_images = processed_data['custom_edges']

        for i, cat_image in enumerate(cat_images):
            # Создание поддиректории для породы
            safe_breed = "".join(c if c.isalnum() else "_" for c in cat_image.breed)
            breed_dir = os.path.join(output_dir, safe_breed)
            os.makedirs(breed_dir, exist_ok=True)

            # Генерация имен файлов
            original_path = os.path.join(breed_dir, f"{i + 1}_{safe_breed}_original.jpg")
            lib_edges_path = os.path.join(breed_dir, f"{i + 1}_{safe_breed}_lib_edges.jpg")
            custom_edges_path = os.path.join(breed_dir, f"{i + 1}_{safe_breed}_custom_edges.jpg")

            try:
                # Сохранение исходного изображения
                cv2.imwrite(original_path, original_images[i])

                # Сохранение обработанных изображений
                cv2.imwrite(lib_edges_path, lib_edges_images[i])
                cv2.imwrite(custom_edges_path, custom_edges_images[i])

                print(f"Сохранено изображение {i + 1}: {cat_image.breed}")

            except Exception as e:
                print(f"Ошибка при сохранении изображения {i + 1}: {e}")

        print(f"Сохранение завершено. Результаты в директории: {output_dir}")
