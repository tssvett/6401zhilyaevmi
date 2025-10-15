import os

import cv2
import numpy as np

from lab1.implementation import ImageProcessing
from lab1.implementation.custom_image_processing import CustomImageProcessing


class CatImage:
    """
    Класс для работы с изображениями кошек.

    Инкапсулирует скаченное изображение и его метаданные,
    а также методы обработки изображения.
    """

    def __init__(self, image: np.ndarray, image_url: str, breed: str) -> None:
        """
        Инициализация объекта CatImage.

        Args:
            image: numpy-массив с изображением
            image_url: URL изображения
            breed: порода животного
        """
        self._image: np.ndarray = image
        self._image_url: str = image_url
        self._breed: str = breed
        self._lib_image_processor: ImageProcessing = ImageProcessing()
        self._custom_image_processor: CustomImageProcessing = CustomImageProcessing()

    @property
    def image(self) -> np.ndarray:
        """Property для получения изображения (только чтение)."""
        return self._image

    @property
    def image_url(self) -> str:
        """Property для получения URL изображения (только чтение)."""
        return self._image_url

    @property
    def breed(self) -> str:
        """Property для получения породы животного (только чтение)."""
        return self._breed

    def detect_edges_using_library(self) -> np.ndarray:
        """
        Выделение контуров с использованием библиотечного метода (OpenCV Canny).

        Returns:
            Изображение с выделенными контурами
        """
        return self._lib_image_processor.edge_detection(self._image)

    def detect_edges_using_custom_method(self) -> np.ndarray:
        """
        Выделение контуров с использованием пользовательского метода (оператор Собеля).

        Returns:
            Изображение с выделенными контурами
        """
        return self._custom_image_processor.edge_detection(self._image)

    def __str__(self) -> str:
        """Строковое представление объекта."""
        return f"CatImage(breed='{self._breed}', url='{self._image_url}', shape={self._image.shape})"

    def __add__(self, other: 'CatImage') -> 'CatImage':
        """
        Перегрузка оператора + для сложения двух изображений.

        Args:
            other: другое изображение CatImage

        Returns:
            Новый объект CatImage с результатом сложения
        """
        if not isinstance(other, CatImage):
            raise TypeError("Можно складывать только объекты CatImage")

        if self._image.shape != other.image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")

        # Сложение изображений с ограничением значений в диапазоне [0, 255]
        result_image = np.clip(self._image.astype(float) + other.image.astype(float), 0, 255).astype(np.uint8)

        return CatImage(result_image, f"combined_{self._breed}", f"{self._breed}_plus_{other.breed}")

    def __sub__(self, other: 'CatImage') -> 'CatImage':
        """
        Перегрузка оператора - для вычитания двух изображений.

        Args:
            other: другое изображение CatImage

        Returns:
            Новый объект CatImage с результатом вычитания
        """
        if not isinstance(other, CatImage):
            raise TypeError("Можно вычитать только объекты CatImage")

        if self._image.shape != other.image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")

        result_image = np.clip(self._image.astype(float) - other.image.astype(float), 0, 255).astype(np.uint8)

        return CatImage(result_image, f"subtracted_{self._breed}", f"{self._breed}_minus_{other.breed}")

    def blur(self, other: 'CatImage') -> 'CatImage':
        if not isinstance(other, CatImage):
            raise TypeError("Можно использовать только объекты CatImage")

        if self._image.shape != other.image.shape:
            print(f"Предупреждение: размеры изображений не совпадают. {self._image.shape} != {other.image.shape}")
            min_height = min(self._image.shape[0], other.image.shape[0])
            min_width = min(self._image.shape[1], other.image.shape[1])
            target_shape = (min_height, min_width, 3)

            # Создаем временные объекты с измененными размерами
            resized_self = CatImage(
                self._resize_to_match(target_shape),
                f"resized_{self._image_url}",
                self._breed
            )
            resized_other = CatImage(
                other._resize_to_match(target_shape),
                f"resized_{other._image_url}",
                other._breed
            )

            summed_image = resized_self + resized_other
        else:
            summed_image = self + other

        averaged_image = (summed_image.image.astype(float) / 2).astype(np.uint8)

        return CatImage(averaged_image, f"blurred_{self._image_url}", self._breed)

    def _resize_to_match(self, target_shape: tuple) -> np.ndarray:
        return cv2.resize(self._image, (target_shape[1], target_shape[0]))

    def save(self, index):
        safe_breed = "".join(c if c.isalnum() else "_" for c in self.breed)

        breed_folder = os.path.join("cat_images", safe_breed)
        os.makedirs(breed_folder, exist_ok=True)
        filename = f"{safe_breed}_original_{index}.jpg"
        path = os.path.join(breed_folder, filename)

        print(f"Сохраняем в {path}")
        success = cv2.imwrite(path, self.image)
        if success:
            print(f"Успешно сохранено: {path}")
        else:
            print(f"Ошибка сохранения: {path}")
