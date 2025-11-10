import os

import cv2
import numpy as np

from lab1.implementation import ImageProcessing
from lab1.implementation.custom_image_processing import CustomImageProcessing


class AsyncCatImage:
    """
    Асинхронный класс для работы с изображениями кошек.
    Содержит индекс для сохранения порядка обработки.
    """

    def __init__(self, image: np.ndarray, image_url: str, breed: str, index: int) -> None:
        """
        Инициализация объекта AsyncCatImage.

        Args:
            image: numpy-массив с изображением
            image_url: URL изображения
            breed: порода животного
            index: порядковый номер изображения
        """
        self._image: np.ndarray = image
        self._image_url: str = image_url
        self._breed: str = breed
        self._index: int = index
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

    @property
    def index(self) -> int:
        """Property для получения порядкового номера (только чтение)."""
        return self._index

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
        return f"AsyncCatImage(breed='{self._breed}', url='{self._image_url}', shape={self._image.shape}, index={self._index})"

    def __add__(self, other: 'AsyncCatImage') -> 'AsyncCatImage':
        """
        Перегрузка оператора + для сложения двух изображений.

        Args:
            other: другое изображение AsyncCatImage

        Returns:
            Новый объект AsyncCatImage с результатом сложения
        """
        if not isinstance(other, AsyncCatImage):
            raise TypeError("Можно складывать только объекты AsyncCatImage")

        if self._image.shape != other.image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")

        # Сложение изображений с ограничением значений в диапазоне [0, 255]
        result_image = np.clip(self._image.astype(float) + other.image.astype(float), 0, 255).astype(np.uint8)

        return AsyncCatImage(result_image, f"combined_{self._breed}", f"{self._breed}_plus_{other.breed}", self._index)

    def __sub__(self, other: 'AsyncCatImage') -> 'AsyncCatImage':
        """
        Перегрузка оператора - для вычитания двух изображений.

        Args:
            other: другое изображение AsyncCatImage

        Returns:
            Новый объект AsyncCatImage с результатом вычитания
        """
        if not isinstance(other, AsyncCatImage):
            raise TypeError("Можно вычитать только объекты AsyncCatImage")

        if self._image.shape != other.image.shape:
            raise ValueError("Изображения должны иметь одинаковые размеры")

        result_image = np.clip(self._image.astype(float) - other.image.astype(float), 0, 255).astype(np.uint8)

        return AsyncCatImage(result_image, f"subtracted_{self._breed}", f"{self._breed}_minus_{other.breed}",
                             self._index)

    def blur(self, other: 'AsyncCatImage') -> 'AsyncCatImage':
        """
        Размытие изображения путем усреднения с другим изображением.

        Args:
            other: другое изображение AsyncCatImage

        Returns:
            Новый объект AsyncCatImage с размытым изображением
        """
        if not isinstance(other, AsyncCatImage):
            raise TypeError("Можно использовать только объекты AsyncCatImage")

        if self._image.shape != other.image.shape:
            print(f"Предупреждение: размеры изображений не совпадают. {self._image.shape} != {other.image.shape}")
            min_height = min(self._image.shape[0], other.image.shape[0])
            min_width = min(self._image.shape[1], other.image.shape[1])
            target_shape = (min_height, min_width, 3)

            # Создаем временные объекты с измененными размерами
            resized_self = AsyncCatImage(
                self._resize_to_match(target_shape),
                f"resized_{self._image_url}",
                self._breed,
                self._index
            )
            resized_other = AsyncCatImage(
                other._resize_to_match(target_shape),
                f"resized_{other._image_url}",
                other._breed,
                other._index
            )

            summed_image = resized_self + resized_other
        else:
            summed_image = self + other

        averaged_image = (summed_image.image.astype(float) / 2).astype(np.uint8)

        return AsyncCatImage(averaged_image, f"blurred_{self._image_url}", self._breed, self._index)

    def _resize_to_match(self, target_shape: tuple) -> np.ndarray:
        """
        Изменяет размер изображения до целевого размера.

        Args:
            target_shape: целевой размер (высота, ширина, каналы)

        Returns:
            Изображение с измененным размером
        """
        return cv2.resize(self._image, (target_shape[1], target_shape[0]))

    def save(self, output_dir: str = "cat_images_async") -> None:
        """
        Сохраняет изображение в файл.

        Args:
            output_dir: директория для сохранения
        """
        safe_breed = "".join(c if c.isalnum() else "_" for c in self.breed)

        breed_folder = os.path.join(output_dir, safe_breed)
        os.makedirs(breed_folder, exist_ok=True)
        filename = f"{self.index + 1}_{safe_breed}_original.jpg"
        path = os.path.join(breed_folder, filename)

        print(f"Сохраняем в {path}")
        success = cv2.imwrite(path, self.image)
        if success:
            print(f"Успешно сохранено: {path}")
        else:
            print(f"Ошибка сохранения: {path}")
