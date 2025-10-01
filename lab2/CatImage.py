import numpy as np

from lab1.implementation import ImageProcessing
from lab1.implementation.custom_image_processing import CustomImageProcessing


class CatImage:
    """
    Класс для работы с изображениями кошек.

    Инкапсулирует скаченное изображение и его метаданные,
    а также методы обработки изображения.
    """

    def __init__(self, image: np.ndarray, image_url: str, breed: str):
        """
        Инициализация объекта CatImage.

        Args:
            image: numpy-массив с изображением
            image_url: URL изображения
            breed: порода животного
        """
        self._image = image
        self._image_url = image_url
        self._breed = breed
        self._lib_image_processor = ImageProcessing()
        self._custom_image_processor = CustomImageProcessing()

    @property
    def image(self) -> np.ndarray:
        """Возвращает изображение как numpy-массив."""
        return self._image

    @property
    def image_url(self) -> str:
        """Возвращает URL изображения."""
        return self._image_url

    @property
    def breed(self) -> str:
        """Возвращает породу животного."""
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
