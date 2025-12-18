"""
Класс для представления и обработки изображений кошек.
"""
import numpy as np
import logging

# Используем относительные импорты
from .lab1.implementation import image_processing
from .lab1.implementation import custom_image_processing
from .lab1.utils.time_measure import ensure_3_channels

logger = logging.getLogger(__name__)

class CatImage:
    def __init__(self, image: np.ndarray, image_url: str, breed: str):
        self._image = image
        self._image_url = image_url
        self._breed = breed
        self._lib_image = None
        self._custom_image = None
        self._lib_image_processor = image_processing.ImageProcessing()
        self._custom_image_processor = custom_image_processing.CustomImageProcessing()

        logger.debug(f"Создан CatImage: {breed}, shape={image.shape}")

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def lib_image(self) -> np.ndarray:
        return self._lib_image

    @property
    def custom_image(self) -> np.ndarray:
        return self._custom_image

    @property
    def image_url(self) -> str:
        return self._image_url

    @property
    def breed(self) -> str:
        return self._breed

    def process_edges(self) -> None:
        """Обработка изображения в одном процессе"""
        logger.debug(f"Обработка границ для породы: {self._breed}")
        self._lib_image = self._lib_image_processor.edge_detection(self._image)
        self._custom_image = self._custom_image_processor.edge_detection(self._image)
        logger.debug(f"Обработка границ завершена для породы: {self._breed}")

    def __add__(self, other):
        if isinstance(other, CatImage):
            other_array = other._image
        elif isinstance(other, np.ndarray):
            other_array = other
        else:
            raise TypeError(f"Неподдерживаемый тип для сложения: {type(other)}")

        if self._image.shape[:2] != other_array.shape[:2]:
            raise ValueError(f"Несовместимые размеры изображений: {self._image.shape} и {other_array.shape}")

        self_3ch = ensure_3_channels(self._image.astype(np.float32))
        other_3ch = ensure_3_channels(other_array.astype(np.float32))

        new_image_float = np.clip(self_3ch + other_3ch, 0, 255)
        new_image = new_image_float.astype(np.uint8)

        logger.debug(f"Сложение изображений: {self._breed}")
        return self.__class__(new_image, image_url=self._image_url, breed=self._breed)

    def __str__(self) -> str:
        return f"CatImage(breed='{self._breed}', url='{self._image_url}', shape={self._image.shape})"