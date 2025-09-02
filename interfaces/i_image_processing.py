from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class IImageProcessing(ABC):
    """
    Интерфейс для реализации методов обработки изображений.

    Определяет набор абстрактных методов, которые должны быть реализованы
    в наследуемых классах для выполнения различных операций над изображениями.
    """

    @abstractmethod
    def _convolution(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Выполняет свёртку изображения с заданным ядром.

        Аргументы:
            image (np.ndarray): Входное изображение.
            kernel (np.ndarray): Ядро свёртки.

        Возвращает:
            np.ndarray: Результат применения свёртки к изображению.
        """
        pass

    @abstractmethod
    def _rgb_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Аргументы:
            image (np.ndarray): Входное RGB-изображение.

        Возвращает:
            np.ndarray: Изображение в оттенках серого.
        """
        pass

    @abstractmethod
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Аргументы:
            image (np.ndarray): Входное изображение.
            gamma (float): Значение гамма-коррекции.

        Возвращает:
            np.ndarray: Изображение после гамма-коррекции.
        """
        pass

    @abstractmethod
    def edge_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Аргументы:
            image (np.ndarray): Входное изображение.

        Возвращает:
            np.ndarray: Изображение с выделенными границами.
        """
        pass

    @abstractmethod
    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении.

        Аргументы:
            image (np.ndarray): Входное изображение.

        Возвращает:
            np.ndarray: Изображение с выделенными углами.
        """
        pass

    @abstractmethod
    def circle_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Аргументы:
            image (np.ndarray): Входное изображение.

        Возвращает:
            np.ndarray: Изображение с выделенными окружностями.
        """
        pass