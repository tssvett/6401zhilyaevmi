"""
Модуль image_processing.py

Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

Содержит класс ImageProcessing, предоставляющий методы для обработки изображений:
- свёртка изображения с ядром
- преобразование RGB-изображения в оттенки серого
- гамма-коррекция
- обнаружение границ (оператор Кэнни)
- обнаружение углов (алгоритм Харриса)
- обнаружение окружностей (метод пока не реализован)

Модуль предназначен для учебных целей
 (лабораторная работа по курсу "Технологии программирования на Python").

 Выполнил 6401 Жиляев Максим Иванович
"""

from lab1 import interfaces

import numpy as np

from lab1.utils.time_measure import measure_time
from numba import njit

max_pixel_value = 255.0

red_coefficient = 0.299
green_coefficient = 0.587
blue_coefficient = 0.114

sobel_kernel_x = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.float32,
)

sobel_kernel_y = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.float32,
)

gaussian_kernel = (
        np.array(
            [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1],
            ],
            dtype=np.float32,
        )
        / 16.0
)


class CustomImageProcessing(interfaces.IImageProcessing):
    """
    Реализация интерфейса IImageProcessing с использованием библиотеки OpenCV.

    Предоставляет методы для обработки изображений, включая свёртку, преобразование
    в оттенки серого, гамма-коррекцию, а также обнаружение границ, углов и окружностей.

    Методы:
        _convolution(image, kernel): Выполняет свёртку изображения с ядром.
        _rgb_to_grayscale(image): Преобразует RGB-изображение в оттенки серого.
        _gamma_correction(image, gamma): Применяет гамма-коррекцию.
        edge_detection(image): Обнаруживает границы (Canny).
        corner_detection(image): Обнаруживает углы (Harris).
        circle_detection(image): Обнаруживает окружности (HoughCircles).
    """

    @measure_time
    def _convolution(
            self: "CustomImageProcessing",
            image: np.ndarray,
            kernel: np.ndarray,
    ) -> np.ndarray:
        """
        Выполняет свертку изображения

        Args:
            image (np.ndarray): Входное двухканальное изображение
            kernel (np.ndarray): Входное ядро свертки

        Returns:
            np.ndarray: Изображение, подвергнутое свертке
        """
        kernel_height, kernel_width = kernel.shape
        pad_height, pad_width = kernel_height // 2, kernel_width // 2
        padded = np.pad(
            image,
            ((pad_height, pad_height), (pad_width, pad_width)),
            mode="reflect",
        )

        return self.conv(image, padded, kernel_height, kernel_width, kernel)

    @staticmethod
    @njit
    def conv(image, padded, kernel_height, kernel_width, kernel):
        output = np.zeros_like(image)
        for rows in range(image.shape[0]):
            for cols in range(image.shape[1]):
                region = padded[rows: rows + kernel_height, cols: cols + kernel_width]
                output[rows, cols] = np.sum(region * kernel)

        return output

    def _rgb_to_grayscale(self: "CustomImageProcessing", image: np.ndarray) -> np.ndarray:
        """
        Преобразует RGB-изображение в оттенки серого.

        Args:
            image (np.ndarray): Входное RGB-изображение.

        Returns:
            np.ndarray: Одноканальное изображение в оттенках серого.
        """
        # Стандартные коэффициенты восприятия яркости человеческим глазом

        blue_channel = image[:, :, 0]  # B-канал (синий)
        green_channel = image[:, :, 1]  # G-канал (зеленый)
        red_channel = image[:, :, 2]  # R-канал (красный)
        grayscale = (
                red_coefficient * red_channel
                + green_coefficient * green_channel
                + blue_coefficient * blue_channel
        )

        # Ограничиваемся диапазоном [0, 255], тк пиксель у нас от 0 до 255
        grayscale_clipped = np.clip(grayscale, 0, max_pixel_value)
        grayscale_float32 = grayscale_clipped.astype(np.float32)

        return grayscale_float32

    def _gamma_correction(
            self: "CustomImageProcessing",
            image: np.ndarray,
            gamma: float,
    ) -> np.ndarray:
        """
        Применяет гамма-коррекцию к изображению.

        Args:
            image (np.ndarray): Входное изображение.
            gamma (float): Коэффициент гамма-коррекции (>0).

        Returns:
            np.ndarray: Изображение после гамма-коррекции.
        """
        if gamma <= 0:
            raise ValueError("Гамма значение должно быть > 0")

        normalized = image.astype(np.float32) / max_pixel_value

        # Если gamma > 1 - изображение становится темнее
        # Если gamma < 1 - изображение становится светлее

        gamma_exponent = 1 / gamma
        corrected = np.power(normalized, gamma_exponent)

        return (corrected * max_pixel_value).astype(np.uint8)

    @measure_time
    def edge_detection(self: "CustomImageProcessing", image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение границ на изображении.

        Использует оператор Кэнни для выделения границ.
        Предварительно изображение преобразуется в оттенки серого.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Returns:
            np.ndarray: Одноканальное изображение с выделенными границами.
        """
        gray = self._rgb_to_grayscale(image)

        gradient_x = self._convolution(gray, sobel_kernel_x)
        gradient_y = self._convolution(gray, sobel_kernel_y)

        # величина градиента показывает "силу границ"
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

        # тут проверяем что больше нуля чтоб на ноль не поделить случайно
        if gradient_magnitude.max() > 0:
            gradient_magnitude = (
                    gradient_magnitude / gradient_magnitude.max() * max_pixel_value
            )

        return gradient_magnitude.astype(np.uint8)

    @measure_time
    def corner_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Выполняет обнаружение углов на изображении с помощью детектора Харриса.

        Args:
            image (np.ndarray): Входное изображение.

        Returns:
            result (np.ndarray): Изображение после поиска углов
        """
        # Конфигурационные параметры
        harris_k = 0.04
        corners_amount = 1000
        gray_image = self._rgb_to_grayscale(image)
        harris_response = self._compute_harris_response(gray_image, harris_k)
        r_norm = self._normalize_harris_response(harris_response)
        corner_mask = self._find_corners_with_adaptive_threshold(r_norm, corners_amount)
        local_maxima = self._non_maximum_suppression(r_norm, corner_mask)
        result = self._visualize_corners(image, local_maxima)

        return result

    def _compute_harris_response(
            self,
            gray_image: np.ndarray,
            harris_coefficient: float,
    ) -> np.ndarray:
        """Вычисляет отклик Харриса для изображения в градациях серого."""
        gradient_x = self._convolution(gray_image, sobel_kernel_x)
        gradient_y = self._convolution(gray_image, sobel_kernel_y)

        gradient_xx = gradient_x * gradient_x
        gradient_yy = gradient_y * gradient_y
        gradient_xy = gradient_x * gradient_y

        smoothed_xx = self._convolution(gradient_xx, gaussian_kernel)
        smoothed_yy = self._convolution(gradient_yy, gaussian_kernel)
        smoothed_xy = self._convolution(gradient_xy, gaussian_kernel)

        determinant = smoothed_xx * smoothed_yy - smoothed_xy * smoothed_xy
        trace = smoothed_xx + smoothed_yy
        harris_response = determinant - harris_coefficient * trace * trace

        return harris_response

    @staticmethod
    def _normalize_harris_response(harris_response: np.ndarray) -> np.ndarray:
        """Нормализует отклик Харриса к диапазону [0, 1]."""
        min_value = np.min(harris_response)
        max_value = np.max(harris_response)

        if max_value - min_value > 0:
            return (harris_response - min_value) / (max_value - min_value)
        else:
            return np.zeros_like(harris_response)

    @staticmethod
    @njit
    def _find_adaptive_threshold(
            response_norm: np.ndarray,
            target_corners_number: int,
    ) -> float:
        """
        Получаем динамический порог основываясь на нужном числе углом
        @param response_norm: норма от отклика харриса
        @param target_corners_number: число углов которое нужно
        @return: порог, который даст необходимое число углов
        """
        flat_response_list = response_norm.flatten()
        sorted_response = np.sort(flat_response_list)[::-1]

        # Если углов много, берем значение, которое находится на target_corners_number
        # Если углов мало, берем самое маленькое значение.
        # Если углов вообще нет, берем 0.5 потому что можем

        if len(sorted_response) > target_corners_number:
            threshold = sorted_response[target_corners_number]
        else:
            threshold = sorted_response[-1] if len(sorted_response) > 0 else 0.5

        return max(0.1, min(0.9, threshold))

    def _find_corners_with_adaptive_threshold(
            self,
            normalized_response: np.ndarray,
            target_corners: int,
    ) -> np.ndarray:
        """Находит углы используя адаптивный порог."""
        threshold = self._find_adaptive_threshold(
            normalized_response,
            target_corners,
        )
        return normalized_response > threshold

    @staticmethod
    def _non_maximum_suppression(
            normalized_response: np.ndarray,
            corners_mask: np.ndarray,
    ) -> np.ndarray:
        """Применяет подавление немаксимумов для устранения дубликатов углов."""
        height, width = normalized_response.shape
        local_maxima_mask = np.zeros_like(corners_mask, dtype=bool)

        for row in range(1, height - 1):
            for col in range(1, width - 1):
                if corners_mask[row, col]:
                    # берем область 3x3 вокруг текущего пикселя
                    # это окно для сравнения силы угла с соседями
                    neighborhood = normalized_response[row - 1: row + 2, col - 1: col + 2]
                    if normalized_response[row, col] == np.max(neighborhood):
                        local_maxima_mask[row, col] = True

        return local_maxima_mask

    @staticmethod
    @njit
    def _visualize_corners(image: np.ndarray, corners_mask: np.ndarray) -> np.ndarray:
        """Визуализирует найденные углы на изображении."""
        result_image = image.copy().astype(np.uint8)
        height, width = image.shape[:2]
        corners_coordinates = np.where(corners_mask)

        if len(corners_coordinates[0]) > 0:
            for col, row in zip(corners_coordinates[0], corners_coordinates[1]):
                # Рисуем квадрат 3x3 пикселя

                y_start = max(0, col - 1)
                y_end = min(height, col + 2)
                x_start = max(0, row - 1)
                x_end = min(width, row + 2)

                result_image[y_start:y_end, x_start:x_end, 0] = 0  # B
                result_image[y_start:y_end, x_start:x_end, 1] = 0  # G
                result_image[y_start:y_end, x_start:x_end, 2] = 255  # R

        return result_image

    def circle_detection(
            self,
            image: np.ndarray,
    ) -> np.ndarray:
        """
        Выполняет обнаружение окружностей на изображении.

        Использует преобразование Хафа (cv2.HoughCircles) для поиска окружностей.
        Найденные окружности выделяются зелёным цветом, центры — красным.

        Args:
            image (np.ndarray): Входное изображение (RGB).

        Raises:
            NotImplementedError: Ошибка о не написании
        """
        raise NotImplementedError("Метод обнаружения окружностей пока не реализован.")
