"""
main.py

Пример лабораторной работы по курсу "Технологии программирования на Python".

Модуль предназначен для демонстрации работы с обработкой изображений с помощью библиотеки
 OpenCV.
Реализован консольный интерфейс для применения различных методов обработки к изображению:
- обнаружение границ (edges)
- обнаружение углов (corners)
- обнаружение окружностей (circles)

Запуск:
    python main.py <метод> <путь_к_изображению> [-o путь_для_сохранения]

Аргументы:
    метод: edges | corners | circles
    путь_к_изображению: путь к входному изображению
    -o, --output: путь для сохранения результата
     (по умолчанию: <имя_входного_файла>_result.png)
    -i,  --impl: выбор реализации - lib (стандартная) или custom (пользовательская)
     (по умолчанию: lib)

Пример:
    python main.py edges input.jpg
    python main.py corners input.jpg -o corners_result.png

Автор: Жиляев Максим
"""

import argparse
import os

import cv2

from implementation import ImageProcessing
from implementation.custom_image_processing import CustomImageProcessing


def main() -> None:
    """
    Основная функция для обработки командной строки и выполнения обработки
     изображений.

    Функция парсит аргументы командной строки, загружает изображение,
    выбирает соответствующую реализацию обработчика и метод обработки,
    выполняет обработку и сохраняет результат.
    """
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV).",
    )
    parser.add_argument(
        "method",
        choices=[
            "edges",
            "corners",
            "circles",
        ],
        help="Метод обработки: edges, corners, circles",
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)",
    )
    parser.add_argument(
        "-i",
        "--impl",
        choices=["lib", "custom"],
        default="lib",
        help="Реализация: lib (стандартная) или custom (пользовательская)"
        " (по умолчанию: lib)",
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

        # Выбор реализации
    if args.impl == "lib":
        processor = ImageProcessing()
        default_dir = "results/lib_images"
    else:
        processor = CustomImageProcessing()
        default_dir = "results/custom_images"

    # Выбор метода
    if args.method == "edges":
        default_dir += "/edges"
        result = processor.edge_detection(image)
    elif args.method == "corners":
        default_dir += "/corners"
        result = processor.corner_detection(image)
    elif args.method == "circles":
        default_dir += "/circles"
        result = processor.circle_detection(image)
    else:
        print("Ошибка: неизвестный метод")
        return

    # Создаем директорию для результатов, если она не существует
    os.makedirs(default_dir, exist_ok=True)

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        filename = os.path.basename(base)
        output_path = f"{default_dir}/{filename}{ext or '.png'}"

    # Сохранение результата
    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в {output_path}")


if __name__ == "__main__":
    main()
