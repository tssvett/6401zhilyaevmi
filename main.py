import argparse
import os
import cv2
from implementation import ImageProcessing

def main():
    parser = argparse.ArgumentParser(
        description="Обработка изображения с помощью методов ImageProcessing (OpenCV)."
    )
    parser.add_argument(
        "method",
        choices=[
            "edges",
            "corners",
            "circles"
        ],
        help="Метод обработки: edges, corners, circles"
    )
    parser.add_argument(
        "input",
        help="Путь к входному изображению"
    )
    parser.add_argument(
        "-o", "--output",
        help="Путь для сохранения результата (по умолчанию: <input>_result.png)"
    )

    args = parser.parse_args()

    # Загрузка изображения
    image = cv2.imread(args.input)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {args.input}")
        return

    processor = ImageProcessing()

    # Выбор метода
    if args.method == "edges":
        result = processor.edge_detection(image)
    elif args.method == "corners":
        result = processor.corner_detection(image)
    elif args.method == "circles":
        result = processor.circle_detection(image)
    else:
        print("Ошибка: неизвестный метод")
        return

    # Определение пути для сохранения
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_result.png"

    # Сохранение результата
    cv2.imwrite(output_path, result)
    print(f"Результат сохранён в {output_path}")

if __name__ == "__main__":
    main()