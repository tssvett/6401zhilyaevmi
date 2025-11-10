from lab2.CatImage import CatImage
from lab2.CatImageProcessor import CatImageProcessor
import cv2
import os


def main():
    try:
        # Запрос количества изображений
        limit = int(input("Введите количество изображений: "))
        if limit > 100:
            print("Максимальное количество изображений за один запрос - 100. Установлено 100.")
            limit = 100

        processor = CatImageProcessor()
        api_data = processor.get_json_images(limit)
        if api_data:
            cat_images = processor.json_to_cat_images(api_data)
            processed_data = processor.process_images(cat_images)
            processor.save_images(cat_images, processed_data)

    except ValueError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()
