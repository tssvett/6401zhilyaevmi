from lab2.CatImage import CatImage
from lab2.CatImageProcessor import CatImageProcessor
import cv2
import os


def main():
    try:
        limit = int(input("Введите количество изображений: "))
        if limit > 100:
            print("Максимальное количество изображений за один запрос - 100. Установлено 100.")
            limit = 100

        processor = CatImageProcessor()

        api_data = processor.get_json_images(limit)

        if api_data:
            cat_images = processor.json_to_cat_images(api_data)
            cat_image: CatImage = cat_images[0]
            cat_image.save("first")
            cat_image_2: CatImage = cat_images[1]
            cat_image_2.save("second")
            blured_image = cat_image.blur(cat_image_2)
            processed_data = processor.process_images([blured_image])
            processor.save_images([blured_image], processed_data)

    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()
