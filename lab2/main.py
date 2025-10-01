

from lab2.CatImageProcessor import CatImageProcessor




def main():
    try:
        processor = CatImageProcessor()

        # Запрос количества изображений
        limit = int(input("Введите количество изображений: "))

        # 1. Получаем данные из API
        api_data = processor.get_images_from_api(limit)

        if api_data:
            # 2. Преобразуем в CatImage
            cat_images = processor.map_to_cat_images(api_data)

            # 3. Обрабатываем изображения (используем методы вашего CatImage)
            processed_data = processor.process_images(cat_images)

            # 4. Сохраняем результаты
            processor.save_images(cat_images, processed_data)

    except ValueError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()
