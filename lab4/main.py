# comparison_test.py
import asyncio
import sys
import time

# Добавляем пути для импорта

sys.path.append('.')

from async_cat_image_processor import AsyncCatImageProcessor
from lab2.CatImageProcessor import CatImageProcessor


async def test_async_version(limit: int = 5):
    """Тестирование асинхронной версии"""
    print(f"\n=== Тестирование АСИНХРОННОЙ версии ({limit} изображений) ===")

    start_time = time.time()

    processor = AsyncCatImageProcessor()
    api_data = await processor.get_json_images_async(limit)

    if api_data:
        cat_images = await processor.json_to_cat_images_async(api_data)
        processed_data = await processor.process_images_parallel(cat_images)
        await processor.save_images_async(cat_images, processed_data)

    end_time = time.time()
    async_time = end_time - start_time
    print(f"Асинхронная версия выполнена за: {async_time:.2f} секунд")

    return async_time


def test_sync_version(limit: int = 5):
    """Тестирование синхронной версии"""
    print(f"\n=== Тестирование СИНХРОННОЙ версии ({limit} изображений) ===")

    start_time = time.time()

    processor = CatImageProcessor()
    api_data = processor.get_json_images(limit)

    if api_data:
        cat_images = processor.json_to_cat_images(api_data)
        processed_data = processor.process_images(cat_images)
        processor.save_images(cat_images, processed_data)

    end_time = time.time()
    sync_time = end_time - start_time
    print(f"Синхронная версия выполнена за: {sync_time:.2f} секунд")

    return sync_time


async def main():
    """Сравнение производительности"""
    limit = 20  # Количество изображений для теста

    # Тестируем асинхронную версию
    async_time = await test_async_version(limit)

    # Тестируем синхронную версию
    sync_time = test_sync_version(limit)

    # Вывод результатов сравнения
    print("\n=== РЕЗУЛЬТАТЫ СРАВНЕНИЯ ===")
    print(f"Синхронная версия: {sync_time:.2f} секунд")
    print(f"Асинхронная версия: {async_time:.2f} секунд")
    print(f"Ускорение: {sync_time / async_time:.2f}x")

    if async_time < sync_time:
        print("Асинхронная версия БЫСТРЕЕ!")
    else:
        print("Синхронная версия быстрее (это неожиданно)")


if __name__ == "__main__":
    asyncio.run(main())
