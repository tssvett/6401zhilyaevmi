import asyncio
import sys
import time

from lab4.AsyncCatImageProcessor import AsyncCatImageProcessor

sys.path.append('.')

from lab2.CatImageProcessor import CatImageProcessor


async def test_async_version(limit: int = 5):
    print(f"\n=== Тестирование АСИНХРОННОЙ версии ({limit} изображений) ===")

    start_time = time.time()

    try:
        processor = AsyncCatImageProcessor(
            max_download_workers=3,
            max_process_workers=4,
            max_save_workers=2
        )
        result = await processor.run_pipeline(limit)

        end_time = time.time()
        async_time = end_time - start_time

        print(f"Асинхронная версия выполнена за: {async_time: .2f} секунд")
        print(f"Статистика: {result.get('successfully_saved', 0)}/{limit} успешно обработано")

        return async_time

    except Exception as e:
        print(f"Ошибка в асинхронной версии: {e}")
        return float('inf')


def test_sync_version(limit: int = 5):
    print(f"\n=== Тестирование СИНХРОННОЙ версии ({limit} изображений) ===")

    start_time = time.time()

    try:
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

    except Exception as e:
        print(f"Ошибка в синхронной версии: {e}")
        return float('inf')


async def main():
    limit = 1

    print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ: СИНХРОННАЯ vs АСИНХРОННАЯ")
    print("=" * 60)

    async_time = await test_async_version(limit)

    print("\nПауза между тестами...")
    await asyncio.sleep(2)

    sync_time = test_sync_version(limit)

    # Вывод результатов сравнения
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    print("=" * 60)
    print(f"Количество изображений: {limit}")
    print(f"Синхронная версия:  {sync_time: .2f} секунд")
    print(f"Асинхронная версия: {async_time: .2f} секунд")

    if async_time < sync_time:
        speedup = sync_time / async_time
        print(f"Ускорение: {speedup: .2f}x")
        print("Асинхронная версия БЫСТРЕЕ! Мультипроцессинг работает!")
    else:
        slowdown = async_time / sync_time
        print(f"Замедление: {slowdown: .2f}x")
        print("Синхронная версия быстрее (возможно из-за накладных расходов)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
