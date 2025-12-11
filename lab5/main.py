import asyncio
import logging
import time

from lab2 import CatImageProcessor
from lab5.processor.AsyncCatImageProcessor import AsyncCatImageProcessor

import lab5.my_logging
logger = logging.getLogger(__name__)


async def test_async_version(limit: int = 5):
    logger.info(f"=== Тестирование АСИНХРОННОЙ версии ({limit} изображений) ===")

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

        logger.info(f"Асинхронная версия выполнена за: {async_time: .2f} секунд")
        logger.info(f"Статистика: {result.get('successfully_saved', 0)}/{limit} успешно обработано")

        return async_time

    except Exception as e:
        logger.error(f"Ошибка в асинхронной версии: {e}")
        return float('inf')


def test_sync_version(limit: int = 5):
    logger.info(f"=== Тестирование СИНХРОННОЙ версии ({limit} изображений) ===")

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
        logger.info(f"Синхронная версия выполнена за: {sync_time:.2f} секунд")
        return sync_time

    except Exception as e:
        logger.error(f"Ошибка в синхронной версии: {e}")
        return float('inf')


async def main():
    limit = 1

    logger.info("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ: СИНХРОННАЯ vs АСИНХРОННАЯ")
    logger.info("=" * 60)

    async_time = await test_async_version(limit)

    logger.info("Пауза между тестами...")
    await asyncio.sleep(2)

    sync_time = test_sync_version(limit)

    # Вывод результатов сравнения
    logger.info("\n" + "=" * 60)
    logger.info("РЕЗУЛЬТАТЫ СРАВНЕНИЯ:")
    logger.info("=" * 60)
    logger.info(f"Количество изображений: {limit}")
    logger.info(f"Синхронная версия:  {sync_time: .2f} секунд")
    logger.info(f"Асинхронная версия: {async_time: .2f} секунд")

    if async_time < sync_time:
        speedup = sync_time / async_time
        logger.info(f"Ускорение: {speedup: .2f}x")
        logger.info("Асинхронная версия БЫСТРЕЕ! Мультипроцессинг работает!")
    else:
        slowdown = async_time / sync_time
        logger.info(f"Замедление: {slowdown: .2f}x")
        logger.info("Синхронная версия быстрее (возможно из-за накладных расходов)")

    logger.info("=" * 60)


if __name__ == "__main__":
    logger.info("Запуск приложения")
    asyncio.run(main())
    logger.info("Приложение завершено")
