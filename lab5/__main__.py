#!/usr/bin/env python3
"""
Главный модуль для обработки изображений кошек.
"""
import argparse
import asyncio

import time
import sys

sys.path.append("D:\\chromedriver\\6401zhilyaevmi")
__package__ = "lab5"
print("Я МЕТКА" + __package__)
# Используем относительные импорты

from .src.CatImageProcessor import CatImageProcessor
from .src.logging_config import setup_logging, add_logging_args



def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Загрузка и обработка изображений кошек'
    )
    parser.add_argument('-l', '--limit', type=int, default=10,
                        help='Количество изображений для загрузки (макс 100)')

    # Добавляем аргументы для логирования
    add_logging_args(parser)

    return parser


async def async_main(args):
    """Асинхронная основная функция."""
    # Настраиваем логирование с аргументами
    logger = setup_logging(log_file=args.log_file, log_dir=args.log_dir)

    try:
        logger.info("Запуск обработки изображений кошек")
        start_time = time.time()

        limit = args.limit
        if limit > 100:
            logger.warning("Максимальное количество изображений - 100. Установлено 100.")
            limit = 100

        logger.debug(f"Запрошено изображений: {limit}")
        processor = CatImageProcessor()

        # Синхронное получение JSON с данными изображений
        logger.info("Получение данных изображений из API...")
        cat_data = await processor.get_cat_images(limit)

        if not cat_data:
            logger.error("Не удалось получить изображения кошек.")
            return

        logger.info(f"Получено {len(cat_data)} изображений")

        # Многопроцессорная обработка изображений
        logger.info("Начало обработки изображений...")
        processed_images = await processor.process_images(cat_data)

        # Асинхронное сохранение изображений
        logger.info("Сохранение изображений...")
        await processor.save_images(processed_images)

        total_time = time.time() - start_time
        logger.info(f"Успешно обработано и сохранено {len(processed_images)} изображений кошек!")
        logger.info(f"Общее время выполнения: {total_time:.2f} секунд")

    except ValueError as e:
        logger.error(f"Ошибка ввода: {e}")
    except Exception as e:
        logger.exception(f"Неожиданная ошибка: {e}")


def main():
    """Синхронная точка входа для консольного скрипта."""
    parser = parse_args()
    args = parser.parse_args()

    # Запускаем асинхронную функцию
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
