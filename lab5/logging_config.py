"""
Модуль конфигурации логирования для проекта обработки изображений кошек.
"""
import argparse
import logging
import os
import sys


def setup_logging(log_file: str = "app.log", log_dir: str = ".",
                  console_level: int = logging.INFO,
                  file_level: int = logging.DEBUG) -> logging.Logger:
    """
    Настройка корневого логгера с обработчиками для консоли и файла.

    Args:
        log_file: Имя файла для логов
        log_dir: Директория для логов
        console_level: Уровень логирования для консоли
        file_level: Уровень логирования для файла

    Returns:
        Корневой логгер
    """
    # Создаем директорию для логов если ее нет
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    # Получаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Удаляем существующие обработчики
    logger.handlers.clear()

    # Форматы для разных обработчиков
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Обработчик для консоли (краткие логи)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Обработчик для файла (подробные логи)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """
    Добавляет аргументы командной строки для настройки логирования.

    Args:
        parser: Парсер аргументов
    """
    parser.add_argument('--log-file', default='app.log',
                        help='Имя файла для логов (по умолчанию: app.log)')
    parser.add_argument('--log-dir', default='.',
                        help='Директория для логов (по умолчанию: текущая)')
