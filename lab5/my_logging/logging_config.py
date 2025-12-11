import logging
import sys
from pathlib import Path


def setup_logging():
    """Настройка логгера - должна вызываться ПЕРВОЙ в main.py"""

    # 1. Определяем пути
    current_dir = Path(__file__).parent
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"

    print(f"Настраиваю логирование...")
    print(f"Файл логов: {log_file.absolute()}")

    # 2. Форматтеры
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # 3. Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # 4. Очищаем старые хендлеры
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 5. Файловый хендлер
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # 6. Консольный хендлер
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # 7. Добавляем хендлеры
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # 8. Тестовое сообщение
    root_logger.info("Логгер настроен успешно!")

    return root_logger


setup_logging()
