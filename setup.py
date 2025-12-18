"""
Конфигурация пакета для установки.
"""
from setuptools import setup, find_packages

with open("lab5/README.md", "r", encoding="utf-8") as fh:
    read_me_description = fh.read()

setup(
    name="6401-gumarov-zhilyaev",
    version="1.0.0",
    author="Gumarov and Zhilyaev",
    author_email="tssvett@mail.ru",
    description="Пакет для загрузки и обработки изображений кошек",
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "aiofiles>=23.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "numba>=0.58.0",
    ],
    entry_points={
        "console_scripts": [
            "cat-processor=lab5.main:main",
        ],
    },
    include_package_data=True,
)
