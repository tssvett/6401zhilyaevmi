"""
Пакет для обработки изображений кошек.
"""
import sys
from .src.CatClient import CatClient
from .src.CatImage import CatImage
from .src.CatImageProcessor import CatImageProcessor
from .src.CatsResponse import CatsResponse, CatImageDTO, Breed


__version__ = "1.0.0"
__author__ = "Gumarov and Zhilyaev"
__all__ = [
    'CatImageProcessor',
    'CatClient',
    'CatImage',
    'CatsResponse',
    'CatImageDTO',
    'Breed'
]

print("Я ЗДЕСЬ")
