"""
Пакет для обработки изображений кошек.
"""

from .CatClient import CatClient
from .CatImage import CatImage
from .CatImageProcessor import CatImageProcessor
from .CatsResponse import CatsResponse, CatImageDTO, Breed

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
