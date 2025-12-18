"""
Клиент для работы с API кошек.
"""
import logging
import os
import time
from typing import List, Dict, Any, Optional

import aiohttp
import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from .CatsResponse import CatsResponse, CatImageDTO, Breed

logger = logging.getLogger(__name__)


class CatClient:
    def __init__(self) -> None:
        self._base_url: str = "https://api.thecatapi.com/v1/images/search"
        self._api_key: str = self._get_api_key()
        logger.debug("Инициализирован CatClient")

    @staticmethod
    def _get_api_key() -> str:
        load_dotenv("lab5/.env")
        api_key: Optional[str] = os.getenv('API_KEY')
        if not api_key:
            logger.error("API_KEY не найден. Добавьте API_KEY в файл .env")
            raise ValueError("API_KEY не найден. Добавьте API_KEY в файл .env")
        return api_key

    def get_cats(self, limit: int = 1) -> CatsResponse:
        logger.info(f"Синхронное получение {limit} изображений из API...")
        start_time = time.time()

        params: Dict[str, Any] = {
            'limit': limit,
            'has_breeds': 1,
            'api_key': self._api_key
        }

        try:
            response = requests.get(self._base_url, params=params)
            response.raise_for_status()
            json_data = response.json()

            api_time = time.time() - start_time
            logger.info(f"Получение данных из API завершено за {api_time:.2f} секунд")

            return self._parse_api_response(json_data)

        except requests.RequestException as e:
            logger.error(f"Ошибка при запросе к API: {e}")
            return CatsResponse(images=[], count=0)
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return CatsResponse(images=[], count=0)

    async def download_image_async(self, session: aiohttp.ClientSession, image_url: str) -> Optional[np.ndarray]:
        try:
            async with session.get(image_url) as response:
                response.raise_for_status()
                img_data = await response.read()

            img_array = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                logger.warning(f"Не удалось декодировать изображение с URL: {image_url}")
                return None

            logger.debug(f"Изображение загружено: {image_url}")
            return image

        except aiohttp.ClientError as e:
            logger.error(f"Ошибка при загрузке изображения {image_url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка при загрузке изображения: {e}")
            return None

    def _parse_api_response(self, api_data: List[Dict[str, Any]]) -> CatsResponse:
        cat_images: List[CatImageDTO] = []

        for item in api_data:
            try:
                breeds: List[Breed] = []
                for breed_data in item.get('breeds', []):
                    breed = Breed(
                        id=breed_data.get('id', ''),
                        name=breed_data.get('name', 'Unknown'),
                        temperament=breed_data.get('temperament'),
                        origin=breed_data.get('origin')
                    )
                    breeds.append(breed)

                cat_image_dto = CatImageDTO(
                    id=item['id'],
                    url=item['url'],
                    width=item['width'],
                    height=item['height'],
                    breeds=breeds
                )
                cat_images.append(cat_image_dto)

            except KeyError as e:
                logger.warning(f"Ошибка парсинга элемента API: отсутствует ключ {e}")
                continue

        logger.info(f"Парсинг API ответа: найдено {len(cat_images)} изображений")
        return CatsResponse(images=cat_images, count=len(cat_images))
