"""
Тесты для класса CatImageProcessor.
"""
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock

import numpy as np

from lab5 import CatImage
from lab5 import CatImageProcessor
from lab5 import CatsResponse


class TestCatImageProcessor(unittest.TestCase):
    def setUp(self):
        """Настройка тестов."""
        # Замокаем CatClient._get_api_key перед созданием процессора. Это нужно чтоб мы за реальным ключем не ходили.
        with patch('lab5.CatClient.CatClient._get_api_key', return_value="test_api_key_123"):
            self.processor = CatImageProcessor()

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Очистка после тестов."""
        self.loop.close()

    def test_get_cat_images_empty_response(self):
        """Тест получения изображений с пустым ответом."""
        with patch.object(self.processor, '_cat_client') as mock_client:
            mock_response = CatsResponse(images=[], count=0)
            mock_client.get_cats.return_value = mock_response

            result = self.loop.run_until_complete(
                self.processor.get_cat_images(1)
            )

            self.assertEqual(len(result), 0)
            mock_client.get_cats.assert_called_once_with(1)

    @patch('cv2.imencode')
    def test_save_images(self, mock_imencode):
        """Тест сохранения изображений."""
        # Создаем тестовое изображение
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cat_image = CatImage(test_image, "http://test.com/cat.jpg", "TestBreed")

        # Мокаем process_edges чтобы установить lib_image и custom_image
        cat_image._lib_image = test_image[:, :, 0]  # градиентное изображение
        cat_image._custom_image = test_image[:, :, 1]  # градиентное изображение

        # Мокаем кодирование изображения
        mock_imencode.return_value = (True, Mock(tobytes=lambda: b'test_image_data'))

        # Создаем мок для aiofiles.open как асинхронного контекстного менеджера
        mock_file = AsyncMock()
        mock_file.write = AsyncMock(return_value=None)

        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_file)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch('aiofiles.open', return_value=mock_context_manager):
            # Запускаем тест
            self.loop.run_until_complete(
                self.processor.save_images([cat_image])
            )

        # Проверяем что файлы были созданы
        mock_imencode.assert_called()
        self.assertTrue(mock_file.write.called)

    def test_save_images_empty_list(self):
        """Тест сохранения пустого списка изображений."""
        # Мокаем внутренний метод, чтобы убедиться он не вызывается
        with patch.object(self.processor, '_save_single_image_async', new_callable=AsyncMock) as mock_save:
            self.loop.run_until_complete(
                self.processor.save_images([])
            )

            mock_save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
