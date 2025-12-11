import unittest
from unittest.mock import Mock, AsyncMock, patch

import aiohttp

from lab5.processor import AsyncCatImageProcessor


class TestAsyncCatImageProcessor(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        print("Настройка тестового окружения для AsyncCatImageProcessor")

        with patch('lab5.processor.AsyncCatImageProcessor.load_dotenv'), \
                patch('os.getenv', return_value='test_api_key'):
            self.processor = AsyncCatImageProcessor(
                max_download_workers=2,
                max_process_workers=2,
                max_save_workers=1
            )

    async def test_get_api_key_success(self):
        print("Тест: Успешное получение API ключа из .env файла")

        with patch('lab5.processor.AsyncCatImageProcessor.load_dotenv'), \
                patch('os.getenv', return_value='test_api_key'):
            key = self.processor._get_api_key()

            self.assertEqual(key, 'test_api_key')

    async def test_get_api_key_failure(self):
        print("Тест: Ошибка при получении API ключа (ключ не найден)")

        with patch('lab5.processor.AsyncCatImageProcessor.load_dotenv'), \
                patch('os.getenv', return_value=None):
            with self.assertRaises(ValueError):
                self.processor._get_api_key()

    async def test_get_image_urls_from_api_success(self):
        print("Тест: Успешное получение URL изображений из API")

        test_urls = ["http://test.com/image1.jpg", "http://test.com/image2.jpg"]
        test_response_data = [
            {"url": "http://test.com/image1.jpg", "id": "1"},
            {"url": "http://test.com/image2.jpg", "id": "2"}
        ]
        mock_response = Mock()
        mock_response.status = 200
        mock_response.raise_for_status = Mock()
        mock_response.json = AsyncMock(return_value=test_response_data)
        mock_response_context_manager = AsyncMock()
        mock_response_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_response_context_manager)
        mock_session_context_manager = AsyncMock()
        mock_session_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session_context_manager):
            urls = await self.processor.get_image_urls_from_api(limit=2)

        self.assertEqual(urls, test_urls)
        mock_session.get.assert_called_once()

    async def test_get_image_urls_from_api_failure(self):
        print("Тест: Ошибка при получении URL из API (HTTP 404)")

        mock_response = Mock()
        mock_response.status = 404
        mock_response.raise_for_status = Mock(
            side_effect=aiohttp.ClientResponseError(
                request_info=Mock(),
                status=404,
                message="Not Found",
                history=None
            )
        )
        mock_response_context_manager = AsyncMock()
        mock_response_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_context_manager.__aexit__ = AsyncMock(return_value=None)
        mock_session = Mock()
        mock_session.get = Mock(return_value=mock_response_context_manager)
        mock_session_context_manager = AsyncMock()
        mock_session_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session_context_manager):
            with self.assertRaises(aiohttp.ClientResponseError):
                await self.processor.get_image_urls_from_api(limit=2)

    @patch('lab5.processor.AsyncCatImageProcessor.AsyncCatImageProcessor.get_image_urls_from_api')
    async def test_run_pipeline_success(self, mock_get_urls):
        print("Тест: Успешный запуск полного пайплайна обработки")

        test_urls = ["http://test.com/image1.jpg", "http://test.com/image2.jpg"]
        mock_get_urls.return_value = test_urls
        mock_manager = AsyncMock()
        mock_manager.initialize_from_api = AsyncMock()
        mock_manager.start_workers = AsyncMock()
        stats_mock = Mock()
        stats_mock.total_images = 2
        stats_mock.downloaded = 2
        stats_mock.processed = 2
        stats_mock.saved = 2
        stats_mock.errors = 0
        mock_manager.wait_for_completion = AsyncMock(return_value=stats_mock)
        self.processor.pipeline_manager = mock_manager

        result = await self.processor.run_pipeline(limit=2)

        mock_get_urls.assert_called_once_with(2)
        mock_manager.initialize_from_api.assert_called_once_with(test_urls)
        mock_manager.start_workers.assert_called_once()
        mock_manager.wait_for_completion.assert_called_once()
        self.assertEqual(result['images_requested'], 2)
        self.assertEqual(result['images_processed'], 2)
        self.assertEqual(result['successfully_saved'], 2)
        self.assertEqual(result['errors'], 0)

    @patch('lab5.processor.AsyncCatImageProcessor.AsyncCatImageProcessor.get_image_urls_from_api')
    async def test_run_pipeline_no_urls(self, mock_get_urls):
        print("Тест: Запуск пайплайна без URL (пустой ответ от API)")

        mock_get_urls.return_value = []

        result = await self.processor.run_pipeline(limit=2)

        self.assertIn('error', result)
        self.assertEqual(result['error'], 'No URLs received from API')

    @patch('lab5.processor.AsyncCatImageProcessor.AsyncCatImageProcessor.get_image_urls_from_api')
    async def test_run_pipeline_exception(self, mock_get_urls):
        print("Тест: Обработка исключений в пайплайне")

        mock_get_urls.side_effect = Exception("API недоступен")

        result = await self.processor.run_pipeline(limit=2)

        self.assertIn('error', result)
        self.assertIn('API недоступен', result['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
