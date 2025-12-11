import asyncio
import unittest
from unittest.mock import Mock, AsyncMock

import cv2
import numpy as np

from lab5.workers import DownloadWorker


class TestDownloadWorker(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.pipeline_manager = Mock()
        self.pipeline_manager.download_queue = asyncio.Queue()
        self.pipeline_manager.process_queue = asyncio.Queue()
        self.pipeline_manager.stats = Mock()
        self.pipeline_manager.stats.downloaded = 0
        self.pipeline_manager.stats.errors = 0

        self.session = Mock()
        self.worker = DownloadWorker(self.pipeline_manager, self.session, "TestWorker")

    async def test_download_success(self):
        print("Тест: Успешная загрузка изображения по URL")

        test_url = "http://test.com/image.jpg"
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=cv2.imencode('.jpg', test_image)[1].tobytes())
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        self.session.get = Mock(return_value=mock_context_manager)

        result = await self.worker._download_single_image(test_url)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, test_image.shape)
        self.session.get.assert_called_once_with(test_url, timeout=10)

    async def test_download_http_error(self):
        print("Тест: Загрузка изображения с HTTP ошибкой (404)")

        test_url = "http://test.com/image.jpg"
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.read = AsyncMock(return_value=b'')
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        self.session.get = Mock(return_value=mock_context_manager)

        result = await self.worker._download_single_image(test_url)

        self.assertIsNone(result)
        self.session.get.assert_called_once_with(test_url, timeout=10)

    async def test_download_timeout(self):
        print("Тест: Таймаут при загрузке изображения")

        test_url = "http://test.com/image.jpg"
        self.session.get = Mock(side_effect=asyncio.TimeoutError("Timeout"))

        result = await self.worker._download_single_image(test_url)

        self.assertIsNone(result)

    async def test_download_decode_error(self):
        print("Тест: Ошибка декодирования загруженного изображения")

        test_url = "http://test.com/image.jpg"
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b'invalid_image_data')
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        self.session.get = Mock(return_value=mock_context_manager)

        result = await self.worker._download_single_image(test_url)

        self.assertIsNone(result)
        self.session.get.assert_called_once_with(test_url, timeout=10)

    async def test_worker_run_with_valid_task(self):
        print("Тест: Запуск воркера с одной задачей в очереди")

        test_url = "http://test.com/image.jpg"
        await self.pipeline_manager.download_queue.put((0, test_url))
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=cv2.imencode('.jpg', test_image)[1].tobytes())
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        self.session.get = Mock(return_value=mock_context_manager)
        self.worker.is_running = True

        task = asyncio.create_task(self.worker.run())
        await asyncio.sleep(0.1)
        self.worker.stop()
        await task

        self.assertEqual(self.pipeline_manager.stats.downloaded, 1)
        self.assertTrue(self.pipeline_manager.download_queue.empty())

    async def test_worker_run_timeout(self):
        print("Тест: Воркер завершается по таймауту при пустой очереди")

        self.worker.is_running = True

        task = asyncio.create_task(self.worker.run())
        await asyncio.sleep(0.1)
        self.worker.stop()
        await task

        self.assertEqual(self.pipeline_manager.stats.downloaded, 0)
        self.assertEqual(self.pipeline_manager.stats.errors, 0)

    async def test_worker_run_exception_handling(self):
        print("Тест: Обработка исключений в воркере при ошибке сети")

        test_url = "http://test.com/image.jpg"
        await self.pipeline_manager.download_queue.put((0, test_url))
        self.session.get = Mock(side_effect=Exception("Network error"))
        self.worker.is_running = True

        task = asyncio.create_task(self.worker.run())
        await asyncio.sleep(0.1)
        self.worker.stop()
        await task

        self.assertEqual(self.pipeline_manager.stats.downloaded, 0)
        self.assertEqual(self.pipeline_manager.stats.errors, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
