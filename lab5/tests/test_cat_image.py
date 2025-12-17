import unittest
from unittest.mock import patch

import numpy as np

from lab5 import CatImage


class TestCatImage(unittest.TestCase):
    def setUp(self):
        """Создание тестового изображения."""
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.cat_image = CatImage(self.test_image, "http://test.com", "TestBreed")

    def test_initialization(self):
        """Тест инициализации CatImage."""
        self.assertEqual(self.cat_image.breed, "TestBreed")
        self.assertEqual(self.cat_image.image_url, "http://test.com")
        self.assertEqual(self.cat_image.image.shape, (100, 100, 3))
        self.assertIsNone(self.cat_image.lib_image)
        self.assertIsNone(self.cat_image.custom_image)

    def test_process_edges(self):
        """Тест обработки границ."""
        # Мокаем обработчики изображений
        with patch.object(self.cat_image, '_lib_image_processor') as mock_lib, \
                patch.object(self.cat_image, '_custom_image_processor') as mock_custom:
            mock_lib.edge_detection.return_value = np.ones((100, 100), dtype=np.uint8)
            mock_custom.edge_detection.return_value = np.ones((100, 100), dtype=np.uint8) * 2

            self.cat_image.process_edges()

            mock_lib.edge_detection.assert_called_once_with(self.test_image)
            mock_custom.edge_detection.assert_called_once_with(self.test_image)

            self.assertIsNotNone(self.cat_image.lib_image)
            self.assertIsNotNone(self.cat_image.custom_image)

    def test_addition_with_cat_image(self):
        """Тест сложения двух CatImage."""
        other_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        other_cat = CatImage(other_image, "http://test2.com", "TestBreed2")

        result = self.cat_image + other_cat

        self.assertIsInstance(result, CatImage)
        self.assertEqual(result.breed, "TestBreed")
        self.assertEqual(result.image.shape, (100, 100, 3))

    def test_addition_with_numpy_array(self):
        """Тест сложения CatImage с numpy массивом."""
        other_array = np.ones((100, 100, 3), dtype=np.uint8) * 50

        result = self.cat_image + other_array

        self.assertIsInstance(result, CatImage)
        self.assertEqual(result.breed, "TestBreed")

        # Проверяем что значения сложились
        expected = np.clip(self.test_image.astype(np.float32) + 50, 0, 255).astype(np.uint8)
        np.testing.assert_array_almost_equal(result.image, expected)

    def test_addition_incompatible_shapes(self):
        """Тест сложения с несовместимыми размерами."""
        other_array = np.ones((50, 50, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            _ = self.cat_image + other_array

    def test_addition_invalid_type(self):
        """Тест сложения с неподдерживаемым типом."""
        with self.assertRaises(TypeError):
            _ = self.cat_image + "invalid"

    def test_string_representation(self):
        """Тест строкового представления."""
        str_repr = str(self.cat_image)
        self.assertIn("CatImage", str_repr)
        self.assertIn("TestBreed", str_repr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
