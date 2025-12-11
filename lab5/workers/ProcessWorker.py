import asyncio
import logging
import os
import time
from typing import Tuple

import numpy as np

from lab1 import CustomImageProcessing
from lab1 import ImageProcessing

import lab5.my_logging
logger = logging.getLogger(__name__)


def process_single_image_wrapper(args: Tuple[np.ndarray, int]) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Обертка для обработки одного изображения в отдельном процессе.
    Принимает и возвращает только простые, сериализуемые типы.
    """
    image, index = args

    try:
        logger.debug(f"Convolution for image {index} started (PID {os.getpid()})")
        lib_processor = ImageProcessing()
        custom_processor = CustomImageProcessing()
        lib_edges = lib_processor.edge_detection(image)
        custom_edges = custom_processor.edge_detection(image)
        logger.debug(f"Convolution for image {index} finished (PID {os.getpid()})")
        return index, lib_edges, custom_edges

    except Exception as e:
        logger.error(f"Processing error for image {index} in PID {os.getpid()}: {e}")
        # Возвращаем пустые массивы в случае ошибки
        empty_image = np.zeros_like(image)
        return index, empty_image, empty_image


class ProcessWorker:
    """
    Воркер для обработки изображений в пуле процессов.
    """

    def __init__(self, pipeline_manager, worker_name: str):
        self.pipeline_manager = pipeline_manager
        self.worker_name = worker_name
        self.is_running = True

    async def run(self) -> None:
        """
        Основной цикл воркера обработки.
        """
        while self.is_running or not self.pipeline_manager.process_queue.empty():
            try:
                try:
                    index, url, image_data = await asyncio.wait_for(
                        self.pipeline_manager.process_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    logger.debug(f"{self.worker_name}: Timeout waiting for task")
                    break

                logger.debug(f"{self.worker_name}: Convolution for image {index} started")
                start_time = time.time()

                try:
                    processed_data = await asyncio.get_event_loop().run_in_executor(
                        self.pipeline_manager.process_executor,
                        process_single_image_wrapper,
                        (image_data, index)
                    )

                    if processed_data is not None:
                        result_index, lib_edges, custom_edges = processed_data
                        save_task = (result_index, url, image_data, lib_edges, custom_edges)
                        await self.pipeline_manager.save_queue.put(save_task)

                        self.pipeline_manager.stats.processed += 1
                        logger.debug(
                            f"{self.worker_name}: Convolution for image {index} finished - {time.time() - start_time:.2f}s")
                    else:
                        self.pipeline_manager.stats.errors += 1
                        logger.error(f"{self.worker_name}: Convolution for image {index} failed")

                except Exception as e:
                    self.pipeline_manager.stats.errors += 1
                    logger.error(f"{self.worker_name}: Error processing image {index}: {e}")

                finally:
                    self.pipeline_manager.process_queue.task_done()

            except Exception as e:
                logger.error(f"{self.worker_name}: Unexpected error: {e}")
                await asyncio.sleep(0.1)

    def stop(self) -> None:
        self.is_running = False
