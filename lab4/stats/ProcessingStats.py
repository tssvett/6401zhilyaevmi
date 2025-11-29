from dataclasses import dataclass


@dataclass
class ProcessingStats:
    total_images: int = 0
    downloaded: int = 0
    processed: int = 0
    saved: int = 0
    errors: int = 0
    start_time: float = 0
    end_time: float = 0
