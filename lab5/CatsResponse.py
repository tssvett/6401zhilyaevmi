from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Breed:
    id: str
    name: str
    temperament: Optional[str] = None
    origin: Optional[str] = None


@dataclass
class CatImageDTO:
    id: str
    url: str
    width: int
    height: int
    breeds: List[Breed]


@dataclass
class CatsResponse:
    images: List[CatImageDTO]
    count: int
