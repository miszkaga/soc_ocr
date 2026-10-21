"""Common interface for OCR engines, so engines can be swapped and compared (decision 4)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels


@dataclass
class OcrWord:
    text: str
    bbox_px: Box
    conf: float  # word confidence, 0..1
    char_conf: list[float]  # one value per character of `text`, 0..1
    # Set by consensus: "accepted" | "uncertain", why, and each reading that voted.
    status: str | None = None
    reason: str | None = None
    votes: dict[str, str] | None = None


class OcrEngine(ABC):
    name: str

    @abstractmethod
    def versions(self) -> dict[str, str]:
        """Engine and model versions, recorded in every report."""

    @abstractmethod
    def detect(self, img: Image.Image) -> list[Box]:
        """Word boxes found in the image."""

    @abstractmethod
    def recognize(self, img: Image.Image, boxes: list[Box]) -> list[OcrWord]:
        """Read the given boxes (e.g. another engine's detections, for consensus)."""

    def read(self, img: Image.Image) -> list[OcrWord]:
        """Detect and recognize. Engines that do both in one pass override this."""
        return self.recognize(img, self.detect(img))
