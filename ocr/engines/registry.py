"""Create engines by name, with optional config overrides (strings, as given on a command line)."""
from __future__ import annotations

import dataclasses

from ocr.config import Config, load_config
from ocr.engines.base import OcrEngine

ENGINES = ("tesseract", "paddle_onnx", "paddle_openvino", "consensus")


def override(cfg, overrides: dict[str, str]):
    """Copy of a config dataclass with string overrides converted to the field types."""
    fields = {f.name for f in dataclasses.fields(cfg)}
    typed = {}
    for k, v in overrides.items():
        if k not in fields:
            raise ValueError(f"unknown option {k!r} for {type(cfg).__name__}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            if v.lower() not in ("true", "false"):
                raise ValueError(f"{k} must be true or false, got {v!r}")
            typed[k] = v.lower() == "true"
        else:
            typed[k] = type(cur)(v)
    return dataclasses.replace(cfg, **typed)


def create(name: str, overrides: dict[str, str] | None = None, config: Config | None = None) -> OcrEngine:
    cfg = config or load_config()
    overrides = overrides or {}
    if name == "tesseract":
        from ocr.engines.tesseract import TesseractEngine
        return TesseractEngine(override(cfg.tesseract, overrides))
    if name == "paddle_onnx":
        from ocr.engines.paddle_onnx import PaddleOnnxEngine
        return PaddleOnnxEngine(override(cfg.paddle, overrides))
    if name == "paddle_openvino":
        from ocr.engines.paddle_openvino import PaddleOpenVinoEngine
        return PaddleOpenVinoEngine(override(cfg.paddle, overrides))
    if name == "consensus":
        from ocr.consensus import ConsensusEngine
        return ConsensusEngine(override(cfg.consensus, overrides), cfg)
    raise ValueError(f"unknown engine {name!r}; known: {', '.join(ENGINES)}")
