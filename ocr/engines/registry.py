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
    """Engine by name. Overrides apply to the engine's own section; "section.key" overrides
    apply to any section of the config (e.g. paddle.refine_boxes=false for the consensus)."""
    cfg = config or load_config()
    overrides = dict(overrides or {})
    by_section: dict[str, dict[str, str]] = {}
    for k in [k for k in overrides if "." in k]:
        section, key = k.split(".", 1)
        by_section.setdefault(section, {})[key] = overrides.pop(k)
    for section, values in by_section.items():
        if not hasattr(cfg, section):
            raise ValueError(f"unknown config section {section!r}")
        cfg = dataclasses.replace(cfg, **{section: override(getattr(cfg, section), values)})
    if name == "tesseract":
        from ocr.engines.tesseract import TesseractEngine
        return TesseractEngine(override(cfg.tesseract, overrides))
    if name == "paddle_onnx":
        from ocr.engines.paddle_onnx import PaddleOnnxEngine
        return PaddleOnnxEngine(override(cfg.paddle, overrides), cfg.box_refine)
    if name == "paddle_openvino":
        from ocr.engines.paddle_openvino import PaddleOpenVinoEngine
        return PaddleOpenVinoEngine(override(cfg.paddle, overrides), cfg.box_refine)
    if name == "consensus":
        from ocr.consensus import ConsensusEngine
        return ConsensusEngine(override(cfg.consensus, overrides), cfg)
    raise ValueError(f"unknown engine {name!r}; known: {', '.join(ENGINES)}")
