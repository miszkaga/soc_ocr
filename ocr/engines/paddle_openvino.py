"""The same PaddleOCR models as paddle_onnx, executed by OpenVINO (Intel CPU / iGPU / NPU)."""
from __future__ import annotations

from ocr.engines.paddle_onnx import PaddleOnnxEngine


class PaddleOpenVinoEngine(PaddleOnnxEngine):
    name = "paddle_openvino"
    backend = "openvino"
