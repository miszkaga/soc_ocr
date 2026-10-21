"""Loads pipeline parameters from config/*.toml so that values never live in code."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "default.toml"


@dataclass(frozen=True)
class RegionConfig:
    scan_min_coverage: float
    scan_max_visible_text_objects: int
    min_image_side_pt: float
    logo_max_page_fraction: float
    path_max_page_fraction: float
    rule_max_thickness_pt: float
    separator_min_page_width_fraction: float
    cluster_gap_pt: float
    diagram_min_paths: int
    diagram_min_shaped_paths: int
    diagram_min_page_fraction: float
    label_margin_pt: float
    absorb_min_overlap: float
    max_paths_per_page: int


@dataclass(frozen=True)
class RenderConfig:
    dpi: int


@dataclass(frozen=True)
class SvgGtConfig:
    target_long_side_px: int
    min_scale: float
    max_scale: float
    min_legible_font_px: float
    min_visible_fraction: float
    min_latin_letter_fraction: float
    glue_gap_em: float
    glue_gap_no_space_em: float
    ink_threshold: int
    ink_min_pixels: int
    ink_min_fraction: float
    occluder_min_alpha: float


@dataclass(frozen=True)
class TesseractConfig:
    lang: str
    oem: int
    psm: int
    recognize_psm: int
    recognize_pad_px: int
    recognize_min_height_px: int
    recognize_max_scale: float
    rotate_min_aspect: float
    workers: int
    tessdata_dir: str
    timeout_s: int


@dataclass(frozen=True)
class PaddleConfig:
    model_root_dir: str
    ocr_version: str
    det_model_type: str
    det_lang: str
    rec_model_type: str
    rec_lang: str
    use_cls: bool
    text_score: float
    det_limit_side_len: int
    det_limit_type: str
    det_thresh: float
    det_box_thresh: float
    det_unclip_ratio: float
    max_side_len: int
    recognize_pad_px: int
    threads: int


@dataclass(frozen=True)
class ConsensusConfig:
    primary: str
    secondary: str
    primary_min_conf: float
    secondary_min_conf: float
    reread_scale: float
    reread_pad_px: int
    reread_min_votes: int
    secondary_edge_junk: str
    flag_ambiguous_glyphs: bool
    flag_edge_dashes: bool
    secondary_detection: bool
    secondary_max_overlap: float


@dataclass(frozen=True)
class EvalConfig:
    match_min_overlap: float
    detect_min_coverage: float
    calibration_bins: int
    accept_thresholds: tuple[float, ...]
    max_error_examples: int


@dataclass(frozen=True)
class Config:
    regions: RegionConfig
    render: RenderConfig
    svg_gt: SvgGtConfig
    tesseract: TesseractConfig
    paddle: PaddleConfig
    consensus: ConsensusConfig
    eval: EvalConfig


def _build(cls, table: dict):
    names = {f.name for f in fields(cls)}
    missing, unknown = names - table.keys(), table.keys() - names
    if missing or unknown:
        raise ValueError(f"{cls.__name__}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    return cls(**table)


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Config(regions=_build(RegionConfig, raw["regions"]), render=_build(RenderConfig, raw["render"]),
                  svg_gt=_build(SvgGtConfig, raw["svg_gt"]), tesseract=_build(TesseractConfig, raw["tesseract"]),
                  paddle=_build(PaddleConfig, raw["paddle"]), consensus=_build(ConsensusConfig, raw["consensus"]),
                  eval=_build(EvalConfig, {**raw["eval"], "accept_thresholds": tuple(raw["eval"]["accept_thresholds"])}))
