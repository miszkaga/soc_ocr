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
class NativeTextConfig:
    enabled: bool
    word_gap_em: float
    ocr_overlap: float


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
    refine_boxes: bool
    threads: int


@dataclass(frozen=True)
class BoxRefineConfig:
    axis_snap_deg: float
    expand_across: float
    expand_along: float
    strong_ink: float
    weak_ink: float
    min_contrast: float
    text_colour_model: bool
    colour_tolerance: float
    fringe_px: int
    rule_h_len: float
    rule_v_len: float
    max_rule_width: float
    min_rule_height: float
    glyph_centre_band: float
    max_glyph_height: float
    max_mark_height: float
    max_mark_gap: float
    w_prior: float
    w_boundary_gap: float
    w_inner_gap: float
    prior_slack_pitch: float
    max_skip_between: int
    shift_check_pitch: float
    letter_gap: float
    grow_into_ink: float
    rotated_margin: float
    max_shift_pitch: float
    margin_px: float
    fallback_margin_pitch: float
    fallback_margin_height: float


@dataclass(frozen=True)
class ConsensusConfig:
    primary: str
    secondary: str
    primary_min_conf: float
    secondary_min_conf: float
    reread_scale: float
    reread_pad_px: int
    reread_min_votes: int
    reread_max_length_change: int
    reread_keep_edge_letters: bool
    secondary_edge_junk: str
    secondary_edge_tokens: bool
    secondary_edge_brackets: bool
    normalize_typography: bool
    review_min_alnum: int
    flag_ambiguous_glyphs: bool
    flag_edge_dashes: bool
    secondary_detection: bool
    secondary_max_overlap: float


@dataclass(frozen=True)
class EvalConfig:
    matching: str
    group_min_overlap: float
    group_line_gap: float
    box_ink_threshold: int
    box_full_coverage: float
    match_min_overlap: float
    detect_min_coverage: float
    calibration_bins: int
    normalize_typography: bool
    strip_edge_chars: str
    require_alnum: bool
    min_scored_chars: int
    accept_thresholds: tuple[float, ...]
    max_error_examples: int


@dataclass(frozen=True)
class Config:
    regions: RegionConfig
    render: RenderConfig
    native_text: NativeTextConfig
    svg_gt: SvgGtConfig
    tesseract: TesseractConfig
    paddle: PaddleConfig
    consensus: ConsensusConfig
    eval: EvalConfig
    box_refine: BoxRefineConfig


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
                  native_text=_build(NativeTextConfig, raw["native_text"]),
                  svg_gt=_build(SvgGtConfig, raw["svg_gt"]), tesseract=_build(TesseractConfig, raw["tesseract"]),
                  paddle=_build(PaddleConfig, raw["paddle"]), consensus=_build(ConsensusConfig, raw["consensus"]),
                  eval=_build(EvalConfig, {**raw["eval"], "accept_thresholds": tuple(raw["eval"]["accept_thresholds"])}),
                  box_refine=_build(BoxRefineConfig, raw["box_refine"]))
