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


@dataclass(frozen=True)
class Config:
    regions: RegionConfig
    render: RenderConfig
    svg_gt: SvgGtConfig


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
                  svg_gt=_build(SvgGtConfig, raw["svg_gt"]))
