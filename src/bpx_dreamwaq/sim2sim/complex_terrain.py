"""Additional mixed-course lanes, with metric heightfields and solid obstacles."""

from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from .config import COMPLEX_TERRAIN_KINDS, TerrainConfig


@dataclass(frozen=True)
class ComplexLane:
    kind: str
    y: float
    start: float = 2.0
    end: float = 10.0
    width: float = 2.0

    @property
    def label(self) -> str:
        return self.kind.replace("_", " ").upper()


def complex_lanes(config: TerrainConfig) -> tuple[ComplexLane, ...]:
    if config.kind != "mixed":
        return ()
    return tuple(ComplexLane(kind, 4.5 + 3.0 * index)
                 for index, kind in enumerate(config.complex_sections))


def _smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lane_heights(lane: ComplexLane, scale: float) -> np.ndarray:
    """Return metres, with exactly flat borders for seamless ground joins."""
    x, y = np.meshgrid(np.linspace(lane.start, lane.end, 321),
                       np.linspace(-lane.width / 2, lane.width / 2, 81))
    shoulder = _smooth((lane.width / 2 - np.abs(y)) / 0.15)
    entry_exit = _smooth((x - lane.start) / 0.5) * _smooth((lane.end - x) / 0.5)
    if lane.kind == "ramps":
        # Two metre ascents/descents, with a four metre summit plateau.
        profile = np.minimum(np.minimum(x - lane.start, lane.end - x), 2.0)
        heights = np.tan(np.deg2rad(12.0)) * profile * shoulder
    elif lane.kind == "cross_slope":
        heights = np.tan(np.deg2rad(10.0)) * (y + lane.width / 2) * shoulder * entry_exit
    elif lane.kind == "waves":
        heights = 0.04 * (1.0 - np.cos(2 * np.pi * (x - lane.start) / 0.8)) * shoulder * entry_exit
    else:
        raise ValueError(f"No heightfield for lane {lane.kind}")
    heights *= scale
    heights[[0, -1], :] = 0.0
    heights[:, [0, -1]] = 0.0
    return heights


def _box(world, name, x, y, top, width, depth, attributes, color):
    ET.SubElement(world, "geom", name=f"terrain_ground_{name}", type="box",
                  pos=f"{x} {y} {top / 2}", size=f"{width / 2} {depth / 2} {top / 2}",
                  group="5", rgba=color, **attributes)


def _carve_heightfield_ground(world, lanes, attributes):
    """Cut the north return box around fields so zero-height seams never overlap."""
    north = world.find("geom[@name='terrain_ground_north']")
    if north is None or not lanes:
        return
    world.remove(north)

    def ground(name, x0, x1, y0, y1):
        if y1 > y0:
            ET.SubElement(world, "geom", name=f"terrain_ground_north_{name}", type="box",
                          pos=f"{(x0 + x1) / 2} {(y0 + y1) / 2} -0.1",
                          size=f"{(x1 - x0) / 2} {(y1 - y0) / 2} 0.1", group="5", **attributes)

    # Original north return area: x=[-3,11], y=[3,50]. All new fields share x=[2,10].
    ground("west", -3, 2, 3, 50)
    ground("east", 10, 11, 3, 50)
    last = 3.0
    for index, lane in enumerate(sorted(lanes, key=lambda lane: lane.y)):
        ground(str(index), 2, 10, last, lane.y - lane.width / 2)
        last = lane.y + lane.width / 2
    ground("end", 2, 10, last, 50)


def add_complex_course(asset, world, config: TerrainConfig, ground_attributes):
    """Add static collision geometry; return heightfield data to upload after compilation."""
    lanes = complex_lanes(config)
    if lanes:
        material = asset.find(f"material[@name='{ground_attributes.get('material', '')}']")
        if material is not None and material.get("texture") is not None:
            # Use a consistent spatial texture scale across subdivided ground boxes.
            material.set("texuniform", "true")
            material.set("texrepeat", "1 1")
    fields = [lane for lane in lanes if lane.kind in ("ramps", "cross_slope", "waves")]
    _carve_heightfield_ground(world, fields, ground_attributes)
    attributes = {key: value for key, value in ground_attributes.items() if key != "material"}
    colors = {
        "ramps": "0.36 0.65 0.48 1", "cross_slope": "0.32 0.56 0.75 1",
        "waves": "0.48 0.45 0.72 1", "blocks": "0.64 0.50 0.35 1",
        "hurdles": "0.85 0.48 0.22 1", "stepping_stones": "0.30 0.65 0.65 1",
    }
    uploads = []
    for lane in lanes:
        scale = config.complex_scale
        color = colors[lane.kind]
        if lane in fields:
            heights = lane_heights(lane, scale)
            elevation = float(heights.max())
            name = f"complex_{lane.kind}"
            ET.SubElement(asset, "hfield", name=name, nrow=str(heights.shape[0]),
                          ncol=str(heights.shape[1]),
                          size=f"{(lane.end - lane.start) / 2} {lane.width / 2} {elevation} 0.1")
            ET.SubElement(world, "geom", name=f"terrain_ground_{name}", type="hfield", hfield=name,
                          pos=f"{(lane.start + lane.end) / 2} {lane.y} 0", group="5", rgba=color,
                          **attributes)
            uploads.append((name, heights / elevation))
        elif lane.kind == "blocks":
            # Independent seed per kind keeps existing lanes reproducible when reordered.
            rng = np.random.default_rng([config.seed, COMPLEX_TERRAIN_KINDS.index(lane.kind)])
            for col in range(20):
                for row in range(5):
                    _box(world, f"blocks_{col}_{row}", lane.start + 0.2 + col * 0.4,
                         lane.y - 0.8 + row * 0.4, rng.uniform(0.03, 0.12) * scale,
                         0.34, 0.34, attributes, color)
        elif lane.kind == "hurdles":
            for index in range(10):
                _box(world, f"hurdles_{index}", lane.start + 0.4 + index * 0.8,
                     lane.y, 0.08 * scale, 0.12, 1.8, attributes, color)
        elif lane.kind == "stepping_stones":
            for col in range(13):
                for row in range(3):
                    _box(world, f"stepping_stones_{col}_{row}", lane.start + 0.3 + col * 0.6,
                         lane.y + (row - 1) * 0.6 + (0.075 if col % 2 else -0.075),
                         0.10 * scale, 0.45, 0.45, attributes, color)
        # Non-colliding colored edges distinguish the lane without affecting rays or contacts.
        for sign in (-1, 1):
            ET.SubElement(world, "geom", type="box", pos=f"6 {lane.y + sign * 1.05} 0.003",
                          size="4 0.025 0.002", rgba=color, contype="0", conaffinity="0", group="1")
    return uploads
