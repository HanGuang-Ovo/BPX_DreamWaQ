"""Deterministic three-level test course; no terrain information enters the policy."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .config import Sim2SimConfig

# One continuous heightfield includes flat shoulders, avoiding internal collider seams.
X_MIN, X_MAX = -3.0, 11.0
Y_MIN, Y_MAX = -3.0, 3.0
SPACING = 0.025
GROUND_PREFIX = "terrain_ground_"
# Absolute amplitude limits in metres; encoding and labels derive from these.
MILD_AMPLITUDE = 0.02
MODERATE_AMPLITUDE = 0.04


def _ramp(value):
    t = np.clip(value, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def course_heights(seed: int) -> np.ndarray:
    if not (np.isfinite(MILD_AMPLITUDE) and np.isfinite(MODERATE_AMPLITUDE)
            and 0 < MILD_AMPLITUDE <= MODERATE_AMPLITUDE):
        raise ValueError("Terrain amplitudes must satisfy 0 < mild <= moderate and be finite")
    x, y = np.meshgrid(
        np.linspace(X_MIN, X_MAX, round((X_MAX - X_MIN) / SPACING) + 1),
        np.linspace(Y_MIN, Y_MAX, round((Y_MAX - Y_MIN) / SPACING) + 1),
    )
    rng = np.random.default_rng(seed)
    roughness = np.zeros_like(x)
    # Smooth random waves, wavelengths 15–30 cm; identical spatial scales in both levels.
    for _ in range(24):
        angle = rng.uniform(0, 2 * np.pi)
        wavelength = rng.uniform(0.15, 0.30)
        roughness += np.sin(2 * np.pi / wavelength * (x * np.cos(angle) + y * np.sin(angle))
                            + rng.uniform(0, 2 * np.pi))
    roughness /= np.max(np.abs(roughness))
    amplitude = (MILD_AMPLITUDE * _ramp((x - 2.0) / 0.5)
                 + (MODERATE_AMPLITUDE - MILD_AMPLITUDE) * _ramp((x - 6.0) / 0.5))
    amplitude *= _ramp((10.0 - x) / 0.5) * _ramp((2.0 - np.abs(y)) / 0.5)
    return amplitude * roughness


def region_at(x: float, y: float) -> str:
    if not (-2 <= x <= 10 and abs(y) <= 2):
        return "FLAT / RETURN PATH"
    if x <= 2:
        return "LEVEL 0 / FLAT"
    if x < 2.5 or 6 <= x < 6.5 or x > 9.5 or abs(y) > 1.5:
        return "TRANSITION"
    return "LEVEL 1 / MILD" if x < 6 else "LEVEL 2 / MODERATE"


def load_model(config: Sim2SimConfig) -> mujoco.MjModel:
    if config.terrain.kind == "flat":
        return mujoco.MjModel.from_xml_path(str(config.paths.mjcf))
    tree = ET.parse(config.paths.mjcf)
    root = tree.getroot()
    # from_xml_string has no source directory; resolve asset directories before compiling.
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for key in ("meshdir", "texturedir"):
        compiler.set(key, str((config.paths.mjcf.parent / compiler.get(key, ".")).resolve()))
    asset = root.find("asset")
    world = root.find("worldbody")
    floor = world.find(f"geom[@name='{config.floor_geom_name}']")
    if floor is None or floor.get("type") != "plane":
        raise ValueError("Three-level course requires a named world plane in the source MJCF")
    heights = course_heights(config.terrain.seed)
    # Encode metres into [0, 1] without clipping peaks or filling valleys.
    elevation_scale = 2.0 * MODERATE_AMPLITUDE
    elevation_offset = -MODERATE_AMPLITUDE
    normalized = (heights - elevation_offset) / elevation_scale
    if not np.isfinite(normalized).all() or np.any((normalized < 0) | (normalized > 1)):
        raise ValueError("Generated terrain exceeds its declared height range")
    ET.SubElement(asset, "hfield", name="three_level_course", nrow=str(heights.shape[0]),
                  ncol=str(heights.shape[1]), size=f"7 3 {elevation_scale} 0.1")
    # Replace the plane, rather than leaving a plane that would fill the negative valleys.
    ground_attributes = {key: floor.get(key) for key in
                         ("friction", "condim", "solref", "solimp", "material") if floor.get(key) is not None}
    floor.attrib.clear()
    floor.attrib.update(name=config.floor_geom_name, type="hfield", hfield="three_level_course",
                        pos=f"4 0 {elevation_offset}", group="5", **ground_attributes)
    # Flat surrounding return area (top at z=0), no overlap with the heightfield interior.
    for name, pos, size in (
        ("west", "-26.5 0 -0.1", "23.5 50 0.1"),
        ("east", "30.5 0 -0.1", "19.5 50 0.1"),
        ("south", "4 -26.5 -0.1", "7 23.5 0.1"),
        ("north", "4 26.5 -0.1", "7 23.5 0.1"),
    ):
        ET.SubElement(world, "geom", name=GROUND_PREFIX + name, type="box", pos=pos,
                      size=size, group="5", **ground_attributes)
    # Decorative zone edges are non-colliding and do not affect ground queries.
    for center, color in ((0, "0.65 0.65 0.65 1"), (4, "0.2 0.8 0.3 1"), (8, "1 0.55 0.1 1")):
        for y_edge in (-2.05, 2.05):
            ET.SubElement(world, "geom", type="box", pos=f"{center} {y_edge} 0.003",
                          size="1.98 0.025 0.002", rgba=color, contype="0", conaffinity="0", group="1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # The 100 m surrounding ground inflates auto extent to ~166 m. Directional
    # shadow coverage and camera clipping scale with extent, producing blocky
    # robot shadows and clipping close-up views. Bound rendering to the course;
    # these statistics do not change collision geometry or physics parameters.
    model.stat.center[:] = (4.0, 0.0, 0.0)
    model.stat.extent = 14.0
    model.vis.map.shadowclip = 0.75  # 10.5 m radius covers the course and shoulders.
    model.vis.map.znear = 0.001  # 1.4 cm camera near plane for robot close-ups.
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "three_level_course")
    address = model.hfield_adr[hid]
    model.hfield_data[address:address + heights.size] = normalized.ravel()
    return model


def add_course_labels(viewer, config: Sim2SimConfig) -> None:
    """Camera-facing labels; call once after the passive viewer is opened."""
    with viewer.lock():
        # MuJoCo hides visual groups 3–5 by default; group 5 contains queryable ground.
        viewer.opt.geomgroup[5] = 1
        if config.terrain.kind != "three_level":
            return
        viewer.cam.lookat[:] = (4.0, 0.0, 0.0)
        viewer.cam.distance = 15.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -55.0
        scene = viewer.user_scn
        for x, text in ((0, "LEVEL 0 | FLAT"), (4, f"LEVEL 1 | MILD | +/-{100 * MILD_AMPLITUDE:g} cm"),
                        (8, f"LEVEL 2 | MODERATE | +/-{100 * MODERATE_AMPLITUDE:g} cm")):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError("Viewer has insufficient capacity for course labels")
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                               np.array([x, 2.3, 0.45]), np.eye(3).ravel(),
                               np.array([1., 1., 1., 1.], dtype=np.float32))
            geom.label = text
            scene.ngeom += 1
