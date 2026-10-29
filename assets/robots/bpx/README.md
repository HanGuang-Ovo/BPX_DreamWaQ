# BPX Robot Model Assets

This directory contains open-source robot description assets for the `BPX` platform.

It is intended for developers who need model files for visualization, simulation setup, and integration research.

## Included Assets

- `bpx/urdf/bpx.urdf`: main URDF model entry
- `bpx/meshes/`: mesh assets referenced by the URDF model
- `mjcf/bpx.xml`: MuJoCo MJCF model entry
- `mjcf/meshes/`: mesh assets referenced by the MJCF model
- `usd/bpx.usd`: generated/cached USD model entry (may be absent in a fresh checkout)
- `usd/configuration/`: USD model, physics, and sensor layers
- `usd/config.yaml`: USD conversion configuration used for asset generation workflows

## Directory Layout

```text
assets/robots/bpx/
├── bpx/
│   ├── meshes/
│   └── urdf/
├── mjcf/
│   ├── meshes/
│   └── bpx.xml
└── usd/
    ├── configuration/
    ├── bpx.usd
    └── config.yaml
```

## Quick Start

### URDF

Use `bpx/urdf/bpx.urdf` as the primary entry when loading the robot in URDF-compatible tools such as RViz, planning pipelines, or custom parsers.

Make sure the relative mesh paths remain unchanged when copying or packaging the asset directory.

### MuJoCo (MJCF)

Use `mjcf/bpx.xml` as the entry file for MuJoCo-based simulation and validation workflows.

The model references meshes from the local `mjcf/meshes/` directory.

### USD

Training generates `usd/bpx.usd` from the URDF when it is missing. The converter settings are defined in
`src/bpx_dreamwaq/tasks/rough_env_cfg.py::_ensure_bpx_usd()`; `usd/config.yaml` records generated settings.
Set `BPX_FORCE_USD_CONVERSION=1` to regenerate after changing the source model or conversion settings.

Keep the `usd/configuration/` directory beside `bpx.usd`, because the main stage composes the model from those USD layers. The `usd/config.yaml` file records the URDF-to-USD conversion settings used to generate the generated assets.

## Notes

- Asset paths inside the model files are relative to their current directory layout.
- Joint limits, collision geometry, and inertial properties are already included in the published model files.
- Review compatibility and coordinate conventions in your target toolchain before integration.

## License And Usage

These robot assets are distributed under the `BSD-3-Clause` license. See `LICENSE`
for the full license text.
