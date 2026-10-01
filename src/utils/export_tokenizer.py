from pathlib import Path
import json
import numpy as np


def _write_f32(path: Path, values):
    values = np.asarray(values, dtype=np.float32)
    values.tofile(path)

def _write_u8(path: Path, value):
    value = np.asarray([value], dtype=np.uint8)
    value.tofile(path)

def export_tokenizer(
    output_dir,
    elec_boundaries,
    gas_boundaries,
    temp_boundaries,
    solar_boundaries,
    elec_bin_means,
    gas_bin_means,
    value_transform,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------
    # Continuous tokenisation parameters
    # --------------------------------------------

    _write_f32(
        output_dir / "elec_boundaries.bin",
        elec_boundaries,
    )

    _write_f32(
        output_dir / "gas_boundaries.bin",
        gas_boundaries,
    )

    _write_f32(
        output_dir / "temp_boundaries.bin",
        temp_boundaries,
    )

    _write_f32(
        output_dir / "solar_boundaries.bin",
        solar_boundaries,
    )

    # --------------------------------------------
    # Reverse-tokenisation information
    # --------------------------------------------

    _write_f32(
        output_dir / "elec_bin_means.bin",
        elec_bin_means,
    )

    _write_f32(
        output_dir / "gas_bin_means.bin",
        gas_bin_means,
    )

    # --------------------------------------------
    # Runtime configuration
    # --------------------------------------------

    use_log1p = 1 if value_transform == "log1p" else 0

    _write_u8(
        output_dir / "use_log1p.bin",
        use_log1p,
    )

    # --------------------------------------------
    # Metadata
    # --------------------------------------------

    metadata = {
        "version": 1,

        "features": [
            {
                "name": "electricity",
                "num_boundaries": len(elec_boundaries),
                "vocab_size": len(elec_boundaries) + 3,
            },
            {
                "name": "gas",
                "num_boundaries": len(gas_boundaries),
                "vocab_size": len(gas_boundaries) + 4,
            },
            {
                "name": "temperature",
                "num_boundaries": len(temp_boundaries),
                "vocab_size": len(temp_boundaries) + 3,
            },
            {
                "name": "solar",
                "num_boundaries": len(solar_boundaries),
                "vocab_size": len(solar_boundaries) + 3,
            },
            {
                "name": "half_hour",
                "vocab_size": 49,
            },
            {
                "name": "day_of_week",
                "vocab_size": 8,
            },
            {
                "name": "month",
                "vocab_size": 13,
            },
        ],

        "value_transform": value_transform,
    }

    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
