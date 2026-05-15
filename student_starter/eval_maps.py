"""Evaluation maps for the 3D sky-road racing arena."""

EVAL_MAPS = {
    "sky_chicane": {"map_config": {"type": "block_sequence", "config": "SCSCS"}},
    "sky_curves": {"map_config": {"type": "block_sequence", "config": "CCSCC"}},
    "sky_slalom": {"map_config": {"type": "block_sequence", "config": "CrCSC"}},
    "sky_sprint": {"map_config": {"type": "block_sequence", "config": "SSCSS"}},
    "sky_gauntlet": {"map_config": {"type": "block_sequence", "config": "CSCSC"}},
}

EPISODES_PER_MAP = 4
