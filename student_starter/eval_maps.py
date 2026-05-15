"""Evaluation maps for the image-based racing arena."""

EVAL_MAPS = {
    "curve_a": {"map_config": {"type": "block_sequence", "config": "CrCSC"}},
    "chicane": {"map_config": {"type": "block_sequence", "config": "SCSCS"}},
    "long_straight": {"map_config": {"type": "block_sequence", "config": "SSSSS"}},
    "tight_s": {"map_config": {"type": "block_sequence", "config": "CCSCC"}},
    "oval": {"map_config": {"type": "block_sequence", "config": "SCCS"}},
}

EPISODES_PER_MAP = 4

