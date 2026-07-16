from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = PACKAGE_DIR / "datasets"
DEFAULT_CKPT_DIR = PACKAGE_DIR / "checkpoints"

JOINT_NAMES = ("J1", "J2", "J3", "J4")
ACT_STATE_DIM = len(JOINT_NAMES)

JOINT_LIMITS = {
    "J1": (0.0015, 1.0),
    "J2": (-1.62316, 1.62316),
    "J3": (0.20944, 6.07375),
    "J4": (-16.7552, 16.7552)
}

DEFAULT_JOINT_TARGET = {
    "J1": 0.0015,
    "J2": 0.00,
    "J3": 1.00,
    "J4": 0.00
}

