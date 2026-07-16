### J5 and J6 are end-effector rotation and cutting actions during experiments set to zero
ACT_STATE_DIM = 4
JOINT_NAMES = ("J1", "J2", "J3", "J4")

JOINT_LIMITS = {
    "J1": (0.0015, 1.0),
    "J2": (-1.62316, 1.62316),
    "J3": (0.20944, 6.07375),
    "J4": (-16.7552, 16.7552)
}

DEFAULT_JOINT_TARGET = {
    "J1": 0.00,
    "J2": 0.00,
    "J3": 0.00,
    "J4": 0.00
}