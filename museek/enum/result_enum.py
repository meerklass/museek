from enum import Enum, auto


class ResultEnum(Enum):
    # in-out
    DATA = auto()
    SCAN_DATA = auto()
    RECEIVERS = auto()
    OUTPUT_PATH = auto()
    OBSERVATION_DATE = auto()
    BLOCK_NAME = auto()
