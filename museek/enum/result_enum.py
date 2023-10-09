from enum import Enum, auto


class ResultEnum(Enum):
    # in-out
    DATA = auto()
    SCAN_DATA = auto()
    TRACK_DATA = auto()
    RECEIVERS = auto()
    OUTPUT_PATH = auto()
    OBSERVATION_DATE = auto()
    BLOCK_NAME = auto()
    STANDING_WAVE_EPSILON_FUNCTION_DICT = auto()
    STANDING_WAVE_LEGENDRE_FUNCTION_DICT = auto()
    STANDING_WAVE_CHANNELS = auto()
