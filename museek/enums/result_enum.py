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
    STANDING_WAVE_CALIBRATOR_LABEL = auto()
    SCAN_OBSERVATION_START = auto()
    SCAN_OBSERVATION_END = auto()
    FLAG_REPORT_WRITER = auto()
    CALIBRATED_VIS = auto()
    FREQ_SELECT = auto()
    COMBINED_FLAG = auto()
    CORRELATION_COEFFICIENT_VIS_SYNCH_ANT = auto()
    POINT_SOURCE_MASK = auto()
    SCAN_FLAGS_BEFOREAOFLAGGER = auto()
