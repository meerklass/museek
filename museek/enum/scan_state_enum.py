from enum import Enum


class ScanStateEnum(Enum):
    """ `Enum` class to define the scan states as named in `KatDal`. """
    SCAN = 'scan'
    TRACK = 'track'
    SLEW = 'slew'
    STOP = 'stop'
