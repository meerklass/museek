from enum import Enum

from museek.factory.data_element_factory import ScanElementFactory, AbstractDataElementFactory


class ScanStateEnum(Enum):
    """
    `Enum` class to define the scan states as named in `KatDal`.
    The individual `enum`s are `tuple`s of `string` and `AbstractDataElementFactory` objects belonging to the
    scan state.
    """
    SCAN = ('scan', ScanElementFactory)
    TRACK = ('track', ScanElementFactory)
    SLEW = ('slew', ScanElementFactory)
    STOP = ('stop', ScanElementFactory)

    def factory(self, *args, **kwargs) -> AbstractDataElementFactory:
        """ Initialise and return the `AbstractDataElementFactory` in `self`. """
        return self.value[1](*args, **kwargs)

    @property
    def scan_name(self) -> str:
        """ Return the `str` name of the scan state. """
        return self.value[0]

    @classmethod
    def get_enum(cls, enum_string) -> 'ScanStateEnum':
        """ Return the `enum` with `scan_name` equal to `enum_string`. """
        for enum_ in cls:
            if enum_.scan_name == enum_string and enum_.name.lower() == enum_string:
                return enum_
