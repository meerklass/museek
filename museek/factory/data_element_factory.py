from abc import ABC, abstractmethod

import numpy as np

from museek.data_element import DataElement
from museek.flag_element import FlagElement


class AbstractDataElementFactory(ABC):
    """ Abstract base class for `DataElement` factories. """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def create(self, **kwargs) -> DataElement:
        """ Initialise and return a `DataElement` object. """
        pass


class DataElementFactory(AbstractDataElementFactory):
    """ `DataElement` factory. """

    def create(self, array: np.ndarray) -> DataElement:
        """ Initialise and return a `DataElement` object with `array`. """
        return DataElement(array=array)


class FlagElementFactory(AbstractDataElementFactory):
    """ `FlagElement` factory. """

    def create(self, array: np.ndarray) -> FlagElement:
        """ Initialise and return a `FlagElement` object with `array`. """
        return FlagElement(array=array)


class ScanElementFactory(AbstractDataElementFactory):
    """
    `AbstractDataElement` factory specific to a certain scan state. Follows the decorator pattern.
    """

    def __init__(self, scan_dumps: list[int], component: DataElementFactory | FlagElementFactory):
        """
        Initialise super class and set a `DataElementFactory` as a component.
        :param scan_dumps: dump indices belonging to the scan state
        """
        super().__init__()
        self._component = component
        self._scan_dumps = scan_dumps

    def create(self, array: np.ndarray) -> DataElement | FlagElement:
        """
        Initialise and return a `DataElement` object with `array` indexed at `self._scan_dumps`.
        """
        if array.shape[0] > 1:
            array = array[self._scan_dumps]
        return self._component.create(array=array)
