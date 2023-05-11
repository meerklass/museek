import numpy as np

from museek.abstract_data_element import AbstractDataElement
from museek.data_element import DataElement


class FlagElement(AbstractDataElement):
    """ Class to encapsulate flags, which are basically `DataElement`s with binary entries. """

    def __init__(self, array: np.ndarray):
        """
        Initialise and check if `array` is binary
        :param array: a binary `numpy` array of shape `(n_dump | 1, n_frequency | 1, n_receiver | n_dish | 1)`
        :raise ValueError: if `array` is not binary
        """
        super().__init__(array=self._make_boolean(array=array))

    def __add__(self, other: 'FlagElement'):
        """
        Adding to masks gives the combined mask.
        Raises a `ValueError` if `other` is not a `FlagElement`.
        """
        if isinstance(other, FlagElement):
            result = self._array + other._array
            return FlagElement(array=result)
        raise ValueError(f'Cannot add class {type(other)} to `FlagElement`.')

    def sum(self, axis: int | list[int, int] | tuple[int, int]) -> DataElement:
        """ Return the sum of `self` along `axis` as a `DataElement`, i.e. the dimensions are kept. """
        return DataElement(array=np.sum(self._array, axis=axis, keepdims=True))

    def insert_receiver_flag(self, flag: 'FlagElement', i_receiver: int):
        """
        Insert `flag` for receiver with index `i_receiver` into `self`.
        :param flag: needs to contain only one receiver
        :param i_receiver: the index of the receiver wrt the receiver list
        :raise ValueError: if `flag` contains flags for more than one single receiver
        """
        if flag.shape[-1] != 1:
            raise ValueError(f'Input `flag` needs to be for one receiver only, got {flag.shape[-1]}')
        self._array[:, :, i_receiver] = np.logical_or(self._array[:, :, i_receiver], flag._array[:, :, 0])

    @staticmethod
    def _make_boolean(array: np.ndarray):
        """
        Return boolean array
        :param array: binary `numpy` array
        :raise ValueError: if `array` is not binary
        :return: boolean array
        """
        boolean_array = array.astype(bool)
        if not np.array_equal(array, boolean_array):
            raise ValueError('`FlagElement` can only initialise with a binary array.')
        return boolean_array
