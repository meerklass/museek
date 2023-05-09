from typing import Union

import numpy as np

from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory


class FlagElement:
    """ Class to contain a `list` of flags encapsulated as `DataElement`s each. """

    def __init__(self, flags: list[DataElement]):
        """ Initialise with `flags`, a `list of `DataElement`s. """
        self._flags = flags
        self._check_flags()
        self._data_element_factory = DataElementFactory()

    def __len__(self):
        """ Return the number of `DataElement`s in `self`. """
        return len(self._flags)

    def __eq__(self, other: 'FlagElement'):
        """ Return `True` if all flags in `self` are equal to the flags in `other` at the same index. """
        if len(self) != len(other):
            return False
        return all([self_flag == other_flag for self_flag, other_flag in zip(self._flags, other._flags)])

    @property
    def shape(self):
        """ Return the shape of the first element in `self._flags`. All elements have the same shape. """
        return self._flags[0].shape

    def add_flag(self, flag: Union[DataElement, 'FlagElement']):
        """ Append `flag` to `self` and check for compatibility. """
        if isinstance(flag, FlagElement):
            if flag_len := len(flag) > 1:
                raise ValueError(f'Adding more than one flag at once is not implemented yet. Got {flag_len} flags.')
            flag = flag._flags[0]
        self._flags.append(flag)
        self._check_flags()

    def remove_flag(self, index: int):
        """ Remove `flag` at `index` in `self.flags`. """
        self._flags = [flag for i, flag in enumerate(self._flags) if i != index]
        self._check_flags()

    def combine(self, threshold: int = 1) -> DataElement:
        """
        Combine all flags and return them as a single boolean `DataElement` after thresholding with `threshold`.
        """
        result_array = np.zeros(self.shape)
        for flag in self._flags:
            result_array += flag.get_array()
        result_array[result_array < threshold] = 0
        return self._data_element_factory.create(array=np.asarray(result_array, dtype=bool))

    def get(self, **kwargs) -> 'FlagElement':
        """ Wraps `DataElement.get()` around each flag in `self` and returns a new `FlagElement`. """
        return FlagElement(flags=[flag.get(**kwargs) for flag in self._flags])

    def insert_receiver_flag(self, flag: DataElement, i_receiver: int, index: int):
        """ Insert `flag` for receiver with index `i_receiver` into the flag in `self` at `index`. """
        if n_flag_recv := flag.shape[-1] != 1:
            raise ValueError(f'Input `flag` needs to be for exactly one receiver, but got {n_flag_recv}')
        flag_at_index = self._flags[index]._array
        flag_at_index[:, :, i_receiver] = np.logical_or(flag_at_index[:, :, i_receiver], flag._array[:, :, 0])
        self._flags[index] = self._data_element_factory.create(array=flag_at_index)

    def array(self) -> np.ndarray[bool]:
        """ Return the flags in format for storage as a `numpy` array. """
        return np.asarray([flag._array for flag in self._flags])

    def _check_flags(self):
        """ Check if all flags are compatible. """
        self._check_flag_shapes()
        self._check_flag_types()

    def _check_flag_shapes(self):
        """
        Check if the flag shapes are identical.
        :raise ValueError: if not all shapes are identical
        """
        for flag in self._flags:
            if flag.shape != self.shape:
                raise ValueError(f'All input flags need to have the same shape {self.shape}. Got {flag.shape}.')

    def _check_flag_types(self):
        """
        Check if all flags are of type `DataElement`.
        :raise ValueError: if at least one of the flags is not a `DataElement`
        """
        for flag in self._flags:
            if not isinstance(flag, DataElement):
                raise ValueError(f'All input flags need to be `DataElement`s. Got {type(flag)}.')
