import numpy as np

from museek.data_element import DataElement


class FlagElement:
    """ Class to contain a `list` of flags encapsulated as `DataElement`s each. """

    def __init__(self, flags: list[DataElement]):
        """ Initialise with `flags`, a `list of `DataElement`s. """
        self._flags = flags
        self._check_flags()

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

    def add_flag(self, flag: DataElement):
        """ Append `flag` to `self` and check for compatibility. """
        self._flags.append(flag)
        self._check_flags()

    def combine(self, threshold: int = 1) -> DataElement:
        """
        Combine all flags and return them as a single boolean `DataElement` after thresholding with `threshold`.
        """
        result_array = np.zeros(self.shape)
        for flag in self._flags:
            result_array += flag.get_array()
        result_array[result_array < threshold] = 0
        return DataElement(array=np.asarray(result_array, dtype=bool))

    def get(self, **kwargs) -> 'FlagElement':
        """ Wraps `DataElement.get()` around each flag in `self` and returns a new `FlagElement`. """
        return FlagElement(flags=[flag.get(**kwargs) for flag in self._flags])

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
