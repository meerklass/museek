from enum import Enum
from typing import Union, Optional

import numpy as np

from museek.enums.flag_enum import FlagEnum
from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_element import FlagElement


class FlagList:
    """ Class to contain a `list` of flags encapsulated as `FlagElement`s each. """

    def __init__(self, flags: list[FlagElement], flag_names: Union[list[Enum], None]):
        """
        Initialise with `flags`, a `list of `FlagElement`s and optional names `flag_names`.
        :raise ValueError: if the lengths of `flags` and `flag_names` are not equal
        """
        self._flags = flags
        if flag_names is None:
            flag_names = self._generic_flag_names()
        self._flag_names = flag_names
        self._check_flags()
        self._check_flag_names()
        self._flag_element_factory = FlagElementFactory()

    def __len__(self):
        """ Return the number of `FlagElement`s in `self`. """
        return len(self._flags)

    def __eq__(self, other: 'FlagList'):
        """ Return `True` if all flags in `self` are equal to the flags in `other` at the same index. """
        if len(self) != len(other):
            return False
        return all([self_flag == other_flag for self_flag, other_flag in zip(self._flags, other._flags)]) \
            and all([self_name == other_name for self_name, other_name in zip(self._flag_names, other._flag_names)])

    @classmethod
    def from_array(
            cls,
            array: np.ndarray,
            element_factory: FlagElementFactory,
            flag_names: Optional[list[Enum]] = None
    ) -> 'FlagList':
        """
        Alternative constructor from a 3 or 4-dimensional `array` using the factory `element_factory`.
        :param array: must be 3 or 4-dimensional boolean array
        :param element_factory: to instantiate `FlagElement`s
        :param flag_names: `list` of `Enum` flag identifiers for creation. defaults to `None` for generic names
        :raise ValueError: if `array` is not 3 or 4-D
        :return: `FlagList` instance
        """
        if len(array.shape) == 3:
            array = array[np.newaxis]
        if wrong_shape := len(array.shape) != 4:
            raise ValueError(f'Input `array` needs to be 4-dimensional, got {wrong_shape}.')
        return cls(flags=[element_factory.create(array=flag) for flag in array], flag_names=flag_names)

    @property
    def shape(self):
        """ Return the shape of the first element in `self._flags`. All elements have the same shape. """
        return self._flags[0].shape

    @property
    def array(self) -> np.ndarray[bool]:
        """ Return the flags in format for storage as a `numpy` array. """
        return np.asarray([flag.array for flag in self._flags])

    @property
    def flag_names(self) -> list[str]:
        """ Return the flag names as a list of strings. """
        return [flag_name.name for flag_name in self._flag_names]

    def add_flag(self, flag: Union[FlagElement, 'FlagList'], flag_names: Union[Enum, list[Enum]]):
        """
        Append `flag` to `self` and check for compatibility.
        :raise ValueError: if more than 1 flag is attempted to be added
        :raise ValueError: if more than 1 flag name is given
        """
        if isinstance(flag, FlagList):
            if (flag_len := len(flag)) > 1:
                raise ValueError(f'Adding more than one flag at once is not implemented yet. Got {flag_len} flags.')
            flag = flag._flags[0]
        if isinstance(flag_names, list):
            if len(flag_names) > 1:
                raise ValueError(
                    f'Parameter `flag_name` cannot be a list of length more than 1. Got {len(flag_names)}. '
                )
            flag_names = flag_names[0]
        self._flags.append(flag)
        self._flag_names.append(flag_names)
        self._check_flags()
        self._check_flag_names()

    def remove_flag(self, index: int):
        """ Remove `flag` at `index` in `self.flags`. """
        self._flags = [flag for i, flag in enumerate(self._flags) if i != index]
        self._flag_names = [name for i, name in enumerate(self._flag_names) if i != index]
        self._check_flags()
        self._check_flag_names()

    def combine(self, threshold: int = 1) -> FlagElement:
        """
        Combine all flags and return them as a single boolean `FlagElement` after thresholding with `threshold`.
        """
        result_array = np.zeros(self.shape)
        for flag in self._flags:
            result_array += flag.get_array()
        result_array[result_array < threshold] = 0
        return self._flag_element_factory.create(array=np.asarray(result_array, dtype=bool))

    def get(self, **kwargs) -> 'FlagList':
        """ Wraps `FlagElement.get()` around each flag in `self` and returns a new `FlagList`. """
        return FlagList(flags=[flag.get(**kwargs) for flag in self._flags],
                        flag_names=self._flag_names)

    def insert_receiver_flag(self, flag: FlagElement, i_receiver: int, index: int):
        """ Insert `flag` for receiver with index `i_receiver` into the flag in `self` at `index`. """
        if flag.shape[-1] != 1:
            raise ValueError(f'Input `flag` needs to be for exactly one receiver, but got {flag.shape[-1]}')
        flag_at_index = self._flags[index]
        flag_at_index.insert_receiver_flag(i_receiver=i_receiver, flag=flag)
        self._flags[index] = flag_at_index

    def _check_flags(self):
        """ Check if all flags are compatible. """
        self._check_flag_shapes()
        self._check_flag_types()

    def _check_flag_names(self):
        """ Check if the flag names are compatible. """
        if len(self._flags) != len(self._flag_names):
            raise ValueError(
                f'The length of `flags` and `flag_names` must be equal, '
                f'got {len(self._flags)} and {len(self._flag_names)}.'
            )

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
        Check if all flags are of type `FlagElement`.
        :raise ValueError: if at least one of the flags is not a `FlagElement`
        """
        for flag in self._flags:
            if not isinstance(flag, FlagElement):
                raise ValueError(f'All input flags need to be `FlagElement`s. Got {type(flag)}.')

    def _generic_flag_names(self) -> list[Enum]:
        """ Returns a list of strings containing consecutive integers. """
        try:
            return [FlagEnum(i) for i in range(len(self._flags))]
        except KeyError:
            raise NotImplementedError(
                f'We do not have enough `FlagEnum`s implemented,'
                f' please extend the list to beyond {len(self._flags)} entries.'
            )
