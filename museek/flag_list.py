from typing import TYPE_CHECKING, Union

import numpy as np

from museek.flag_element import FlagElement

if TYPE_CHECKING:
    from museek.factory.data_element_factory import FlagElementFactory


class FlagList:
    """Class to contain a `list` of flags encapsulated as `FlagElement`s each."""

    def __init__(self, flags: list[FlagElement]):
        """Initialise with `flags`, a `list of `FlagElement`s."""
        from museek.factory.data_element_factory import FlagElementFactory

        self._flags = flags
        self._check_flags()
        self._flag_element_factory = FlagElementFactory()

    def __len__(self):
        """Return the number of `FlagElement`s in `self`."""
        return len(self._flags)

    def __eq__(self, other: "FlagList"):
        """Return `True` if all flags in `self` are equal to the flags in `other` at the same index."""
        if len(self) != len(other):
            return False
        return all(
            [
                self_flag == other_flag
                for self_flag, other_flag in zip(self._flags, other._flags)
            ]
        )

    @classmethod
    def from_array(
        cls, array: np.ndarray, element_factory: "FlagElementFactory"
    ) -> "FlagList":
        """
        Alternative constructor from a 3 or 4-dimensional `array` using the factory `element_factory`.
        :param array: must be 3 or 4-dimensional boolean array
        :param element_factory: to instantiate `FlagElement`s
        :raise ValueError: if `array` is not 3 or 4-D
        :return: `FlagList` instance
        """
        if len(array.shape) == 3:
            array = array[np.newaxis]
        if wrong_shape := len(array.shape) != 4:
            raise ValueError(
                f"Input `array` needs to be 4-dimensional, got {wrong_shape}."
            )
        return cls(flags=[element_factory.create(array=flag) for flag in array])

    @property
    def shape(self):
        """Return the shape of the first element in `self._flags`. All elements have the same shape."""
        return self._flags[0].shape

    @property
    def array(self) -> np.ndarray[bool]:
        """Return the flags in format for storage as a `numpy` array."""
        return np.asarray([flag.array for flag in self._flags])

    def add_flag(self, flag: Union[FlagElement, "FlagList"]):
        """Append `flag` to `self` and check for compatibility."""
        if isinstance(flag, FlagList):
            if flag_len := len(flag) > 1:
                raise ValueError(
                    f"Adding more than one flag at once is not implemented yet. Got {flag_len} flags."
                )
            flag = flag._flags[0]
        self._flags.append(flag)
        self._check_flags()

    def remove_flag(self, index: int):
        """Remove `flag` at `index` in `self.flags`."""
        self._flags = [flag for i, flag in enumerate(self._flags) if i != index]
        self._check_flags()

    def combine(self, threshold: int = 1) -> FlagElement:
        """
        Combine all flags and return them as a single boolean `FlagElement` after thresholding with `threshold`.
        """
        result_array = np.zeros(self.shape)
        for flag in self._flags:
            result_array += flag.get_array()
        result_array[result_array < threshold] = 0
        return self._flag_element_factory.create(
            array=np.asarray(result_array, dtype=bool)
        )

    def get(self, **kwargs) -> "FlagList":
        """Wraps `FlagElement.get()` around each flag in `self` and returns a new `FlagList`."""
        return FlagList(flags=[flag.get(**kwargs) for flag in self._flags])

    def insert_receiver_flag(self, flag: FlagElement, i_receiver: int, index: int):
        """Insert `flag` for receiver with index `i_receiver` into the flag in `self` at `index`."""
        if flag.shape[-1] != 1:
            raise ValueError(
                f"Input `flag` needs to be for exactly one receiver, but got {flag.shape[-1]}"
            )
        flag_at_index = self._flags[index]
        flag_at_index.insert_receiver_flag(i_receiver=i_receiver, flag=flag)
        self._flags[index] = flag_at_index

    def _check_flags(self):
        """Check if all flags are compatible."""
        self._check_flag_shapes()
        self._check_flag_types()

    def _check_flag_shapes(self):
        """
        Check if the flag shapes are identical.
        :raise ValueError: if not all shapes are identical
        """
        for flag in self._flags:
            if flag.shape != self.shape:
                raise ValueError(
                    f"All input flags need to have the same shape {self.shape}. Got {flag.shape}."
                )

    def _check_flag_types(self):
        """
        Check if all flags are of type `FlagElement`.
        :raise ValueError: if at least one of the flags is not a `FlagElement`
        """
        for flag in self._flags:
            if not isinstance(flag, FlagElement):
                raise ValueError(
                    f"All input flags need to be `FlagElement`s. Got {type(flag)}."
                )
