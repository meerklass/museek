from abc import ABC

import numpy as np


class AbstractDataElement(ABC):
    """ Abstract base class for `DataElement`s and `FlagElement`s. Their shared methods are found here. """

    def __init__(self, array: np.ndarray):
        """
        :param array: a `numpy` array of shape `(n_dump | 1, n_frequency | 1, n_receiver | n_dish | 1)`
        :raise ValueError: if `array is not 3-dimensional
        """
        if len(array.shape) != 3:
            raise ValueError(f'Input `array` needs to be 3-dimensional, got shape {array.shape}')
        self.array = array

    def __getitem__(self, index: int | list[int]) -> np.ndarray:
        """ Returns `numpy`s getitem evaluated at `index` coupled with a `squeeze`. """
        return np.squeeze(self.array[index])

    def __str__(self):
        """ Return the string of the underlying array. """
        return str(self.array)

    def __eq__(self, other: 'AbstractDataElement'):
        """
        Return `True` if the underlying arrays are equal.
        This means their `shape` and content must be equal.
        """
        if self.shape != other.shape:
            return False
        return (self.array == other.array).all()

    @property
    def squeeze(self) -> np.ndarray:
        """ Returns a `numpy` `array` containing the all dumps of `self` without redundant dimensions. """
        array = self.get_array()
        if array.shape == (1, 1, 1):  # squeeze behaves weirdly in this case
            return array[0, 0, 0]
        return np.squeeze(array)

    @property
    def shape(self) -> tuple[int, int, int]:
        """ Returns the shape of the underlying numpy array. """
        return self.array.shape

    def get(self,
            *,  # force named parameters
            time: int | list[int] | slice | range | None = None,
            freq: int | list[int] | slice | range | None = None,
            recv: int | list[int] | slice | range | None = None,
            ):
        """
        Simplified indexing
        :param time: indices or slice along the zeroth (dump) axis
        :param freq: indices or slice along the first (frequency) axis
        :param recv: indices or slice along the second (receiver) axis
        :return: a copy of `self` indexed at the input indices
        """

        array = self.array.copy()

        if isinstance(time, int | np.int64):
            time = [time]
        if isinstance(freq, int | np.int64):
            freq = [freq]
        if isinstance(recv, int | np.int64):
            recv = [recv]

        if time is not None:
            array = array[time, :, :]
        if freq is not None:
            array = array[:, freq, :]
        if recv is not None:
            array = array[:, :, recv]

        # return new object
        return self.__class__(array=array)

    def get_array(self, **kwargs) -> np.ndarray | float:
        """
        Returns `self._array` after passing `kwargs` to `self.get()`.
        :param kwargs: passed on to `self.get()`
        """
        return self.get(**kwargs).array

    @classmethod
    def channel_iterator(cls, data_element: 'AbstractDataElement'):
        """ Iterate through the frequency channels of `data_element`. """
        for channel in np.moveaxis(data_element.array, 1, 0):
            yield cls(array=channel[:, np.newaxis, :])
