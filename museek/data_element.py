import numbers
from typing import Union

import numpy as np


class DataElement:
    """
    Class to access an 'element' of time ordered data, e.g. the visibility data, or temperature values.
    All elements are internally stored with shape `(n_dump, n_frequency, n_receiver)`. If one of these axes only
    contains copies, e.g. the temperature is the same for all frequencies, then the corresponding shape is `1`.
    The elements should be accessed using one of the properties and manipulated with the public methods.
    """

    def __init__(self, array: np.ndarray):
        """
        :param array: a `numpy` array of shape `(n_dump | 1, n_frequency | 1, n_receiver | n_dish | 1)`
        :raise ValueError: if `array is not 3-dimensional
        """
        if len(array.shape) != 3:
            raise ValueError(f'Input `array` needs to be 3-dimensional, got shape {array.shape}')
        self._array = array
        self.shape = array.shape

    def __mul__(self, other: Union['DataElement', np.ndarray, numbers.Number]) -> 'DataElement':
        """
        Multiplication of two `DataElement`s and of one `DataElement` with a `np.ndarray` or any `Number`.
        :raise ValueError: if the shapes of `self` and `other` do not match
        """
        if not isinstance(other, numbers.Number):
            if self.shape != other.shape:
                raise ValueError(f'Cannot multiply instances with different shapes, '
                                 f'got {self.shape} and {other.shape}.')
        if isinstance(other, DataElement):
            return DataElement(array=self._array * other._array)
        if isinstance(other, np.ndarray | numbers.Number):
            return DataElement(array=self._array * other)

    def __getitem__(self, index: int | list[int]) -> np.ndarray:
        """ Returns `numpy`s getitem evaluated at `index` coupled with a `squeeze`. """
        return np.squeeze(self._array[index])

    @property
    def squeeze(self) -> np.ndarray:
        """ Returns a `numpy` `array` containing the all dumps of `self` without redundant dimensions. """
        array = self.get_array()
        if array.shape == (1, 1, 1):  # squeeze behaves weirdly in this case
            return array[0, 0, 0]
        return np.squeeze(array)

    def mean(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Return the mean of `self` along `axis` as a `DataElement`, i.e. the dimensions are kept. """
        return DataElement(array=np.mean(self._array, axis=axis, keepdims=True))

    def get(self,
            *,  # force named parameters
            time: int | list[int] | slice | None = None,
            freq: int | list[int] | slice | None = None,
            recv: int | list[int] | slice | None = None,
            ) -> 'DataElement':
        """
        Simplified indexing
        :param time: indices or slice along the zeroth (dump) axis
        :param freq: indices or slice along the first (frequency) axis
        :param recv: indices or slice along the second (receiver) axis
        :return: a copy of `self` indexed at the input indices
        """

        array = self._array.copy()

        if isinstance(time, int):
            time = [time]
        if isinstance(freq, int):
            freq = [freq]
        if isinstance(recv, int):
            recv = [recv]

        if time is not None:
            array = array[time, :, :]
        if freq is not None:
            array = array[:, freq, :]
        if recv is not None:
            array = array[:, :, recv]

        return DataElement(array=array)

    def get_array(self, **kwargs) -> np.ndarray | float:
        """
        Returns `self._array` after passing `kwargs` to `self.get()`.
        :param kwargs: passed on to `self.get()`
        """
        return self.get(**kwargs)._array
