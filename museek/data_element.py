import numbers
from typing import Union

import numpy as np


class DataElement:
    """
    Class to access an 'element' of time ordered data, e.g. the visibility data, or temperature values.
    All elements are internally stored with shape `(n_dump, n_frequency, n_receiver)`. If one of these axes only
    contains copies, e.g. the temperature is the same for all frequencies, then the corresponding shape is `1`.
    """

    def __init__(self, array: np.ndarray):
        """
        :param array: a `numpy` array of shape `(n_dump | 1, n_frequency | 1, n_receiver | n_dish | 1)`
        :raise ValueError: if `array is not 3-dimensional
        """
        if len(array.shape) != 3:
            raise ValueError(f'Input `array` needs to be 3-dimensional, got shape {array.shape}')
        self._array = array

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

    def __truediv__(self, other: Union['DataElement', np.ndarray, numbers.Number]) -> 'DataElement':
        """
        Division of two `DataElement`s and of one `DataElement` with a `np.ndarray` or any `Number`.
        :raise ValueError: if the shapes of `self` and `other` do not match
        """
        if not isinstance(other, numbers.Number):
            if self.shape != other.shape:
                raise ValueError(f'Cannot multiply instances with different shapes, '
                                 f'got {self.shape} and {other.shape}.')
        if isinstance(other, DataElement):
            return DataElement(array=self._array / other._array)
        if isinstance(other, np.ndarray | numbers.Number):
            return DataElement(array=self._array / other)

    def __getitem__(self, index: int | list[int]) -> np.ndarray:
        """ Returns `numpy`s getitem evaluated at `index` coupled with a `squeeze`. """
        return np.squeeze(self._array[index])

    def __str__(self):
        """ Return the string of the underlying array. """
        return str(self._array)

    def __eq__(self, other: 'DataElement'):
        """
        Return `True` if the underlying arrays are equal.
        This means their `shape` and content must be equal.
        """
        if self.shape != other.shape:
            return False
        return (self._array == other._array).all()

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
        return self._array.shape

    def mean(
            self,
            axis: int | list[int, int] | tuple[int, int],
            flags: Union['FlagList', None] = None
    ) -> 'DataElement':
        """
        Return the mean of the unflagged entries in `self` along `axis` as a `DataElement`,
        i.e. the dimensions are kept.
        :param axis: axis along which to calculate the mean
        :param flags: optional, only entries not flagged by these are used
        :return: `DataElement` containing the mean along `axis`
        """
        if flags is None:
            return DataElement(array=np.mean(self._array, axis=axis, keepdims=True))
        return self._flagged_mean(axis=axis, flags=flags)

    def sum(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Return the sum of `self` along `axis` as a `DataElement`, i.e. the dimensions are kept. """
        return DataElement(array=np.sum(self._array, axis=axis, keepdims=True))

    def get(self,
            *,  # force named parameters
            time: int | list[int] | slice | range | None = None,
            freq: int | list[int] | slice | range | None = None,
            recv: int | list[int] | slice | range | None = None,
            ) -> 'DataElement':
        """
        Simplified indexing
        :param time: indices or slice along the zeroth (dump) axis
        :param freq: indices or slice along the first (frequency) axis
        :param recv: indices or slice along the second (receiver) axis
        :return: a copy of `self` indexed at the input indices
        """

        array = self._array.copy()

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

        return DataElement(array=array)

    def get_array(self, **kwargs) -> np.ndarray | float:
        """
        Returns `self._array` after passing `kwargs` to `self.get()`.
        :param kwargs: passed on to `self.get()`
        """
        return self.get(**kwargs)._array

    def min(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Wrapper of `numpy.min()`. """
        return DataElement(array=np.min(self._array, axis=axis, keepdims=True))

    def max(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Wrapper of `numpy.max(). """
        return DataElement(array=np.max(self._array, axis=axis, keepdims=True))

    @classmethod
    def channel_iterator(cls, data_element: 'DataElement'):
        """ Iterate through the frequency channels of `data_element`. """
        for channel in np.moveaxis(data_element._array, 1, 0):
            yield cls(array=channel[:, np.newaxis, :])

    def _flagged_mean(self, axis: int | list[int, int] | tuple[int, int], flags: 'FlagList') -> 'DataElement':
        """
        Return the mean of the unflagged entries in `self` along `axis` as a `DataElement`,
        i.e. the dimensions are kept.
        :param axis: axis along which to calculate the mean
        :param flags: only entries not flagged by these are used
        :return: `DataElement` containing the mean along `axis`
        """
        combined = flags.combine(threshold=1)
        masked = np.ma.masked_array(self._array, combined._array)
        return DataElement(array=masked.mean(axis=axis, keepdims=True))
