import numbers
from copy import deepcopy
from typing import Union

import numpy as np

from museek.abstract_data_element import AbstractDataElement


class DataElement(AbstractDataElement):
    """
    Class to access an 'element' of time ordered data, e.g. the visibility data, or temperature values.
    All elements are internally stored with shape `(n_dump, n_frequency, n_receiver)`. If one of these axes only
    contains copies, e.g. the temperature is the same for all frequencies, then the corresponding shape is `1`.
    """

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
            return DataElement(array=self.array * other.array)
        if isinstance(other, np.ndarray | numbers.Number):
            return DataElement(array=self.array * other)

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
            return DataElement(array=self.array / other.array)
        if isinstance(other, np.ndarray | numbers.Number):
            return DataElement(array=self.array / other)

    def __sub__(self, other: Union['DataElement', np.ndarray, numbers.Number]) -> 'DataElement':
        """
        Subtraction of two `DataElement`s and of one `DataElement` with a `np.ndarray` or any `Number`.
        :raise ValueError: if the shapes of `self` and `other` do not match
        """
        if not isinstance(other, numbers.Number):
            if self.shape != other.shape:
                raise ValueError(f'Cannot subtract instances with different shapes, '
                                 f'got {self.shape} and {other.shape}.')
        if isinstance(other, DataElement):
            return DataElement(array=self.array - other.array)
        if isinstance(other, np.ndarray | numbers.Number):
            return DataElement(array=self.array - other)

    def __add__(self, other: Union['DataElement', np.ndarray, numbers.Number]) -> 'DataElement':
        """
        Addition of two `DataElement`s and of one `DataElement` with a `np.ndarray` or any `Number`.
        :raise ValueError: if the shapes of `self` and `other` do not match
        """
        if not isinstance(other, numbers.Number):
            if self.shape != other.shape:
                raise ValueError(f'Cannot add instances with different shapes, '
                                 f'got {self.shape} and {other.shape}.')
        if isinstance(other, DataElement):
            return DataElement(array=self.array + other.array)
        if isinstance(other, np.ndarray | numbers.Number):
            return DataElement(array=self.array + other)

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
            return self._mean(axis=axis)
        return self._flagged_mean(axis=axis, flags=flags)

    def standard_deviation(
            self,
            axis: int | list[int, int] | tuple[int, int],
            flags: Union['FlagList', None] = None
    ) -> 'DataElement':
        """
        Return the standard deviation of the unflagged entries in `self` along `axis` as a `DataElement`,
        i.e. the dimensions are kept.
        :param axis: axis along which to calculate the mean
        :param flags: optional, only entries not flagged by these are used
        :return: `DataElement` containing the standard deviation along `axis`
        """
        if flags is None:
            return self._std(axis=axis)
        return self._flagged_std(axis=axis, flags=flags)

    def sum(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Return the sum of `self` along `axis` as a `DataElement`, i.e. the dimensions are kept. """
        return DataElement(array=np.sum(self.array, axis=axis, keepdims=True))

    def min(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Wrapper of `numpy.min()`. """
        return DataElement(array=np.min(self.array, axis=axis, keepdims=True))

    def max(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Wrapper of `numpy.max(). """
        return DataElement(array=np.max(self.array, axis=axis, keepdims=True))

    def fill(self, to_shape: tuple[int, int, int]) -> 'DataElement':
        """
        Repeat `self` to have `to_shape` and return the result.
        :raise ValueError: if no axis of `self` has the same length as `to_shape`
        :raise ValueError: if an axis of `self` is both inequal to 1 and to the corresponding entry in `to_shape`
        """
        message = f'Cannot fill `self` with shape {self.shape} to {to_shape}.'
        if self.shape == to_shape:
            return self
        shapes_equal = np.asarray(self.shape) == np.asarray(to_shape)
        if not shapes_equal.any():
            raise ValueError(message)
        new_array = deepcopy(self.array)
        for i, continue_ in enumerate(shapes_equal):
            if not continue_:
                if self.shape[i] != 1:
                    raise ValueError(message)
                new_array = np.repeat(new_array, to_shape[i], axis=i)
        return DataElement(array=new_array)

    def _mean(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Return a `DataElement` created from the output of `np.mean` applied along `axis`. """
        return DataElement(array=np.mean(self.array, axis=axis, keepdims=True))

    def _std(self, axis: int | list[int, int] | tuple[int, int]) -> 'DataElement':
        """ Return a `DataElement` created from the output of `np.std` applied along `axis`. """
        return DataElement(array=np.std(self.array, axis=axis, keepdims=True))

    def _flagged_mean(self, axis: int | list[int, int] | tuple[int, int], flags: 'FlagList') -> 'DataElement':
        """
        Return the mean of the unflagged entries in `self` along `axis` as a `DataElement`,
        i.e. the dimensions are kept.
        :param axis: axis along which to calculate the mean
        :param flags: only entries not flagged by these are used
        :return: `DataElement` containing the mean along `axis`
        """
        combined = flags.combine(threshold=1)
        masked = np.ma.masked_array(self.array, combined.array)
        return DataElement(array=masked.mean(axis=axis, keepdims=True))

    def _flagged_std(self, axis: int | list[int, int] | tuple[int, int], flags: 'FlagList') -> 'DataElement':
        """
        Return the standard deviation of the unflagged entries in `self` along `axis` as a `DataElement`,
        i.e. the dimensions are kept.
        :param axis: axis along which to calculate the mean
        :param flags: only entries not flagged by these are used
        :return: `DataElement` containing the standard deviation along `axis`
        """
        combined = flags.combine(threshold=1)
        masked = np.ma.masked_array(self.array, combined.array)
        return DataElement(array=masked.std(axis=axis, keepdims=True))
