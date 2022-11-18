import os
from datetime import datetime
from enum import Enum
from typing import Optional, NamedTuple

import katdal
import numpy as np
from katdal import DataSet
from katdal.lazy_indexer import DaskLazyIndexer
from katpoint import Target, Antenna

from museek.receiver import Receiver
from museek.time_ordered_data_element import TimeOrderedDataElement

MODULE_ROOT = os.path.dirname(__file__)


class ScanTuple(NamedTuple):
    """
    A `NamedTuple` to hold a given scan's dump indices, the state as `str`, the scan index and the `Target` object.
    The definitions relate to `KatDal`.
    """
    dumps: list[int]
    state: str
    index: int
    target: Target


class ScanStateEnum(Enum):
    """ `Enum` class to define the scan states as named in `KatDal`. """
    SCAN = 'scan'
    TRACK = 'track'
    SLEW = 'slew'
    STOP = 'stop'


class TimeOrderedData:
    """
    Class for handling of time ordered data as provided by `katdal`.
    """

    def __init__(self,
                 block_name: str,
                 receivers: list[Receiver],
                 token: Optional[str],
                 data_folder: Optional[str],
                 force_load_from_correlator_data: bool = False,
                 do_create_cache: bool = True):
        """
        Initialize
        :param block_name: name of the observation block
        :param receivers: list of receivers to load the data of
        :param token: to access the data, usage of `token` is prioritized over `data_folder`
        :param data_folder: folder where data is stored
        :param force_load_from_correlator_data: if `True` ignores local cache files of visibility, flag or weights
        :param do_create_cache: if `True` a cache file of visibility, flag and weight data is created if it is not
                                already present
        """
        self.scan_dumps: list[int] | None = None
        self.track_dumps: list[int] | None = None
        self.slew_dumps: list[int] | None = None
        self.stop_dumps: list[int] | None = None

        # these can consume a lot of memory, so they are only loaded when needed
        self.visibility: TimeOrderedDataElement | None = None
        self.flags: TimeOrderedDataElement | None = None
        self.weights: TimeOrderedDataElement | None = None

        # to be able to load the katdal data again if needed
        self._katdal_open_argument: Optional[str] = None

        self._force_load_from_correlator_data = force_load_from_correlator_data
        self._do_save_to_disc = do_create_cache

        self.receivers = receivers
        self.correlator_products = self._get_correlator_products()

        data = self.load_data(block_name=block_name, data_folder=data_folder, token=token)
        self.all_antennas = data.ants
        self._select(data=data)
        self._data_str = str(data)
        self._cache_file_name = f'{data.name}_auto_visibility_flags_weights.npz'

        self.obs_script_log = data.obs_script_log
        self.shape = data.shape
        self.name = data.name
        self.dump_period = data.dump_period
        self.antennas = data.ants
        self._antenna_name_list = [antenna.name for antenna in self.antennas]

        self._scan_tuple_list = self._get_scan_tuple_list(data=data)
        self._set_scan_state_dumps()

        # data elements
        self.timestamps = self._element(array=data.timestamps[:, np.newaxis, np.newaxis])
        self.timestamp_dates = self._element(
            array=np.asarray([datetime.fromtimestamp(stamp) for stamp in data.timestamps])[:, np.newaxis, np.newaxis]
        )
        self.frequencies = self._element(array=data.freqs[np.newaxis, :, np.newaxis])

        # sky coordinates
        self.azimuth = self._element(array=data.az[:, np.newaxis, :])
        self.elevation = self._element(array=data.el[:, np.newaxis, :])
        self.declination = self._element(array=data.dec[:, np.newaxis, :])
        self.right_ascension = self._element(array=data.ra[:, np.newaxis, :])

        # climate
        self.temperature = self._element(array=data.temperature[:, np.newaxis, np.newaxis])
        self.humidity = self._element(array=data.humidity[:, np.newaxis, np.newaxis])
        self.pressure = self._element(array=data.pressure[:, np.newaxis, np.newaxis])

    def __str__(self):
        """ Returns the same `str` as `katdal`. """
        return self._data_str

    def load_data(self, block_name: str, token: Optional[str], data_folder: Optional[str]) -> DataSet:
        """
        Loads the data from `katdal` for `block_name` using either `token` or if it is `None`, the `data_folder`.
        """
        if token is not None:
            katdal_open_argument = f'https://archive-gw-1.kat.ac.za/{block_name}/{block_name}_sdp_l0.full.rdb?{token}'
        elif data_folder is not None:
            katdal_open_argument = os.path.join(data_folder,
                                                f'{block_name}/{block_name}/{block_name}_sdp_l0.full.rdb')
        else:
            raise ValueError('Either `token` or `data_folder` must be given and not `None`!')
        self._katdal_open_argument = katdal_open_argument
        return katdal.open(self._katdal_open_argument)

    def antenna(self, receiver) -> Antenna:
        """ Returns the `Antenna` object belonging to `receiver`. """
        return self.antennas[self._antenna_name_list.index(receiver.antenna_name)]

    def load_visibility_flags_weights(self):
        """ Load visibility, flag and weights and set them as attributes to `self`. """
        visibility, flags, weights = self._visibility_flags_weights()
        self.visibility = self._element(array=visibility)
        self.flags = [self._element(array=flags)]  # this will contain all kinds of flags
        self.weights = self._element(array=weights)

    def delete_visibility_flags_weights(self):
        """ Delete large arrays from memory, i.e. replace them with `None`. """
        self.visibility = None
        self.flags = None
        self.weights = None

    def _visibility_flags_weights(self, data: DataSet | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a tuple of visibility, flags and weights as `np.ndarray`s.
        It first looks for a cache file containing these. If that file is unavailabe, incomplete or
        if `self._force_load_from_correlator_data` is `True`, the cache file is created again.
        :param data: optional `katdal` `DataSet`, defaults to `None`
        :return: a tuple of visibility, flags and weights as `np.ndarray` each
        """
        cache_file_directory = os.path.join(MODULE_ROOT, '../cache')
        os.makedirs(cache_file_directory, exist_ok=True)
        cache_file = os.path.join(cache_file_directory, self._cache_file_name)
        if not os.path.exists(cache_file) or self._force_load_from_correlator_data:
            if data is None:
                data = katdal.open(self._katdal_open_argument)
                self._select(data=data)
            visibility, flags, weights = self._load_autocorrelation_visibility(data=data)
            if self._do_save_to_disc:
                np.savez_compressed(cache_file,
                                    visibility=visibility,
                                    flags=flags,
                                    weights=weights,
                                    correlator_products=data.corr_products)
        else:
            data_from_cache = np.load(cache_file)
            correlator_products = data_from_cache['correlator_products']
            try:  # if this fails it means that the cache file does not contain the correlator products
                correlator_products_indices = self._correlator_products_indices(
                    all_correlator_products=correlator_products
                )
            except ValueError:
                print(f'Recreating cache file for {self.name}...')
                self._force_load_from_correlator_data = True
                self._do_save_to_disc = True
                return self._visibility_flags_weights(data=data)
            visibility = data_from_cache['visibility'][:, :, correlator_products_indices]
            flags = data_from_cache['flags'][:, :, correlator_products_indices]
            weights = data_from_cache['weights'][:, :, correlator_products_indices]
        return visibility.real, flags, weights

    def _load_autocorrelation_visibility(self, data: DataSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the visibility, flags and weights from katdal lazy indexer.
        Note: this consumes a lot of memory depending on the selection of `data`.
        :param data: a `katdal` `DataSet`
        :return: a tuple of visibility, flags and weights as `np.ndarray` each
        """
        visibility = np.zeros(shape=self.shape, dtype=complex)
        flags = np.zeros(shape=self.shape, dtype=bool)
        weights = np.zeros(shape=self.shape, dtype=float)
        DaskLazyIndexer.get(arrays=[data.vis, data.flags, data.weights],
                            keep=...,
                            out=[visibility, flags, weights])
        return visibility, flags, weights

    def _set_scan_state_dumps(self):
        """
        Sets the dumps for each scan state defined in `ScanStateEnum`.
        For examle, the dumps for state `scan` will be set to `self.scan_dumps`.
        """
        for scan_state_enum in ScanStateEnum:
            scan_dumps = self._dumps_of_scan_state(scan_state=scan_state_enum)
            self.__setattr__(f'{scan_state_enum.value}_dumps', scan_dumps)

    def _correlator_products_indices(self, all_correlator_products: np.ndarray) -> list[int]:
        """
        Returns the indices belonging to the autocorrelation of the input receivers
        relative to `all_correlator_products`.
        """
        result = [np.where(np.prod(all_correlator_products == correlator_product, axis=1))[0]
                  for correlator_product in self.correlator_products]
        result = np.asarray(result, dtype=object)
        if len(result) != len(self.correlator_products) or len(result.shape) != 2 or result.shape[1] == 0:
            raise ValueError('Input `all_correlator_products` must contain all receivers.')
        result = np.atleast_1d(np.squeeze(result)).tolist()
        return result

    def _dumps_of_scan_state(self, scan_state: ScanStateEnum) -> list[int]:
        """ Returns the dump indices that belong to a certain `scan_sate`. """
        result = []
        for scan_tuple in self._scan_tuple_list:
            if scan_tuple.state == scan_state.value:
                result.extend(scan_tuple.dumps)
        return result

    @staticmethod
    def _get_scan_tuple_list(data: DataSet) -> list[ScanTuple]:
        """ Returns a `list` containing all `ScanTuple`s for `data`. """
        scan_tuple_list: list[ScanTuple] = []
        for index, state, target in data.scans():
            scan_tuple = ScanTuple(dumps=data.dumps, state=state, index=index, target=target)
            scan_tuple_list.append(scan_tuple)
        return scan_tuple_list

    def _get_correlator_products(self) -> list[list[str, str]]:
        """
        Returns a `list` containing a `list` with the same element twice, namely the `receiver` name,
        for each `receiver` in `self.receivers`.
        """
        return [[str(receiver)] * 2 for receiver in self.receivers]

    def _select(self, data: DataSet):
        """ Run `data._select()` on the correlator products in `self`. """
        data.select(corrprods=self._correlator_products_indices(all_correlator_products=data.corr_products))

    def _element(self, array: np.ndarray):
        """ Initialises and returns a `TimeOrderedDataElement` with `array` and `self` as parent. """
        return TimeOrderedDataElement(array=array, parent=self)
