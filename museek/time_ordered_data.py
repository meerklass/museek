import os
from copy import copy
from datetime import datetime
from typing import Optional, NamedTuple, Any

import katdal
import numpy as np
from katdal import DataSet
from katdal.lazy_indexer import DaskLazyIndexer
from katpoint import Target, Antenna

from definitions import ROOT_DIR
from museek.data_element import DataElement
from museek.enum.scan_state_enum import ScanStateEnum
from museek.factory.data_element_factory import AbstractDataElementFactory, DataElementFactory
from museek.flag_element import FlagElement
from museek.receiver import Receiver
from museek.util.clustering import Clustering


class ScanTuple(NamedTuple):
    """
    A `NamedTuple` to hold a given scan's dump indices, the state as `ScanStateEnum` and
    the scan index and the `Target` object.
    The definitions relate to `KatDal`.
    """
    dumps: list[int]
    state: ScanStateEnum
    index: int
    target: Target


class TimeOrderedData:
    """
    Class for handling time ordered data coming from `katdal`.
    """

    def __init__(self,
                 block_name: str,
                 receivers: list[Receiver],
                 token: Optional[str],
                 data_folder: Optional[str],
                 scan_state: ScanStateEnum | None = None,
                 force_load_from_correlator_data: bool = False,
                 do_create_cache: bool = True):
        """
        Initialise
        :param block_name: name of the observation block
        :param receivers: list of receivers to load the data of
        :param token: to access the data, usage of `token` is prioritized over `data_folder`
        :param data_folder: folder where data is stored
        :param scan_state: optional `ScanStateEnum` defining the scan state name. If it is given, only timestamps for
                           that scan state are loaded.
        :param force_load_from_correlator_data: if `True` ignores local cache files of visibility, flag or weights
        :param do_create_cache: if `True` a cache file of visibility, flag and weight data is created if it is not
                                already present
        """
        # these can consume a lot of memory, so they are only loaded when needed
        self.visibility: DataElement | None = None
        self.flags: FlagElement | None = None
        self.weights: DataElement | None = None

        self._block_name = block_name
        self._token = token
        self._data_folder = data_folder
        if self._token is None and self._data_folder is None:
            raise ValueError('Either `token` or `data_folder` must be given and not `None`!')

        # to be able to load the katdal data again if needed
        self._katdal_open_argument: Optional[str] = None

        self._force_load_from_correlator_data = force_load_from_correlator_data
        self._do_save_to_disc = do_create_cache

        data = self._get_data()
        self.receivers = self._get_receivers(receivers=receivers, data=data)
        self.correlator_products = self._get_correlator_products()
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

        self.scan_state: ScanStateEnum | None = None
        self._element_factory: AbstractDataElementFactory | None = None

        self.timestamps: DataElement | None = None
        self.original_timestamps: DataElement | None = None
        self.timestamp_dates: DataElement | None = None
        self.frequencies: DataElement | None = None
        # sky coordinates
        self.azimuth: DataElement | None = None
        self.elevation: DataElement | None = None
        self.declination: DataElement | None = None
        self.right_ascension: DataElement | None = None

        # climate
        self.temperature: DataElement | None = None
        self.humidity: DataElement | None = None
        self.pressure: DataElement | None = None

        self._scan_tuple_list = self._get_scan_tuple_list(data=data)
        self.set_data_elements(data=data, scan_state=scan_state)

        self.gain_solution: DataElement | None = None

    def __str__(self):
        """ Returns the same `str` as `katdal`. """
        return self._data_str

    def set_data_elements(self, scan_state: ScanStateEnum | None, data: DataSet | None = None):
        """
        Initialises all `DataElement`s for `scan_state` using the element factory. Sets the elements as attributes.
        :param scan_state: the scan state as a `ScanStateEnum`, this is set as an attribute to `self`
        :param data: a `DataSet` object from `katdal`, can be `None`
        """
        if data is None:
            data = self._get_data()
            self._select(data=data)
        self.scan_state = scan_state
        self._element_factory = self._get_data_element_factory()

        self.timestamps = self._element_factory.create(array=data.timestamps[:, np.newaxis, np.newaxis])
        if self.original_timestamps is None:
            self.original_timestamps = copy(self.timestamps)
        self.timestamp_dates = self._element_factory.create(
            array=np.asarray([datetime.fromtimestamp(stamp) for stamp in data.timestamps])[:, np.newaxis, np.newaxis]
        )
        self.frequencies = self._element_factory.create(array=data.freqs[np.newaxis, :, np.newaxis])

        # sky coordinates
        self.azimuth = self._element_factory.create(array=data.az[:, np.newaxis, :])
        self.elevation = self._element_factory.create(array=data.el[:, np.newaxis, :])
        self.declination = self._element_factory.create(array=data.dec[:, np.newaxis, :])
        self.right_ascension = self._element_factory.create(
            array=self._coherent_right_ascension(right_ascension=data.ra)[:, np.newaxis, :]
        )

        # climate
        self.temperature = self._element_factory.create(array=data.temperature[:, np.newaxis, np.newaxis])
        self.humidity = self._element_factory.create(array=data.humidity[:, np.newaxis, np.newaxis])
        self.pressure = self._element_factory.create(array=data.pressure[:, np.newaxis, np.newaxis])

    def load_visibility_flags_weights(self):
        """ Load visibility, flag and weights and set them as attributes to `self`. """
        if self.flags is not None and self.weights is not None and self.visibility is not None:
            print('Visibility, flag and weight data is already loaded.')
            return
        visibility, flags, weights = self._visibility_flags_weights()
        self.visibility = self._element_factory.create(array=visibility)
        if self.flags is not None:
            print('Overwriting existing flags.')
        self.flags = FlagElement(flags=[self._element_factory.create(array=flags)])
        if self.weights is not None:
            print('Overwriting existing weights.')
        self.weights = self._element_factory.create(array=weights)

    def delete_visibility_flags_weights(self):
        """ Delete large arrays from memory, i.e. replace them with `None`. """
        self.visibility = None
        self.flags = None
        self.weights = None

    def antenna(self, receiver) -> Antenna:
        """ Returns the `Antenna` object belonging to `receiver`. """
        return self.antennas[self._antenna_name_list.index(receiver.antenna_name)]

    def antenna_index_of_receiver(self, receiver: Receiver) -> int | None:
        """ Returns the index of the `Antenna` belonging to `receiver`. Returns `None` if it is not found. """
        try:
            return self.antennas.index(self.antenna(receiver=receiver))
        except ValueError:
            return

    def receiver_indices_of_antenna(self, antenna: Antenna) -> list[int] | None:
        """ Returns the indices of the `Receiver`s on `Antenna`. Returns empty `list` if none are found. """
        return [i for i, receiver in enumerate(self.receivers) if receiver.antenna_name == antenna.name]

    def set_gain_solution(self, gain_solution_array: np.ndarray, gain_solution_mask_array: np.ndarray):
        """ Sets the gain solution with data `gain_solution_array` and mask `gain_solution_mask_array`. """
        self.gain_solution = self._element_factory.create(array=gain_solution_array)
        self.flags.add_flag(flag=self._element_factory.create(array=gain_solution_mask_array))

    def corrected_visibility(self) -> DataElement | None:
        """ Returns the gain-corrected visibility data. """
        if self.gain_solution is None:
            print('Gain solution not available.')
            return
        return self.visibility / self.gain_solution

    def _get_data(self) -> DataSet:
        """
        Loads and returns the data from `katdal` for `self._block_name` using either `self._token`
        or if it is `None`, the `self._data_folder`.
        """
        if self._token is not None:
            katdal_open_argument = f'https://archive-gw-1.kat.ac.za/' \
                                   f'{self._block_name}/{self._block_name}_sdp_l0.full.rdb?token={self._token}'
        else:
            katdal_open_argument = os.path.join(
                self._data_folder,
                f'{self._block_name}/{self._block_name}/{self._block_name}_sdp_l0.full.rdb'
            )
        self._katdal_open_argument = katdal_open_argument
        return katdal.open(self._katdal_open_argument)

    def _dumps_of_scan_state(self) -> list[int] | None:
        """
        Returns the dump indices that belong to `self.scan_sate`. If the scan state is `None`, `None` is returned.
        """
        if self.scan_state is None:
            return
        result = []
        for scan_tuple in self._scan_tuple_list:
            if scan_tuple.state == self.scan_state:
                result.extend(scan_tuple.dumps)
        return result

    def _dumps(self) -> list[int]:
        return self._dumps_of_scan_state()

    def _get_data_element_factory(self) -> AbstractDataElementFactory:
        """
        Returns the `DataElementFactory` taken from `self.scan_state`. If `self.scan_state` is None,
        a default factory instance is returned.
        """
        if self.scan_state is None:
            return DataElementFactory()
        return self.scan_state.factory(scan_dumps=self._dumps())

    def _visibility_flags_weights(self, data: DataSet | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns a tuple of visibility, flags and weights as `np.ndarray`s.
        It first looks for a cache file containing these. If that file is unavailabe, incomplete or
        if `self._force_load_from_correlator_data` is `True`, the cache file is created again.
        :param data: optional `katdal` `DataSet`, defaults to `None`
        :return: a tuple of visibility, flags and weights as `np.ndarray` each
        """
        cache_file_directory = os.path.join(ROOT_DIR, 'cache')
        os.makedirs(cache_file_directory, exist_ok=True)
        cache_file = os.path.join(cache_file_directory, self._cache_file_name)
        if not os.path.exists(cache_file) or self._force_load_from_correlator_data:
            if data is None:
                data = katdal.open(self._katdal_open_argument)
                self._select(data=data)
            visibility, flags, weights = self._load_autocorrelation_visibility(data=data)
            if self._do_save_to_disc:
                print(f'Creating cache file for {self.name}...')
                np.savez_compressed(cache_file,
                                    visibility=visibility,
                                    flags=flags,
                                    weights=weights,
                                    correlator_products=data.corr_products)
        else:
            print(f'Loading visibility, flags and weights for {self.name} from cache file...')
            data_from_cache = np.load(cache_file)
            correlator_products = data_from_cache['correlator_products']
            try:  # if this fails it means that the cache file does not contain the correlator products
                correlator_products_indices = self._correlator_products_indices(
                    all_correlator_products=correlator_products
                )
            except ValueError:
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

    def _correlator_products_indices(self, all_correlator_products: np.ndarray) -> Any:
        """
        Returns the indices belonging to the autocorrelation of the input receivers
        relative to `all_correlator_products`.
        """
        result = [np.where(np.prod(all_correlator_products == correlator_product, axis=1))[0]
                  for correlator_product in self.correlator_products]
        result = np.asarray(result, dtype=object)
        if len(result) != len(self.correlator_products) or len(result.shape) != 2 or result.shape[1] == 0:
            raise ValueError(f'Input `all_correlator_products` must contain all receivers.')
        result = np.atleast_1d(np.squeeze(result)).tolist()
        return result

    def _get_correlator_products(self) -> list[list[str, str]]:
        """
        Returns a `list` containing a `list` with the same element twice, namely the `receiver` name,
        for each `receiver` in `self.receivers`.
        """
        return [[str(receiver)] * 2 for receiver in self.receivers]

    def _select(self, data: DataSet):
        """ Run `data._select()` on the correlator products in `self`. """
        data.select(corrprods=self._correlator_products_indices(all_correlator_products=data.corr_products))

    @staticmethod
    def _get_receivers(receivers: list[Receiver] | None, data: DataSet) -> list[Receiver]:
        """ Returns `receivers` unmodified if it is not `None`, otherwise it returns all receivers in `data`. """
        if receivers is not None:
            return receivers
        all_receiver_names = np.unique(data.corr_products.flatten())
        return [Receiver.from_string(receiver_string=name) for name in all_receiver_names]

    @staticmethod
    def _get_scan_tuple_list(data: DataSet) -> list[ScanTuple]:
        """ Returns a `list` containing all `ScanTuple`s for `data`. """
        scan_tuple_list: list[ScanTuple] = []
        for index, state, target in data.scans():
            scan_tuple = ScanTuple(dumps=data.dumps, state=ScanStateEnum.get_enum(state), index=index, target=target)
            scan_tuple_list.append(scan_tuple)
        return scan_tuple_list

    def _coherent_right_ascension(self, right_ascension: np.ndarray) -> np.ndarray:
        """
        Checks if the elements in `right_ascension` are coherent and if yes returns them as is. If not, elements
        at and above 180 degrees are shifted with `self._shift_right_ascension()`.
        """
        for right_ascension_per_dish in right_ascension.T:
            _, cluster_centres = Clustering().split_clusters(feature_vector=right_ascension_per_dish, n_clusters=2)
            if (abs(cluster_centres[1] - cluster_centres[0]) > 180).any():
                return self._shift_right_ascension(right_ascension=right_ascension)
        return right_ascension

    @staticmethod
    def _shift_right_ascension(right_ascension: np.ndarray) -> np.ndarray:
        """ Subtracts 360 from all entries in `right_ascension` that are 180 or higher an returns the result. """
        return np.asarray([[timestamp_ra if timestamp_ra < 180 else timestamp_ra - 360
                            for timestamp_ra in dish_ra]
                           for dish_ra in right_ascension])
