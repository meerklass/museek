import os
from enum import Enum
from typing import Optional, NamedTuple

import katdal
import numpy as np
from katdal import DataSet
from katpoint import Target, Antenna

from museek.receiver import Receiver

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
    Class for easy handling of time ordered data as provided by `katdal`.
    """

    def __init__(self,
                 block_name: str,
                 receivers: list[Receiver],
                 token: Optional[str],
                 data_folder: Optional[str],
                 force_load_from_correlator_data: bool = False,
                 do_save_to_disc: bool = True):
        self.scan_dumps: list[int] | None = None
        self.track_dumps: list[int] | None = None
        self.slew_dumps: list[int] | None = None
        self.stop_dumps: list[int] | None = None

        self._katdal_open_argument: Optional[str] = None
        self._force_load_from_correlator_data = force_load_from_correlator_data
        self._do_save_to_disc = do_save_to_disc

        self.receivers = receivers
        self.correlator_products = self._get_correlator_products()

        data = self.load_data(block_name=block_name, data_folder=data_folder, token=token)
        self.all_antennas = data.ants
        data.select(corrprods=self._correlator_products_indices(all_correlator_products=data.corr_products))
        self._data_str = str(data)

        self.obs_script_log = data.obs_script_log
        self.shape = data.shape
        self.name = data.name
        self.dump_period = data.dump_period
        self.antennas = data.ants
        self._antenna_name_list = [antenna.name for antenna in self.antennas]

        self._scan_tuple_list = self._get_scan_tuple_list(data=data)
        self._set_scan_state_dumps()

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
        result = np.asarray(result)
        if len(result) != len(self.correlator_products) or len(result.shape) != 2 or result.shape[1] == 0:
            raise ValueError('Input `all_correlator_products` must contain all receivers.')
        return list(np.squeeze(result))

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
