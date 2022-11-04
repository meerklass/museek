import os
from typing import Optional

import katdal
import numpy as np
from katdal import DataSet
from katdal.lazy_indexer import DaskLazyIndexer

MODULE_ROOT = os.path.dirname(__file__)


class TimeOrderedDataElement(np.ndarray):
    def __new__(cls, array):
        if not isinstance(array, np.ndarray):
            raise ValueError('')
        if len(array.shape) != 3:
            raise ValueError('')
        object = np.asarray(array).view(cls)
        return object


class TimeOrderedData:
    def __init__(self,
                 block_name: str,
                 token: Optional[str],
                 data_folder: Optional[str],
                 force_load_from_correlator_data: bool = False,
                 do_save_to_disc: bool = True):
        data = self.load_data(block_name=block_name, data_folder=data_folder, token=token)
        data.select(corrprods='auto')
        self.shape = data.shape
        self.name = data.name
        self._force_load_from_correlator_data = force_load_from_correlator_data
        self._do_save_to_disc = do_save_to_disc

        visibility, flags, weights = self.visibility_flags_weights(data=data)
        self.visibility = TimeOrderedDataElement(visibility.real)
        self.flags = TimeOrderedDataElement(flags)
        self.weights = TimeOrderedDataElement(weights)

    def visibility_flags_weights(self, data):
        cache_file_directory = os.path.join(MODULE_ROOT, '../cache')
        os.makedirs(cache_file_directory, exist_ok=True)
        cache_file_name = f'{data.name}_auto_visibility_flags_weights.npz'
        cache_file = os.path.join(cache_file_directory, cache_file_name)
        if not os.path.exists(cache_file) or self._force_load_from_correlator_data:
            visibility, flags, weights = self.extract_autocorrelation_visibility(data=data)
            if self._do_save_to_disc:
                np.savez_compressed(cache_file, visibility=visibility, flags=flags, weights=weights)
        else:
            data_from_cache = np.load(cache_file)
            visibility = data_from_cache['visibility']
            flags = data_from_cache['flags']
            weights = data_from_cache['weights']
        return visibility, flags, weights

    def extract_autocorrelation_visibility(self, data: DataSet):
        visibility = np.zeros(shape=self.shape, dtype=complex)
        flags = np.zeros(shape=self.shape, dtype=complex)
        weights = np.zeros(shape=self.shape, dtype=complex)
        DaskLazyIndexer.get(arrays=[data.vis, data.flags, data.weights],
                            keep=...,
                            out=[visibility, flags, weights])
        return visibility, flags, weights

    @staticmethod
    def load_data(block_name: str, token: Optional[str], data_folder: Optional[str]) -> DataSet:
        if token is not None:
            data = katdal.open(
                f'https://archive-gw-1.kat.ac.za/{block_name}/{block_name}_sdp_l0.full.rdb?{token}'
            )
        else:
            data = katdal.open(os.path.join(data_folder,
                                            f'{block_name}/{block_name}/{block_name}_sdp_l0.full.rdb'))
        return data
