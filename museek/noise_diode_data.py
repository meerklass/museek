from typing import Optional

from museek.noise_diode import NoiseDiode
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData


class NoiseDiodeData(TimeOrderedData):
    """
    This class handles time ordered data with periodic noise diode firings.
    Timestamps with non-zero noise diode contribution are hidden by default but accessible if needed
    for RFI mitigation or gain calibration.
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
        super().__init__(block_name=block_name,
                         receivers=receivers,
                         token=token,
                         data_folder=data_folder,
                         force_load_from_correlator_data=force_load_from_correlator_data,
                         do_create_cache=do_create_cache)
        self.scan_dumps = [self.scan_dumps[i] for i in NoiseDiode(data=self).get_noise_diode_off_scan_dumps()]
