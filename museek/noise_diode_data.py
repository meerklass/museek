from museek.enum.scan_state_enum import ScanStateEnum
from museek.noise_diode import NoiseDiode
from museek.time_ordered_data import TimeOrderedData


class NoiseDiodeData(TimeOrderedData):
    """
    This class handles time ordered data with periodic noise diode firings.
    Timestamps with non-zero noise diode contribution are hidden by default but accessible if needed
    for RFI mitigation or gain calibration.
    """

    def __init__(self, *args, **kwargs):
        """
        Forwards `args` and `kwargs` to the super class.
        Sets an additional attribute `noise_diode` and restricts the dumps of the scan state `SCAN` to
        those with zero noise diode contribution.
        """
        self.noise_diode: NoiseDiode | None = None
        super().__init__(*args, **kwargs)
        self.noise_diode = NoiseDiode(dump_period=self.dump_period, observation_log=self.obs_script_log)
        if self.scan_state == ScanStateEnum.SCAN or self.scan_state == ScanStateEnum.TRACK:
            self.set_data_elements(scan_state=self.scan_state)

    def _dumps(self) -> list[int]:
        """
        Returns the dump indices which have zero noise doide contribution and belong to the `scan_sate` `SCAN`.
        """
        dumps_of_scan_state = self._dumps_of_scan_state()
        if self.scan_state not in [ScanStateEnum.SCAN, ScanStateEnum.TRACK] or self.noise_diode is None:
            return dumps_of_scan_state
        return [i for i in dumps_of_scan_state
                if i in self.noise_diode.get_noise_diode_off_scan_dumps(timestamps=self.timestamps)]
