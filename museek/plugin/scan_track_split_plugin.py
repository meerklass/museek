from copy import deepcopy

from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enum.result_enum import ResultEnum
from museek.enum.scan_state_enum import ScanStateEnum
from museek.time_ordered_data import TimeOrderedData


class ScanTrackSplitPlugin(AbstractPlugin):
    """
    Plugin to split the scanning and tracking part from the data.
    For the scanning part and calibrator tracking parts new `TimeOrderedData` objects are created.
    """

    def __init__(self, do_delete_unsplit_data: bool):
        """
        Initialise with `do_delete_unsplit_data`, a switch that determines wether the object containing the entire
        data should be deleted to save memory.
        """
        super().__init__()
        self.do_delete_unsplit_data = do_delete_unsplit_data

    def set_requirements(self):
        """ Only requirement is the data. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data')]

    def run(self, data: TimeOrderedData):
        """
        Split `data` into scanning and tracking part and save.
        If `self.do_delete_unsplit_data` is `True`, `data` is deleted.
        """
        scan_data = deepcopy(data)
        scan_data.set_data_elements(scan_state=ScanStateEnum.SCAN)

        track_data = deepcopy(data)
        track_data.set_data_elements(scan_state=ScanStateEnum.TRACK)

        if self.do_delete_unsplit_data:
            data = None
            self.set_result(result=Result(location=ResultEnum.DATA, result=data, allow_overwrite=True))

        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=track_data, allow_overwrite=True))
