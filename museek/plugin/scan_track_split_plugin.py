import os
from copy import deepcopy

from definitions import ROOT_DIR
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

    def __init__(self, do_delete_unsplit_data: bool, do_store_context: bool):
        """
        Initialise with `do_delete_unsplit_data`, a switch that determines wether the object containing the entire
        data should be deleted to save memory.
        :param do_delete_unsplit_data: switch that determines wether the data should be deleted after split
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
        """
        super().__init__()
        self.do_delete_unsplit_data = do_delete_unsplit_data
        self.do_store_context = do_store_context

    def set_requirements(self):
        """ Only requirement is the data. """
        self.requirements = [Requirement(location=ResultEnum.DATA, variable='data'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name')]

    def run(self, data: TimeOrderedData, block_name: str):
        """
        Split `data` into scanning and tracking part and save.
        If `self.do_delete_unsplit_data` is `True`, `data` is deleted.
        :param data: the complete time ordered data
        :param block_name: name of the observation block
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

        if self.do_store_context:
            context_file_name = 'scan_track_split_plugin.pickle'
            context_folder = os.path.join(ROOT_DIR, 'results/')
            context_directory = os.path.join(context_folder, f'{block_name}/')
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=context_directory)
