import os
from datetime import datetime

from definitions import ROOT_DIR
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData


class ReplaceTrackPlugin(AbstractPlugin):
    """ Plugin to load replacement Track data from another block. """

    def __init__(self,
                 replacement_track_block_name: str | None,
                 token: str | None,
                 data_folder: str | None,
                 force_load_from_correlator_data: bool,
                 do_save_visibility_to_disc: bool):
        """
        Initialise the plugin.
        
        :param replacement_track_block_name: the name of the replacement block to use, usually an integer timestamp as string
        :param token: token to access the SARAO archive
        :param data_folder: if `token` is `None`, data will be loaded from a local `data_folder`
        :param force_load_from_correlator_data: if this is `True` the cache files are ignored
        :param do_save_visibility_to_disc: if `True` the visibilities, flags and weights are stored to disc as cache
        """
        super().__init__()
        self.replacement_track_block_name = replacement_track_block_name
        self.token = token
        self.data_folder = data_folder
        self.force_load_from_correlator_data = force_load_from_correlator_data
        self.do_save_visibility_to_disc = do_save_visibility_to_disc
        

    def set_requirements(self):
        """Requires block_name and receivers to have been set, to match InPlugin, and track_data to exist already (to avoid risk of ScanTrackSplitPlugin being run afterwards and undoing the replacement)."""
        self.requirements = [Requirement(location=ResultEnum.RECEIVERS, variable='receivers'),
                             Requirement(location=ResultEnum.BLOCK_NAME, variable='block_name'),
                             Requirement(location=ResultEnum.TRACK_DATA, variable='track_data')]

    def run(self, receivers: list, block_name: str):
        """
        Loads the complete data from another block as `TimeOrderedData`, 
        extracts the Track data, and sets it as a result.
        
        :param receivers: list of receivers
        :param block_name: name of the current (not replacement) observation block
        """
        actual_track_source_block = self.replacement_track_block_name
        
        # Check block_name vs current block name
        if self.replacement_track_block_name is None:
            # No replacement block specified; do nothing with the data
            actual_track_source_block = block_name
        else:
            # Load data from alternative block
            # FIXME: Can this be done selectively to avoid loading so much non-Track data?
            replacement_track_data = TimeOrderedData(
                token=self.token,
                data_folder=self.data_folder,
                block_name=self.replacement_track_block_name,
                receivers=receivers,
                force_load_from_correlator_data=self.force_load_from_correlator_data,
                do_create_cache=self.do_save_visibility_to_disc
            )
            
            # Select only Track data
            replacement_track_data.set_data_elements(scan_state=ScanStateEnum.TRACK)
        
            # This should overwrite the existing track_data result
            self.set_result(result=Result(location=ResultEnum.TRACK_DATA, result=replacement_track_data, allow_overwrite=True))
        
        # Keep a result containing the actual source of the track data now being used
        self.set_result(result=Result(location=ResultEnum.TRACK_DATA_ACTUAL_SOURCE_BLOCK_NAME, result=actual_track_source_block, allow_overwrite=True))
        
