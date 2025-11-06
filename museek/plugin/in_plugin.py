import os
from museek.definitions import ROOT_DIR
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData
from museek.util.report_writer import ReportWriter
from museek.util.tools import flag_percent_recv, git_version_info
import datetime


class InPlugin(AbstractPlugin):
    """ Plugin to load data and to set output paths. """

    def __init__(self,
                 block_name: str,
                 receiver_list: list[str] | None,
                 token: str | None,
                 data_folder: str | None,
                 force_load_auto_from_correlator_data: bool,
                 force_load_cross_from_correlator_data: bool,
                 do_save_visibility_to_disc: bool,
                 do_store_context: bool,
                 context_folder: str | None):
        """
        Initialise the plugin.
        :param block_name: the name of the block, usually an integer timestamp as string
        :param receiver_list: the list of receivers to consider, if `None`, all available receivers are used
        :param token: token to access the SARAO archive
        :param data_folder: if `token` is `None`, data will be loaded from a local `data_folder`
        :param force_load_auto_from_correlator_data: if this is `True` the cache files are ignored
        :param force_load_cross_from_correlator_data: if this is `True` the cache files are ignored
        :param do_save_visibility_to_disc: if `True` the visibilities, flags and weights are stored to disc as cache
        :param do_store_context: if `True` the context is stored to disc after finishing the plugin
                                 if `True` it is recommended to also have `do_save_visibility_to_disc` set to `True`
        :param context_folder: the context is stored to this directory after finishing the plugin, if `None`, a
                                  default directory is chosen
        """
        super().__init__()
        self.block_name = block_name
        self.receiver_list = receiver_list
        self.token = token
        self.data_folder = data_folder
        self.force_load_auto_from_correlator_data = force_load_auto_from_correlator_data
        self.force_load_cross_from_correlator_data = force_load_cross_from_correlator_data
        self.do_save_visibility_to_disc = do_save_visibility_to_disc
        self.do_store_context = do_store_context
        self.report_file_name = 'flag_report.md'

        self.context_folder = context_folder
        if self.context_folder is None:
            self.context_folder = os.path.join(ROOT_DIR, 'results/')
        #self.check_context_folder_exists()

    def set_requirements(self):
        """ First plugin, no requirements. """
        pass

    def run(self):
        """
        Loads the complete data and the scanning part as `TimeOrderedData` and sets it as a result.
        Depending on the config, will store the context to hard disc with visibility loaded.
        """
        receivers = None
        if self.receiver_list is not None:
            receivers = [Receiver.from_string(receiver_string=receiver) for receiver in self.receiver_list]
        data = TimeOrderedData(
            token=self.token,
            data_folder=self.data_folder,
            block_name=self.block_name,
            receivers=receivers,
            force_load_auto_from_correlator_data=self.force_load_auto_from_correlator_data,
            force_load_cross_from_correlator_data=self.force_load_cross_from_correlator_data,
            do_create_cache=self.do_save_visibility_to_disc,
        )

        # observation date from file name
        observation_date = datetime.datetime.fromtimestamp(int(data.name.split('_')[0]))
        context_directory = os.path.join(self.context_folder, f'{self.block_name}/')
        os.makedirs(context_directory, exist_ok=True)

        flag_report_writer = ReportWriter(output_path=context_directory,
                                         report_name=self.report_file_name,
                                         data_name=self.block_name,
                                         plugin_name=self.name)

        flag_name_list = ['SARAO']

        branch, commit = git_version_info()
        current_datetime = datetime.datetime.now()
        lines = ['...........................', 'Running InPlugin with '+f"MuSEEK version: {branch} ({commit})", ' Finished at ' + current_datetime.strftime("%Y-%m-%d %H:%M:%S")]
        flag_report_writer.write_to_report(lines)

        if self.do_store_context:

            # to create cache file
            data.load_visibility_flags_weights(polars='auto')
            data.delete_visibility_flags_weights(polars='auto')

            context_file_name = 'in_plugin.pickle'
            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=context_directory)

        self.set_result(result=Result(location=ResultEnum.FLAG_REPORT_WRITER, result=flag_report_writer, allow_overwrite=True))
        self.set_result(result=Result(location=ResultEnum.DATA, result=data))
        self.set_result(result=Result(location=ResultEnum.RECEIVERS, result=receivers))
        self.set_result(result=Result(location=ResultEnum.OBSERVATION_DATE, result=observation_date))
        self.set_result(result=Result(location=ResultEnum.BLOCK_NAME, result=self.block_name))
        self.set_result(result=Result(location=ResultEnum.OUTPUT_PATH, result=context_directory))
        self.set_result(result=Result(location=ResultEnum.FLAG_NAME_LIST, result=flag_name_list))


    def check_context_folder_exists(self):
        """ Raises a `ValueError` if `self.context_folder` does not exist. """
        if not os.path.exists(self.context_folder):
            raise ValueError(f'The output folder does not exists: {self.context_folder}')
