import os
from datetime import datetime

from definitions import ROOT_DIR
from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.receiver import Receiver
from museek.time_ordered_data import TimeOrderedData


class InPlugin(AbstractPlugin):
    """ Plugin to load data and to set output paths. """

    def __init__(self,
                 block_name: str,
                 receiver_list: list[str] | None,
                 token: str | None,
                 data_folder: str | None,
                 force_load_from_correlator_data: bool,
                 do_save_visibility_to_disc: bool,
                 do_store_context: bool,
                 context_folder: str | None):
        """
        Initialise the plugin.
        :param block_name: the name of the block, usually an integer timestamp as string
        :param receiver_list: the list of receivers to consider, if `None`, all available receivers are used
        :param token: token to access the SARAO archive
        :param data_folder: if `token` is `None`, data will be loaded from a local `data_folder`
        :param force_load_from_correlator_data: if this is `True` the cache files are ignored
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
        self.force_load_from_correlator_data = force_load_from_correlator_data
        self.do_save_visibility_to_disc = do_save_visibility_to_disc
        self.do_store_context = do_store_context

        self.context_folder = context_folder
        if self.context_folder is None:
            self.context_folder = os.path.join(ROOT_DIR, 'results/')
        self.check_context_folder_exists()

    def set_requirements(self):
        """ First plugin, no requirements. """
        pass

    def run(self):
        """
        Loads the complete data and the scanning part as `TimeOrderedData` and sets it as a result.
        Depending on the config, will store the context to hard disc with visibility loaded.
        """
        print(f'Working on block {self.block_name}...')
        receivers = None
        if self.receiver_list is not None:
            receivers = [Receiver.from_string(receiver_string=receiver) for receiver in self.receiver_list]
        data = TimeOrderedData(
            token=self.token,
            data_folder=self.data_folder,
            block_name=self.block_name,
            receivers=receivers,
            force_load_from_correlator_data=self.force_load_from_correlator_data,
            do_create_cache=self.do_save_visibility_to_disc,
        )

        # observation date from file name
        observation_date = datetime.fromtimestamp(int(data.name.split('_')[0]))

        if self.do_store_context:
            # to create cache file
            data.load_visibility_flags_weights()
            data.delete_visibility_flags_weights()

            context_file_name = 'in_plugin.pickle'
            context_directory = os.path.join(self.context_folder, f'{self.block_name}/')
            os.makedirs(context_directory, exist_ok=True)

            self.store_context_to_disc(context_file_name=context_file_name,
                                       context_directory=context_directory)

        self.set_result(result=Result(location=ResultEnum.DATA, result=data))
        self.set_result(result=Result(location=ResultEnum.RECEIVERS, result=receivers))
        self.set_result(result=Result(location=ResultEnum.OBSERVATION_DATE, result=observation_date))
        self.set_result(result=Result(location=ResultEnum.BLOCK_NAME, result=self.block_name))

    def check_context_folder_exists(self):
        """ Raises a `ValueError` if `self.context_folder` does not exist. """
        if not os.path.exists(self.context_folder):
            raise ValueError(f'The output folder does not exists: {self.context_folder}')
