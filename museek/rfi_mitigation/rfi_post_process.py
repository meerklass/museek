import numpy as np
from scipy import ndimage

from museek.factory.data_element_factory import FlagElementFactory
from museek.flag_element import FlagElement


class RfiPostProcess:
    """ Class to post-process rfi masks. """

    def __init__(self, new_flag: FlagElement, initial_flag: FlagElement | None, struct_size: tuple[int, int]):
        """
        Initialise the post-processing of RFI flags.
        :param new_flag: newly generated RFI flag
        :param initial_flag: initial flags the RFI flags were built upon
        :param struct_size: structure size for binary dilation, closing etc
        """
        self._flag = new_flag
        self._initial_flag = initial_flag
        self._struct_size = struct_size
        self._struct = np.ones((self._struct_size[0], self._struct_size[1]), dtype=bool)
        self._factory = FlagElementFactory()

    def get_flag(self):
        """ Return the flag. """
        return self._flag

    def binary_mask_dilation(self):
        """ Dilate the mask. """
        if self._initial_flag is not None:
            to_dilate = self._flag.squeeze ^ self._initial_flag.squeeze
        else:
            to_dilate = self._flag.squeeze
        dilated = ndimage.binary_dilation(to_dilate,
                                          structure=self._struct,
                                          iterations=5)
        self._flag = self._factory.create(array=dilated[:, :, np.newaxis])

    def binary_mask_closing(self):
        """ Close the mask. """
        closed = ndimage.binary_closing(self._flag.squeeze, structure=self._struct, iterations=5)
        self._flag = self._factory.create(array=closed[:, :, np.newaxis])

    def flag_all_channels(self, channel_flag_threshold: float):
        """ If the fraction of flagged channels exceeds `channel_flag_threshold`, all channels are flagged. """
        flagged_fraction = self._flag.sum(axis=1).squeeze / self._flag.shape[1]
        timestamps_to_flag = np.where(flagged_fraction > channel_flag_threshold)[0]
        flag = self._flag._array
        flag[timestamps_to_flag] = True
        self._flag = self._factory.create(array=flag)

    def flag_all_time_dumps(self, time_dump_flag_threshold: float):
        """ If the fraction of flagged time dumps exceeds `time_dump_flag_threshold`, all time dumps are flagged. """
        flagged_fraction = self._flag.sum(axis=0).squeeze / self._flag.shape[0]
        channels_to_flag = np.where(flagged_fraction > time_dump_flag_threshold)[0]
        flag = self._flag._array
        flag[:, channels_to_flag, :] = True
        self._flag = self._factory.create(array=flag)
