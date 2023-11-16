from ivory.plugin.abstract_plugin import AbstractPlugin
from ivory.utils.requirement import Requirement
from ivory.utils.result import Result
from museek.enums.result_enum import ResultEnum
from museek.flag_factory import FlagFactory
from museek.time_ordered_data import TimeOrderedData


class PointSourceFlaggerPlugin(AbstractPlugin):
    """ Plugin to calculate TOD masks for point sources. """

    def __init__(self, point_source_file_path: str, angle_threshold: float):
        """
        Initialise the plugin
        :param point_source_file_path: path to the point source location file
        :param angle_threshold: dumps in the TOD closer than this threshold to a point source are masked
        """
        super().__init__()
        self.point_source_file_path = point_source_file_path
        self.angle_threshold = angle_threshold

    def set_requirements(self):
        """ Set the requirements. """
        self.requirements = [Requirement(location=ResultEnum.SCAN_DATA, variable='scan_data')]

    def run(self, scan_data: TimeOrderedData):
        """ Run the plugin and calculate the TOD masks for point sources in the footprint of `scan_data`. """

        scan_data.load_visibility_flags_weights()
        point_source_mask = FlagFactory().get_point_source_mask(shape=scan_data.visibility.shape,
                                                                receivers=scan_data.receivers,
                                                                right_ascension=scan_data.right_ascension,
                                                                declination=scan_data.declination,
                                                                point_source_file_path=self.point_source_file_path,
                                                                angle_threshold=self.angle_threshold)
        scan_data.flags.add_flag(point_source_mask)
        self.set_result(result=Result(location=ResultEnum.SCAN_DATA, result=scan_data))
