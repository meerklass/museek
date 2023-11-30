import numpy as np

from museek.data_element import DataElement


class NoiseDiode:
    def __init__(self, dump_period: float, observation_log: list[str]):
        """
        Initialise with `dump_period` and the MeerKAT observation script log `observation_log`.
        """
        self._dump_period = dump_period
        self._observation_log = observation_log
        duration, period, first_set_at = self._get_noise_diode_settings()
        self.duration = duration
        self.period = period
        self.first_set_at = first_set_at
        self.noise_diode_ratios: np.ndarray | None = None

    def get_noise_diode_off_scan_dumps(self, timestamps: DataElement) -> np.ndarray:
        """ Returns a `list` of integer indices of `timestamps` where the noise diode is off. """
        noise_diode_cycle_start_times = self._get_noise_diode_cycle_start_times(timestamps)
        noise_diode_off = self._get_where_noise_diode_is_off(timestamps=timestamps,
                                                             noise_diode_cycle_starts=noise_diode_cycle_start_times)
        return noise_diode_off

    def _get_noise_diode_settings(self) -> tuple[float, float, float]:
        """
        Returns the noise diode settings, the duration it is turned on, the period, i.e. on-time plus off-time and
        the time stamp it is turned on first.
        :raise NotImplementedError: If any of these three settings cannot be read from observation log or calculated.
        :return: a `tuple` of duration of the noise diode turned on, the duration of one cycle or period, and the
                 timestamp of it being turned on.
        """
        duration, period, first_set_at = self._get_noise_diode_settings_from_obs_script()
        if duration is None or period is None or first_set_at is None:
            duration, period, first_set_at = self._get_noise_diode_settings_from_2019_obs_script()
        if duration is None or period is None or first_set_at is None:
            raise NotImplementedError('Noise diode settings could not be read from the observation script log.')
        return duration, period, first_set_at

    def _get_noise_diode_settings_from_obs_script(self) -> tuple[float | None, float | None, float | None]:
        """
        Reads the noise diode settings from `data.obs_script_log` `list` of `string`s.
        This method assumes exact wording in `data.obs_script_log` and will fail if that is not present.
        If reading one of them fails, `None` is returned within the `tuple`.
        :return: a `tuple` of duration of the noise diode turned on, the duration of one cycle or period, and the
                 timestamp of it being turned on. Each can be `None`.
        """
        noise_diode_set_at: float | None = None
        noise_diode_on_duration: float | None = None
        noise_diode_period: float | None = None
        for observation_log_line in self._observation_log:
            if 'INFO' in observation_log_line:
                if (search_string_1 := 'Repeat noise diode pattern every ') in observation_log_line:
                    if (search_string_2 := ' sec on') in observation_log_line:
                        start_index = observation_log_line.index(search_string_1) + len(search_string_1)
                        end_index = observation_log_line.index(search_string_2)
                        string_with_two_floats = observation_log_line[start_index:end_index]
                        string_with_two_floats_split = string_with_two_floats.split(' ')
                        noise_diode_period = float(string_with_two_floats_split[0])
                        noise_diode_on_duration = float(string_with_two_floats_split[-1])
                elif (search_string_1 := 'Report: Switch noise-diode pattern on at ') in observation_log_line:
                    start_index = observation_log_line.index(search_string_1) + len(search_string_1)
                    string_with_one_float = observation_log_line[start_index:]
                    noise_diode_set_at = float(string_with_one_float)
        return noise_diode_on_duration, noise_diode_period, noise_diode_set_at

    def _get_noise_diode_settings_from_2019_obs_script(self) -> tuple[float | None, float | None, float | None]:
        """
        Reads the noise diode settings from `data.obs_script_log` `list` of `string`s.
        This method assumes exact wording in `data.obs_script_log` and will fail if that is not present.
        If reading one of them fails, `None` is returned within the `tuple`.
        :return: a `tuple` of duration of the noise diode turned on, the duration of one cycle or period, and the
                 timestamp of it being turned on. Each can be `None`.
        """
        noise_diode_set_at: float | None = None
        noise_diode_on_duration: float | None = None
        noise_diode_period: float | None = None
        for observation_log_line in self._observation_log:
            if 'INFO' in observation_log_line:
                if (search_string_1 := 'Request noise diode pattern to repeat every ') in observation_log_line:
                    if (search_string_2 := ' sec on') in observation_log_line:
                        start_index = observation_log_line.index(search_string_1) + len(search_string_1)
                        end_index = observation_log_line.index(search_string_2)
                        string_with_two_floats = observation_log_line[start_index:end_index]
                        string_with_two_floats_split = string_with_two_floats.split(' ')
                        noise_diode_period = float(string_with_two_floats_split[0])
                        noise_diode_on_duration = float(string_with_two_floats_split[-1])
            elif 'WARNING' in observation_log_line:
                if (search_string_1 := 'Set noise diode pattern to activate at ') in observation_log_line:
                    start_index = observation_log_line.index(search_string_1) + len(search_string_1)
                    end_index = observation_log_line.index(', with ')
                    string_with_one_float = observation_log_line[start_index:end_index]
                    noise_diode_set_at = float(string_with_one_float)
        return noise_diode_on_duration, noise_diode_period, noise_diode_set_at

    def _get_where_noise_diode_is_off(
            self,
            timestamps: DataElement,
            noise_diode_cycle_starts: np.ndarray,
    ) -> np.ndarray:
        """
        Returns the timestamp indices where the noise diode is entirely off.
        :param timestamps: the output indices are relative to these timestamps
        :param noise_diode_cycle_starts: `array` of noise diode cycle starting timestamps
        :return: `array` on integer timestamp indices where the noise diode is off
        """
        self._set_noise_diode_ratios(timestamps=timestamps,
                                     noise_diode_cycle_starts=noise_diode_cycle_starts,
                                     dump_period=self._dump_period)
        return np.where(self.noise_diode_ratios == 0)[0]

    def _set_noise_diode_ratios(self,
                                timestamps: DataElement,
                                noise_diode_cycle_starts: np.ndarray,
                                dump_period: float,
                                force: bool = False) \
            -> np.ndarray:
        """
        Sets `self.noise_diode_ratios`, an `array` of floats between 0 and 1 indicating the relative duration 
        of noise diode firing within the timestamp. If it already exists nothing is done, unless `force` is `True`.
        :param timestamps: the output indices are relative to these timestamps
        :param noise_diode_cycle_starts: `array` of noise diode cycle starting timestamps
        :param dump_period: timestamp duration
        :param force: if `True` the calculation is repeated
        """
        if force or self.noise_diode_ratios is None:
            timestamp_array = timestamps.squeeze
            noise_diode_ratios = np.zeros_like(timestamp_array, dtype=float)

            for noise_diode_cycle_start_time in noise_diode_cycle_starts:
                gap_list = abs(noise_diode_cycle_start_time - timestamp_array)
                dump_closest_timestamp = np.argmin(gap_list)
                gap_closest_timestamp = gap_list[dump_closest_timestamp]

                if gap_closest_timestamp <= dump_period / 2.:
                    timestamp_edge = timestamp_array[dump_closest_timestamp] + dump_period / 2
                    cycle_start_to_timestamp_edge = timestamp_edge - noise_diode_cycle_start_time
                    if cycle_start_to_timestamp_edge >= self.duration:  # diode in one dump
                        noise_diode_ratios[dump_closest_timestamp] = self.duration / dump_period
                    elif cycle_start_to_timestamp_edge < self.duration:  # diode in two dumps
                        noise_diode_ratios[dump_closest_timestamp] = cycle_start_to_timestamp_edge / dump_period
                        if dump_closest_timestamp + 1 < len(timestamp_array):
                            noise_diode_ratios[dump_closest_timestamp + 1] = self.duration / dump_period \
                                - noise_diode_ratios[dump_closest_timestamp]
            self.noise_diode_ratios = noise_diode_ratios

    def _get_noise_diode_cycle_start_times(self, timestamps: DataElement) -> np.ndarray:
        """
        Returns an array with the timestamps of each cycle's start within the array `timestamps`.
        :raise ValueError: if `self.first_set_at` is after the end of `timestamps`
        """
        timespan = timestamps[-1] - self.first_set_at
        if timespan < 0:
            raise ValueError('Input `timestamps` end before the noise diode was first started.')
        if timespan < self.period:
            return np.array([self.first_set_at])
        noise_diode_cycle_starts = np.arange(timespan // self.period) * self.period + self.first_set_at
        return noise_diode_cycle_starts
