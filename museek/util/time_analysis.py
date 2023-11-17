import numpy as np
from datetime import datetime, timedelta
import ephem
from museek.time_ordered_data import TimeOrderedData

class TimeAnalysis:
    """ Class to do time-related analysis. """

    def __init__(self, data: TimeOrderedData):
        """
        Initialise
        :param data: the analysed `TimeOrderedData` object
        """
        self.obs_start = datetime.utcfromtimestamp(float(data.original_timestamps[0])) 
        self.obs_end = datetime.utcfromtimestamp(float(data.original_timestamps[-1])) 
        self.lat = data.antennas[0].ref_observer.lat
        self.long = data.antennas[0].ref_observer.long

    def time_difference_to_sunset_sunrise(self, timezone):

        """
        Calculate the closeness between start/end time and sunset/sunrise time
        :param timezone: float, the time zone where the observer is located
        """

        observer = ephem.Observer()
        observer.lat = self.lat
        observer.lon = self.long

        # Set the observer's date (in local time)
        obs_start_local = self.obs_start + timedelta(hours=timezone)
        obs_end_local = self.obs_end + timedelta(hours=timezone)

        # Calculate sunset and sunrise times for start time / end time (in UTC)
        observer.date = obs_start_local
        sunset_time_start = observer.previous_setting(ephem.Sun())
        sunrise_time_start = observer.next_rising(ephem.Sun())

        observer.date = obs_end_local
        sunset_time_end = observer.previous_setting(ephem.Sun())
        sunrise_time_end = observer.next_rising(ephem.Sun())

        # Calculate time differences
        end_to_sunrise_diff = (self.obs_end - sunrise_time_end.datetime()).total_seconds()
        start_to_sunset_diff = (self.obs_start - sunset_time_start.datetime()).total_seconds()

        return sunset_time_start.datetime(), sunrise_time_end.datetime(), end_to_sunrise_diff, start_to_sunset_diff

