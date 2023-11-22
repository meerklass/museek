from datetime import datetime, timedelta
import ephem


class TimeAnalysis:
    """ Class to do time-related analysis. """

    def __init__(self, latitude: str, longitude: str):
        """
        Initialise
        :param latitude: latitude of the telescope
        :param longitude: longitude of the telescope
        """
        self.latitude = latitude
        self.longitude = longitude

    def time_difference_to_sunset_sunrise(
        self,
        obs_start: datetime,
        obs_end: datetime,
        utcoffset: float
    ) -> tuple[datetime, datetime, float, float]:
        """
        Calculate the closeness between start/end time and sunset/sunrise time
        :param obs_start: the start time of whole observation
        :param obs_end: the end time of whole observation
        :param utcoffset: float, offset between local time and the UTC in hours
        :return: `tuple` of 
                - sunset_start.datetime(), sunrise_end.datetime() : datetime object
                - the nearest sunset/sunrise time before/after observation started/ended
                - end_sunrise_diff, start_sunset_diff : float [seconds] 
                - the time difference between end/start and sunrise/sunset in float [seconds]

        Notes
        -----
        When observations start before sunset or end after sunrise, 
        previous_setting() or next_rising() would give you the last or next day's sunset or sunrise date
        """

        observer = ephem.Observer()
        observer.lat = self.latitude
        observer.lon = self.longitude

        # Set the observer's date (in local time)
        obs_start_local = obs_start + timedelta(hours=utcoffset)
        obs_end_local = obs_end + timedelta(hours=utcoffset)

        # Calculate sunset and sunrise times for start time / end time (in UTC)
        observer.date = obs_start_local
        sunset_start = observer.previous_setting(ephem.Sun())

        observer.date = obs_end_local
        sunrise_end = observer.next_rising(ephem.Sun())

        # Calculate time differences
        end_sunrise_diff = (obs_end - sunrise_end.datetime()).total_seconds()
        start_sunset_diff = (obs_start - sunset_start.datetime()).total_seconds()

        return sunset_start.datetime(), sunrise_end.datetime(), end_sunrise_diff, start_sunset_diff
