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
        obs_end: datetime
    ) -> tuple[datetime, datetime, float]:
        """
        Calculate the closeness between start/end time and sunset/sunrise time
        :param obs_start: the start time of whole observation
        :param obs_end: the end time of whole observation
        :return: `tuple` of 
                - sunset_start.datetime(), sunrise_end.datetime() : datetime object
                - the nearest sunset/sunrise time before/after observation started/ended
                - sunrise_end_diff, start_sunset_diff : float [seconds] 
                - the time difference between sunrise/start and end/sunset in float [seconds]

        Notes
        -----
        When observations start before sunset or end after sunrise, 
        previous_setting() or next_rising() would give you the last or next day's sunset or sunrise date
        """

        observer = ephem.Observer()
        observer.lat = self.latitude
        observer.lon = self.longitude

        # Calculate closest sunset and sunrise times for start time / end time (in UTC)
        observer.date = obs_start
        if abs(obs_start - observer.previous_setting(ephem.Sun()).datetime()).total_seconds() > \
           abs(obs_start - observer.next_setting(ephem.Sun()).datetime()).total_seconds():
            sunset_start = observer.next_setting(ephem.Sun())
        else:
            sunset_start = observer.previous_setting(ephem.Sun())

        observer.date = obs_end
        if abs(obs_end - observer.previous_rising(ephem.Sun()).datetime()).total_seconds() > \
           abs(obs_end - observer.next_rising(ephem.Sun()).datetime()).total_seconds():
            sunrise_end = observer.next_rising(ephem.Sun())
        else:
            sunrise_end = observer.previous_rising(ephem.Sun())


        # Calculate time differences
        sunrise_end_diff = (sunrise_end.datetime() - obs_end).total_seconds()
        start_sunset_diff = (obs_start - sunset_start.datetime()).total_seconds()

        return sunset_start.datetime(), sunrise_end.datetime(), sunrise_end_diff, start_sunset_diff
