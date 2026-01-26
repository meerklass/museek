from datetime import datetime, timezone

import ephem


class TimeAnalysis:
    """Class to do time-related analysis."""

    def __init__(self, latitude: str, longitude: str):
        """
        Initialise
        :param latitude: latitude of the telescope
        :param longitude: longitude of the telescope
        """
        self.latitude = latitude
        self.longitude = longitude

    def time_difference_to_sunset_sunrise(
        self, obs_start: datetime, obs_end: datetime
    ) -> tuple[datetime, datetime, float, float]:
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

        Notes
        -----
        This function normalises input datetimes to UTC for deterministic behaviour across systems.
        If `obs_start` or `obs_end` is timezone-aware it will be converted to UTC; naive datetimes
        are assumed to be UTC.
        """

        # Normalize inputs to timezone-aware UTC datetimes
        if obs_start.tzinfo is None:
            obs_start_utc = obs_start.replace(tzinfo=timezone.utc)
        else:
            obs_start_utc = obs_start.astimezone(timezone.utc)

        if obs_end.tzinfo is None:
            obs_end_utc = obs_end.replace(tzinfo=timezone.utc)
        else:
            obs_end_utc = obs_end.astimezone(timezone.utc)
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

        # Use UTC naive datetimes for ephem observer.date
        observer.date = obs_start_utc.replace(tzinfo=None)
        if abs(
            (
                obs_start_utc
                - observer.previous_setting(ephem.Sun())
                .datetime()
                .replace(tzinfo=timezone.utc)
            ).total_seconds()
        ) > abs(
            (
                obs_start_utc
                - observer.next_setting(ephem.Sun())
                .datetime()
                .replace(tzinfo=timezone.utc)
            ).total_seconds()
        ):
            sunset_start = observer.next_setting(ephem.Sun())
        else:
            sunset_start = observer.previous_setting(ephem.Sun())

        observer.date = obs_end_utc.replace(tzinfo=None)
        if abs(
            (
                obs_end_utc
                - observer.previous_rising(ephem.Sun())
                .datetime()
                .replace(tzinfo=timezone.utc)
            ).total_seconds()
        ) > abs(
            (
                obs_end_utc
                - observer.next_rising(ephem.Sun())
                .datetime()
                .replace(tzinfo=timezone.utc)
            ).total_seconds()
        ):
            sunrise_end = observer.next_rising(ephem.Sun())
        else:
            sunrise_end = observer.previous_rising(ephem.Sun())

        # Calculate time differences using timezone-aware UTC datetimes
        sunrise_end_dt = sunrise_end.datetime().replace(tzinfo=timezone.utc)
        start_sunset_dt = sunset_start.datetime().replace(tzinfo=timezone.utc)

        sunrise_end_diff = (sunrise_end_dt - obs_end_utc).total_seconds()
        start_sunset_diff = (obs_start_utc - start_sunset_dt).total_seconds()

        return (
            sunset_start.datetime(),
            sunrise_end.datetime(),
            sunrise_end_diff,
            start_sunset_diff,
        )
