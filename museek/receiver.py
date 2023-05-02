from enum import Enum

import numpy as np


class Polarisation(Enum):
    """ Enum helper to contain the two polarisations"""
    v = 'v'
    h = 'h'


class Receiver:
    """ Helper class to contain receiver related infos. """

    def __init__(self, antenna_number: int, polarisation: Polarisation):
        """ Initializes a `Receiver` with the `antenna_index` of the dish and `polarisation`. """
        self.antenna_number = antenna_number
        self.antenna_name = f'm{self.antenna_number:03d}'
        self._polarisation_enum = polarisation
        self.polarisation = self._polarisation_enum.name

    def __str__(self) -> str:
        """ Overload `str` """
        return self.name

    def __eq__(self, other) -> bool:
        """ Overload equality """
        return self.name == other.name

    @property
    def name(self) -> str:
        """ Returns the string name of `self`. """
        return f'{self.antenna_name}{self.polarisation}'

    @classmethod
    def from_string(cls, receiver_string: str):
        """
        Initializes a `Receiver` starting with an identifying string, e.g. 'm063v'.
        :raise ValueError: if the input `receiver_string` does not conform
        """
        if (not receiver_string.startswith('m')
                or not len(receiver_string) == 5
                or not receiver_string[-1] in ['h', 'v']):
            raise ValueError(f'Input `receiver_string` needs to be like e.g. "m063v", got {receiver_string}.')
        return cls(antenna_number=int(receiver_string[1:-1]), polarisation=Polarisation(receiver_string[-1]))

    def antenna_index(self, receivers: list['Receiver']):
        return self.receivers_to_antennas(receivers=receivers).index(self.antenna_name)

    @staticmethod
    def receivers_to_antennas(receivers: list['Receiver']) -> list[str]:
        return list(np.unique([receiver.antenna_name for receiver in receivers]))
