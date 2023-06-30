from abc import ABC

from museek.data_element import DataElement
from museek.factory.data_element_factory import DataElementFactory


class AbstractModel(ABC):
    """ Abstract base class for models, e.g. ground spill, atmosphere or galaxy temperature. """
    def __init__(self, data_element_factory: DataElementFactory):
        """ Initialise with a data element factory. """
        self.data_element_factory = data_element_factory

    def temperature(self, **kwargs) -> DataElement:
        """ Abstract method. Return the model of the temperature. """
