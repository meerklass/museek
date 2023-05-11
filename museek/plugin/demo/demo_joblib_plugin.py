from math import sqrt
from typing import Generator

from ivory.plugin.abstract_parallel_joblib_plugin import AbstractParallelJoblibPlugin
from ivory.utils.result import Result
from museek.enum.demo_enum import DemoEnum


class DemoJoblibPlugin(AbstractParallelJoblibPlugin):
    """ Demo plugin utilising the `joblib` parallelisation. """

    def __init__(self, n_iter: int, **kwargs):
        """ Initialise the super class and set the number of iterations `n_iter`. """
        super().__init__(**kwargs)
        self.n_iter = n_iter

    def gather_and_set_result(self, result_list: list[float]):
        """ Sum all results together and save that. """
        result = sum(result_list)
        self.set_result(result=Result(location=DemoEnum.PARALLEL_RESULT, result=result))

    def map(self) -> Generator[int, None, None]:
        """ Yield squared integers. """
        for i in range(self.n_iter):
            yield i ** 2

    def run_job(self, anything: int) -> float:
        """ Return the square root of `anything`. """
        return sqrt(anything)

    def set_requirements(self):
        """ No requirements. """
        pass
