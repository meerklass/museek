# SEEK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# SEEK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with SEEK.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jan 15, 2016

author: jakeret
'''

import os

import h5py
import numpy as np

from ivy.plugin.abstract_plugin import AbstractPlugin
from seek import Coords
from seek.plugins import load_data


class Plugin(AbstractPlugin):
    """
    Loads the data, mask and frequencies of the current iteration from disk. Can
    be used for closer analysis of the masking (sum threshold). The data is
    read from the current folder using the same filename as the first input
    filename.
    """

    def run(self):
        filename = os.path.basename(self.ctx.file_paths[0])
        filepath = os.path.join(self.ctx.params.post_processing_prefix,
                                filename)

        if self.ctx.params.verbose:
            print(filepath)

        self.ctx.strategy_start = load_data.get_observation_start_from_hdf5(filepath)
        with h5py.File(filepath, "r") as fp:
            tod = np.ma.array(fp["data"][()], mask=fp["mask"][()])
            self.ctx.tod_vx = tod
            self.ctx.tod_vy = tod.copy()
            self.ctx.frequencies = fp["frequencies"][()]
            self.ctx.time_axis = fp["time"][()]
            self.ctx.coords = Coords(fp["ra"][()], fp["dec"][()], None, None, self.ctx.time_axis)
            self.ctx.ref_channel = fp["ref_channel"][()]

    def __str__(self):
        return "loading processed data"
