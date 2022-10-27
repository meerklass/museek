import h5py
import healpy as hp
from matplotlib import pyplot

with h5py.File("./BGS_maps.hdf", 'r') as fp:
    bgs_map = fp['MAPS'][20, 0]
    counts = fp['COUNTS'][20, 0]

bgs_map[counts == 0] = hp.UNSEEN
hp.mollview(bgs_map, cmap="gist_earth")
pyplot.show()
