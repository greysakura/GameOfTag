import numpy as np
import h5py
from scipy.io import loadmat

f = h5py.File('../data/nuswide_mm2016.mat')

print type(f)
print f

# ff = loadmat('../data/nuswide_mm2016.mat')
#
# print ff

f.keys()