import numpy as np
import h5py
from scipy.io import loadmat

f = h5py.File('/Tokei_Mac/Work_for_work/project-mm2016/data/NUS-WIDE/nuswide_mm2016.mat')

print type(f)
print f

ff = loadmat('/Tokei_Mac/Work_for_work/project-mm2016/data/NUS-WIDE/nuswide_mm2016.mat')

print ff