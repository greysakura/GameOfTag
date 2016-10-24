import numpy as np
import h5py
from scipy.io import loadmat

f = h5py.File('../data/nuswide_mm2016.mat', 'r')

print type(f)
print f

# ff = loadmat('../data/nuswide_mm2016.mat')
#
# print ff

print f.keys()
print f.name

trainIdx = None
I_tr = None


print f.keys()
for name in f.keys():
    exec name + '=f[\'' + name + '\'][:]'

print I_tr.shape
print type(I_tr)

print trainIdx.shape
