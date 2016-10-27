import h5py
from numpy import loadtxt
from scipy.io import loadmat

numtrain = 1000

SX = h5py.File('../data/Flickr30k/flickr30k_features.mat')['SX'][:]
SY = h5py.File('../data/Flickr30k/flickr30k_features.mat')['SY'][:]

trainind = h5py.File('../data/Flickr30k/index_training_testing/ind_train.mat')['ind_train'][:]
trainind = trainind.astype(int).flatten().tolist()

tmpfile = loadmat('../data/Flickr30k/index_training_testing/3k_ind.mat')

valind = tmpfile['devind'][:]
valind = valind.astype(int).flatten().tolist()
testind = tmpfile['testind'][:]
testind = testind.astype(int).flatten().tolist()

tmpfile = loadmat('../data/Flickr30k/flickr30k_wordMatrix.mat')
wordMatrix = loadmat('../data/Flickr30k/flickr30k_wordMatrix.mat')['wordMatrix'][:]


xTr = SX[:, trainind[0:numtrain]]
xTe = SX[:, testind]
tTr = SY[:, trainind[0:numtrain]]
tTe = SY[:, testind]

yTr = wordMatrix[trainind[0:numtrain], :].transpose().astype(float)
yTe = wordMatrix[testind, :].transpose().astype(float)

nTr = xTr.shape[1]
nTe = xTe.shape[1]

print nTr, nTe
