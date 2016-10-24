import numpy as np
import os
import random
import h5py
from numpy import genfromtxt
from fasttag_cross import fasttag_cross
from evaluatePR import resultPR, evaluatePR

if __name__ == "__main__":
    numval = 1000
    topK = 5

    type_file_input = 'mat'

    # when using mat files...
    if 'mat' == type_file_input:

        MatFile = h5py.File('../data/nuswide_mm2016.mat', 'r')

        trainIdx = MatFile['trainIdx'][:] - 1
        trainIdx = trainIdx.astype(int).flatten().tolist()

        xTr = MatFile['I_tr'][:][:, trainIdx]
        xTe = MatFile['I_te'][:]
        tTr = MatFile['T_tr'][:][:, trainIdx]
        tTe = MatFile['T_te'][:]
        yTr = MatFile['L_tr'][:][:, trainIdx]
        yTe = MatFile['L_te'][:]


    # when using txt files...
    if 'txt' == type_file_input:
        tmpDataset = '../NUS_WIDE'

        xTr = genfromtxt(os.path.join(tmpDataset, 'xTr.txt'), delimiter=' ')
        xTe = genfromtxt(os.path.join(tmpDataset, 'xTe.txt'), delimiter=' ')
        yTr = genfromtxt(os.path.join(tmpDataset, 'yTr.txt'), delimiter=' ')
        yTe = genfromtxt(os.path.join(tmpDataset, 'yTe.txt'), delimiter=' ')
        tTr = genfromtxt(os.path.join(tmpDataset, 'tTr.txt'), delimiter=' ')
        tTe = genfromtxt(os.path.join(tmpDataset, 'tTe.txt'), delimiter=' ')

    nTr = xTr.shape[1]
    nTe = xTe.shape[1]

    # TrList = range(nTr)
    # random.shuffle(TrList)
    # valIdx = TrList[:numval]

    valIdx = random.sample(range(nTr), numval)
    valIdx.sort()
    print valIdx

    xTr = np.concatenate((xTr, np.ones((1, xTr.shape[1]), int)), axis=0)
    xTe = np.concatenate((xTe, np.ones((1, xTe.shape[1]), int)), axis=0)
    tTr = np.concatenate((tTr, np.ones((1, tTr.shape[1]), int)), axis=0)
    tTe = np.concatenate((tTe, np.ones((1, tTe.shape[1]), int)), axis=0)

    W, U, B = fasttag_cross(xTr, tTr, yTr, xTe, tTe, yTe, topK, valIdx)

    queryW = W.dot(xTe)
    queryU = U.dot(tTe)

    print '\n'

    resultsAllTag = evaluatePR(yTe, (queryW + queryU), topK, 'tag')
    print ('... For tag measure from all modality: \n\t P %f, R %f, N+ %d \n'
           % (resultsAllTag.prec, resultsAllTag.rec, resultsAllTag.retrieved))

    resultsAllImage = evaluatePR(yTe, (queryW + queryU), topK, 'image')
    print ('... For image measure from all modality: \n\t P %f, R %f, N+ %d \n'
           % (resultsAllImage.prec, resultsAllImage.rec, resultsAllImage.retrieved))

    print 'Fin!'
