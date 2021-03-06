import numpy as np
import random
import h5py
# from numpy import genfromtxt
from fasttag_cross import fasttag_cross
from evaluatePR import resultPR, evaluatePR
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity
from perf_metric4Label import perf_metric4Label


# def defect_on_ratio(MatIn):
#
#     ## MatIn:  m * n
#     ## m: number of instances
#     ## n: number of labels
#     ## for each instance, turn at least one tag of "1" into "0".
#
#     defect_ratio = 0.2
#
#     ## Find where label == 1
#     for i in range(MatIn.shape[0]):
#
#         Location_1s = np.asarray(np.where(MatIn[i, :] != 0))
#         NumOf1 = len(Location_1s[0])
#
#         # Locate those 0s
#         Location_0s = np.asarray(np.where(MatIn[i, :] == 0))
#
#         ## How many tags of 0 exist?
#         NumOf0 = len(Location_0s[0])
#
#         if NumOf0 > 0:
#             ## How many of 0s need to be changed into 1?
#             ToBe1 = max(1, int(defect_ratio * NumOf1))
#
#             random.shuffle(Location_0s.transpose())  # Need to transpose!
#             MatIn[i, :][Location_0s[:, 0:ToBe1]] = [1] * ToBe1
#     return MatIn

def incomplete_and_defect(MatIn):
    ## MatIn:  m * n
    ## m: number of instances
    ## n: number of labels
    ## 2 steps.

    incomplete_ratio = 0.2
    defect_ratio = 0.2

    for i in range(MatIn.shape[0]):

        ## Step 0: Divide 1s and 0s.

        Location_1s = np.asarray(np.where(MatIn[i, :] != 0))
        NumOf1 = len(Location_1s[0])

        Location_0s = np.asarray(np.where(MatIn[i, :] == 0))
        NumOf0 = len(Location_0s[0])

        ## Step 1: Incomplete the mat. On ratio, take at least one 1s to 0.
        if NumOf1 > 0:
            ToBe0 = max(1, int(incomplete_ratio * NumOf1))
            random.shuffle(Location_1s.transpose())  # Need to transpose!
            MatIn[i, :][Location_1s[:, 0:ToBe0]] = [0] * ToBe0

        ## Step 2: Defect the mat. On ratio,  Take at least one 0s to 1.
        if NumOf0 > 0:
            ToBe1 = max(1, int(defect_ratio * NumOf1))
            random.shuffle(Location_0s.transpose())  # Need to transpose!
            MatIn[i, :][Location_0s[:, 0:ToBe1]] = [1] * ToBe1

    return MatIn


if __name__ == "__main__":
    numval = 1000
    topK = 5

    # type_file_input = 'mat'
    dataset_to_test = 'nuswide'
    # dataset_to_test = 'flickr30k'

    MatFile = None

    # when using mat files...
    # if 'mat' == type_file_input:
    if 'nuswide' == dataset_to_test:
        MatFile = h5py.File('../data/nuswide_mm2016.mat', 'r')
        trainIdx = MatFile['trainIdx'][:] - 1
        trainIdx = trainIdx.astype(int).flatten().tolist()

        xTr = MatFile['I_tr'][:][:, trainIdx]
        xTe = MatFile['I_te'][:]
        tTr = MatFile['T_tr'][:][:, trainIdx]
        tTe = MatFile['T_te'][:]
        yTr = MatFile['L_tr'][:][:, trainIdx]
        yTe = MatFile['L_te'][:]
        nTr = xTr.shape[1]
        nTe = xTe.shape[1]

        ## Read full label set for train images.
        # MatFile_FullSet = h5py.File('../data/NUSWIDE_fasttag.mat', 'r')
        # yTr = MatFile_FullSet['Y_train'][:]
        # print yTr.shape
        # print xTr.shape
        # raw_input()

        # print xTr.shape
        # print xTe.shape
        # print tTr.shape
        # print tTe.shape
        # print yTr.shape
        # print yTe.shape

        valIdx = random.sample(range(nTr), numval)
        valIdx.sort()

    elif 'flickr30k' == dataset_to_test:
        numtrain = 5000
        # MatFile = h5py.File('../data/flickr30k_index_mm2016.mat', 'r')

        SX = h5py.File('../data/Flickr30k/flickr30k_features.mat')['SX'][:]
        SY = h5py.File('../data/Flickr30k/flickr30k_features.mat')['SY'][:]

        trainIdx = h5py.File('../data/Flickr30k/index_training_testing/ind_train.mat')['ind_train'][:] - 1
        trainIdx = trainIdx.astype(int).flatten().tolist()
        print min(trainIdx)

        tmpfile = loadmat('../data/Flickr30k/index_training_testing/3k_ind.mat')

        testind = tmpfile['testind'][:] - 1
        testind = testind.astype(int).flatten().tolist()

        tmpfile = loadmat('../data/Flickr30k/flickr30k_wordMatrix.mat')
        wordMatrix = loadmat('../data/Flickr30k/flickr30k_wordMatrix.mat')['wordMatrix'][:]

        xTr = SX[:, trainIdx[0:numtrain]]
        xTe = SX[:, testind]
        tTr = SY[:, trainIdx[0:numtrain]]
        tTe = SY[:, testind]

        yTr = wordMatrix[trainIdx[0:numtrain], :].transpose().astype(float)
        yTe = wordMatrix[testind, :].transpose().astype(float)

        nTr = xTr.shape[1]
        nTe = xTe.shape[1]

        print 'Calc: ', nTr, nTe

        valIdx = random.sample(range(nTr), numval)
        valIdx.sort()

        print xTr.shape
        print xTe.shape
        print tTr.shape
        print tTe.shape
        print yTr.shape
        print yTe.shape

    else:
        print 'Invalid dataset!'
        exit()

    # # when using txt files...
    # if 'txt' == type_file_input:
    #     tmpDataset = '../NUS_WIDE'
    #
    #     xTr = genfromtxt(os.path.join(tmpDataset, 'xTr.txt'), delimiter=' ')
    #     xTe = genfromtxt(os.path.join(tmpDataset, 'xTe.txt'), delimiter=' ')
    #     yTr = genfromtxt(os.path.join(tmpDataset, 'yTr.txt'), delimiter=' ')
    #     yTe = genfromtxt(os.path.join(tmpDataset, 'yTe.txt'), delimiter=' ')
    #     tTr = genfromtxt(os.path.join(tmpDataset, 'tTr.txt'), delimiter=' ')
    #     tTe = genfromtxt(os.path.join(tmpDataset, 'tTe.txt'), delimiter=' ')

    ## If we want to use defected labels...

    # yTr = incomplete_and_defect(yTr.transpose())
    # yTr = yTr.transpose()

    ## Go on

    xTr = np.concatenate((xTr, np.ones((1, xTr.shape[1]), int)), axis=0)
    xTe = np.concatenate((xTe, np.ones((1, xTe.shape[1]), int)), axis=0)
    tTr = np.concatenate((tTr, np.ones((1, tTr.shape[1]), int)), axis=0)
    tTe = np.concatenate((tTe, np.ones((1, tTe.shape[1]), int)), axis=0)

    W, U, B = fasttag_cross(xTr, tTr, yTr, xTe, tTe, yTe, topK, valIdx)

    queryW = W.dot(xTe)
    queryU = U.dot(tTe)

    resultsAllTag = evaluatePR(yTe, (queryW + queryU), topK, 'tag')
    print ('... For tag measure from all modality: \n\t P %f, R %f, N+ %d \n'
           % (resultsAllTag.prec, resultsAllTag.rec, resultsAllTag.retrieved))

    resultsAllImage = evaluatePR(yTe, (queryW + queryU), topK, 'image')
    print ('... For image measure from all modality: \n\t P %f, R %f, N+ %d \n'
           % (resultsAllImage.prec, resultsAllImage.rec, resultsAllImage.retrieved))

    # # evaluate with cross-modal measures
    # dbW = W.dot(xTr)
    # dbU = U.dot(tTr)
    #
    # # I to T
    # dist_WtoU = 1.0 - cosine_similarity(queryW.transpose(), dbU.transpose())
    # # T to I
    # dist_UtoW = 1.0 - cosine_similarity(queryU.transpose(), dbW.transpose())
    #
    # MAP_I2T = perf_metric4Label(yTr.transpose(), yTe.transpose(), dist_WtoU.transpose())
    # MAP_T2I = perf_metric4Label(yTr.transpose(), yTe.transpose(), dist_UtoW.transpose())
    #
    # print ('I2T: %f, T2I: %f \n' %(MAP_I2T, MAP_T2I))
