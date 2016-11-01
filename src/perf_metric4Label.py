import numpy as np
from numpy.matlib import repmat
import datetime

# def ismember(a, b):
#     return [1.0 if (i in b) else 0.0 for i in a]

def ismember(a, b):
    mytmp = []
    for i in a:
        index = np.where(np.asarray(b) == i)[0]
        if index.size == 0:
            mytmp.append(0)
        else:
            mytmp.append(1)
    return mytmp

def perf_metric4Label(RetrLabels, QueryLabels, HammingDist):
    ##############################################
    # Calculating mAP for retrieval
    # RetrLabels: m*l binary matrix ({0, 1}), m: retrieval set size, l: vocabulary size
    # QueryLabels: n*l binary matrix ({0, 1}), n: query set size, l: vocabulary size
    # HammingDist: m*n distance matrix between retrieval and query sets
    ##############################################
    starttime = datetime.datetime.now()
    tsN, tagNum = QueryLabels.shape

    multiLabel = tagNum > 1
    mAP = 0.0
    goodQueryNum = 0

    tmp_m = HammingDist.shape[0]

    if multiLabel:
        rM = RetrLabels.dot(QueryLabels.transpose())

    for ti in range(tsN):

        gnd = np.where(rM[:, ti] > 0)[0]

        gndNum = len(gnd)

        if gndNum == 0:
            continue

        goodQueryNum += 1

        tmpI = np.argsort(HammingDist[:, ti], axis=0)

        ## Too slow!!!!!
        rightInd = ismember(tmpI, gnd)
        indecies = np.where(np.asarray(rightInd) == 1.0)[0]
        P = np.divide(np.asarray(range(gndNum)) + 1.0, np.asarray(indecies, float) + 1.0)
        AP = np.mean(P)
        mAP = mAP + AP
    mAP = 1.0 * mAP / goodQueryNum
    endtime = datetime.datetime.now()
    print 'evaluation time: ', (endtime - starttime).seconds, ' seconds'
    return mAP