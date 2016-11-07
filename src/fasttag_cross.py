import numpy as np
from scipy.sparse import dia_matrix
from numpy.linalg import inv
from multiHyperTunning_cross import multiHyperTunning_cross
from optBWU_cross import optBWU_cross
from evaluate import evaluate


def fasttag_cross(xTr, tTr, yTr, xTe, tTe, yTe, topK, valIdx):
    ##############################################
    # fasttag_cross:
    # input: xTr, tTr, yTr, xTe, tTe, yTe, topK, valIdx
    # output: bestW, bestU, bestB

    # d: dimension of training feature
    d = xTr.shape[0]
    # m: dimension of training labeling
    m = tTr.shape[0]
    # nTr: number of training samples
    nTr = xTr.shape[1]
    # K = size(yTr, 1);
    K = yTr.shape[0]
    # trainIdx: index of training samples
    trainIdx = list(set(range(nTr)) - set(valIdx))

    outHyperparams = multiHyperTunning_cross(xTr[:, trainIdx], tTr[:, trainIdx], yTr[:, trainIdx], xTr[:, valIdx],
                                             tTr[:, valIdx], yTr[:, valIdx], topK)

    print 'multiHyperTunning Finished!\n'

    iW = dia_matrix((np.ones(d - 1), np.array([0])), shape=(d, d))
    iU = dia_matrix((np.ones(m - 1), np.array([0])), shape=(m, m))

    yTr_sum = np.maximum(np.sum((yTr > 0).astype(float), axis=1), 1.0)
    Wpos = np.divide(np.ones(yTr_sum.shape, float), yTr_sum)

    W = np.zeros((K, d), float)
    U = np.zeros((K, m), float)
    B = np.zeros((K, K), float)

    bestF1 = 0
    bestW = W
    bestU = U
    bestB = B
    bestOptIter = 0

    print 'After tuning, now testing on test samples:'
    print 'len of outHyperparams', len(outHyperparams)
    for optIter in range(len(outHyperparams)):
        tagIdx = outHyperparams[optIter].tagIdx
        beta = outHyperparams[optIter].beta
        mu = outHyperparams[optIter].mu
        noise = outHyperparams[optIter].noise
        alpha = outHyperparams[optIter].alpha
        maxLayer = outHyperparams[optIter].layers
        sigmav = outHyperparams[optIter].sigmav
        sigmat = outHyperparams[optIter].sigmat

        instanceIdx = np.where(np.sum((yTr[tagIdx, :] > 0).astype(float), axis=0) > 0)[0].tolist()

        yTr_consider = yTr[tagIdx, :][:, instanceIdx] * np.repeat(Wpos[tagIdx].reshape((-1, 1)), len(instanceIdx),
                                                                  axis=1)
        weights = np.max(yTr_consider, axis=0)

        print ('optIter = %d, len(tagIdx) = %d, len(instanceIdx) = %d' % (optIter, len(tagIdx), len(instanceIdx)))

        Sx = (
        xTr[:, instanceIdx].dot(dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).toarray())).dot(
            xTr[:, instanceIdx].transpose())

        invSx = dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).dot(
            np.transpose(xTr[:, instanceIdx])).dot(inv(sigmav * Sx + beta * iW))

        St = (
        tTr[:, instanceIdx].dot(dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).toarray())).dot(
            tTr[:, instanceIdx].transpose())

        intTr = dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).dot(
            np.transpose(tTr[:, instanceIdx])).dot(inv(sigmat * St + mu * iU))

        Ms, Ws, Us, Bs = optBWU_cross(yTr[tagIdx, :][:, instanceIdx], yTr[:, instanceIdx], alpha, noise, maxLayer,
                                      weights, invSx, xTr[:, instanceIdx], intTr, tTr[:, instanceIdx], sigmav, sigmat)

        W[tagIdx, :] = Ws[maxLayer - 1]
        U[tagIdx, :] = Us[maxLayer - 1]

        predTe = W.dot(xTe) + U.dot(tTe)
        prec, rec, f1, retrieved, f1Ind, precInd, recInd = evaluate(yTe, predTe, topK)

        print ('FastTag_cross: Beta = %f, Noise = %f, Layer = %d, Alpha = %f, Prec = %f, Rec = %f, F1 = %f, N+ = %d' % (
        beta, noise, maxLayer, alpha, prec, rec, f1, retrieved))
        if 0 == bestF1:
            bestF1 = f1
            bestW = W
            bestU = U
            bestB = B
            bestOptIter = optIter
        elif f1 > bestF1:
            bestW = W
            bestU = U
            bestB = B
            bestF1 = f1
            bestOptIter = optIter
        else:
            pass

    # return W, U, B
    print 'The best iter is:', bestOptIter
    return bestW, bestU, bestB
