import numpy as np
from scipy.sparse import dia_matrix
from numpy.linalg import inv
from myHyperparams import myHyperparams
from optBWU_cross import optBWU_cross
from evaluate import evaluate


def multiHyperTunning_cross(xTr, tTr, yTr, xVal, tVal, yVal, topK):
    print 'hyper-parameter tuning with cross modal data... '
    outHyperparams = []

    threshold = 0
    thresholdPos = 2.2204e-16
    maxLayer = 3

    beta = 0.1
    mu = 0.1

    sigmav = 1.0
    sigmat = 1.0

    d = xTr.shape[0]
    m = tTr.shape[0]
    K = yTr.shape[0]

    iW = dia_matrix((np.ones(d - 1), np.array([0])), shape=(d, d))
    iU = dia_matrix((np.ones(m - 1), np.array([0])), shape=(m, m))

    bestF1 = 0
    bestL = np.zeros(yTr.shape, np.float32)
    bestRetrievedIdxF1 = np.zeros((K, 1), np.float32)
    bestWF1 = np.zeros((K, d), np.float32)
    bestUF1 = np.zeros((K, m), np.float32)

    yTr_sum = np.maximum(np.sum((yTr > 0).astype(float), axis=1), 1.0)
    Wpos = np.divide(np.ones(yTr_sum.shape, float), yTr_sum)

    myIter = 0

    while myIter < 5:

        tmpHyperparams = myHyperparams()

        improved = False

        predVal = bestWF1.dot(xVal) + bestUF1.dot(tVal)

        tagIdx = np.where(bestRetrievedIdxF1 == 0)[0].tolist()

        instanceIdx = np.where(np.sum((yTr[tagIdx, :] > 0).astype(float), axis=0) > 0)[0].tolist()

        yTr_consider = yTr[tagIdx, :][:, instanceIdx] * np.repeat(Wpos[tagIdx].reshape((-1, 1)), len(instanceIdx),
                                                                  axis=1)
        weights = np.max(yTr_consider, axis=0)

        print ('tagIdx = %d, instanceIdx = %d, thresholdPos = %f' % (
            len(tagIdx), len(instanceIdx), thresholdPos))

        SxTr = (
            xTr[:, instanceIdx].dot(
                dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).toarray())).dot(
            xTr[:, instanceIdx].transpose())

        invTr = dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).dot(
            np.transpose(xTr[:, instanceIdx])).dot(inv(np.multiply(SxTr, sigmav) + np.multiply(iW, beta)))

        StTr = (
            tTr[:, instanceIdx].dot(
                dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).toarray())).dot(
            tTr[:, instanceIdx].transpose())

        intTr = dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0])).dot(
            np.transpose(tTr[:, instanceIdx])).dot(inv(np.multiply(StTr, sigmat) + np.multiply(iU, mu)))

        for alpha in [10]:
            for noise in map(lambda x: x / 10.0, range(10)):

                Ms, Ws, Us, Bs = optBWU_cross(yTr[tagIdx, :][:, instanceIdx], yTr[:, instanceIdx], alpha, noise,
                                              maxLayer, weights, invTr, xTr[:, instanceIdx], intTr, tTr[:, instanceIdx],
                                              sigmav, sigmat)

                for layer in range(len(Ws)):
                    L = Ms[layer]

                    if np.mean(L[np.where(yTr[tagIdx, :][:, instanceIdx] > 0)]) < thresholdPos:
                        continue

                    W = Ws[layer]
                    U = Us[layer]

                    gamma = 0.5
                    predVal[tagIdx, :] = gamma * W.dot(xVal) + (1.0 - gamma) * U.dot(tVal)
                    precVal, recVal, f1Val, retrievedVal, f1Ind, precInd, recInd = evaluate(yVal, predVal, topK)

                    if f1Val > bestF1:
                        print (
                            'beta = %f, alpha = %f, noise = %f, layer = %d, precVal = %f, recVal = %f, f1Val = %f, retrievedVal = %d' % (
                                beta, alpha, noise, layer + 1, precVal, recVal, f1Val, retrievedVal))

                        bestL = L
                        bestF1 = f1Val
                        bestWF1[tagIdx, :] = W
                        bestUF1[tagIdx, :] = U
                        bestRetrievedIdxF1 = (f1Ind > threshold).astype(int)

                        tmpHyperparams.tagIdx = tagIdx
                        tmpHyperparams.beta = beta
                        tmpHyperparams.mu = mu
                        tmpHyperparams.noise = noise
                        tmpHyperparams.alpha = alpha
                        tmpHyperparams.layers = layer + 1
                        tmpHyperparams.sigmav = sigmav
                        tmpHyperparams.sigmat = sigmat

                        improved = True

        if not improved:
            break

        outHyperparams.append(tmpHyperparams)
        thresholdPos = np.mean(bestL[np.where(yTr[tagIdx, :][:, instanceIdx] > 0)])
        myIter += 1

    return outHyperparams
