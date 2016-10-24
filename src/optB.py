import numpy as np
from scipy.sparse import dia_matrix
from numpy.linalg import inv


def optB(Lfreq, L, weights, noise, maxLayers, mylambda):
    [r, n] = Lfreq.shape
    M = L
    weights = dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0]))

    B0s = []

    for i in range(maxLayers):
        k = M.shape[0]
        iB = np.eye(k + 1)
        iB[-1, -1] = 0
        Mb = np.concatenate((M, np.ones((1, M.shape[1]))), axis=0)

        weightedMb = weights.dot(np.transpose(Mb))

        Sl = Mb.dot(weightedMb)
        q = np.multiply(np.ones((k + 1, 1), float), (1 - noise))
        q[-1] = 1

        Q = Sl * (q.dot(q.transpose()))

        Q = Q - np.diag(np.diag(Q)) + np.diag(np.multiply(q.flatten(), np.diag(Sl)))

        if i == 0:
            P = (Lfreq.dot(weightedMb)) * (np.repeat(np.transpose(q), r, axis=0))
        else:
            P = np.multiply(Sl[0:-1, :], np.repeat(np.transpose(q), k, axis=0))

        # print 'This is Marker 3'
        B = P.dot(inv(Q + mylambda * iB))

        B0s.append(B)
        M = np.tanh(B.dot(Mb))

    return B0s
