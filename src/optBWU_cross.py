import numpy as np
from scipy.sparse import dia_matrix
from numpy.linalg import inv, norm
from optB import optB


def optBWU_cross(Lfreq, L, alpha, noise, maxLayer, weights, invTr, Xb, intTr, Tb, sigmav, sigmat):
    Ws = []
    Us = []
    Ms = []
    Bs = []

    mylambda = 0.00001
    maxIter = 10
    tol = 0.01

    d = Xb.shape[0]
    m = Tb.shape[0]
    [r, n] = Lfreq.shape
    M = L

    B0s = optB(Lfreq, L, weights, noise, maxLayer, mylambda)

    weights = dia_matrix((weights, [0]), shape=(weights.shape[0], weights.shape[0]))

    for ii in range(maxLayer):
        k = M.shape[0]
        iB = np.eye(k + 1)
        iB[-1, -1] = 0

        Mb = np.concatenate((M, np.ones((1, n), float)), axis=0)

        weightedMb = weights.dot(np.transpose(Mb))

        Sl = Mb.dot(weightedMb)

        q = np.ones((k + 1, 1), float) * (1 - noise)

        q[-1, 0] = 1

        Q = Sl * (q.dot(np.transpose(q)))

        Q = Q - np.diag(np.diag(Q)) + np.diag(np.multiply(q.flatten(), np.diag(Sl)))

        if ii == 0:
            P = Lfreq.dot(weightedMb) * np.repeat(np.transpose(q), r, axis=0)
        else:
            P = np.multiply(Sl[0:-1, :], np.repeat(np.transpose(q), k, axis=0))

        B = B0s[ii]

        prevB = B
        prevW = np.random.rand(r, d)
        prevU = np.random.rand(r, m)

        Wd = Mb.dot(invTr)
        Ud = Mb.dot(intTr)

        for myiter in range(maxIter):
            W = sigmav * B.dot(Wd)
            U = sigmat * B.dot(Ud)

            predW = W.dot(Xb)
            predU = U.dot(Tb)

            # here we compute for the new B with both visual and text features
            B = alpha * P + sigmav * predW.dot(weightedMb) + sigmat * predU.dot(weightedMb)
            B = B.dot(inv(alpha * Q + alpha * mylambda * iB + (sigmav + sigmat) * Sl))

            # optcondW = norm(W-prevW, 'fro')/norm(prevW, 'fro')
            optcondW = norm(W - prevW, 'fro') / norm(prevW, 'fro')
            # optcondU = norm(U-prevU, 'fro')/norm(prevU, 'fro');
            optcondU = norm(U - prevU, 'fro') / norm(prevU, 'fro')
            # optcondB = norm(B-prevB, 'fro')/norm(prevB, 'fro');
            optcondB = norm(B - prevB, 'fro') / norm(prevB, 'fro')

            if (optcondW < tol) & (optcondU < tol) & (optcondB < tol):
                break

            prevW = W
            prevU = U
            prevB = B

        M = np.tanh(B.dot(Mb))

        Ms.append(M)
        Ws.append(W)
        Us.append(U)
        Bs.append(B)

    return Ms, Ws, Us, Bs
