import numpy as np


def evaluate(GTs, PREDs, topK):
    # Input:
    # GTs: K x n matrix containing the groundtruth
    # PREDs: K x n matrix containing the prediction confidence
    # topK: number of tags given to each image
    # Output:
    eps = 2.2204e-16
    GTs = (GTs > 0)
    GTs = GTs.astype(float)

    hardPREDs = np.zeros(PREDs.shape, float)

    for n in range(GTs.shape[1]):
        confidence = PREDs[:, n]
        si = np.argsort(-confidence).flatten().tolist()[0:topK]
        hardPREDs[si, n] = 1.0

    precInd = np.divide(np.sum(np.multiply(hardPREDs, GTs), axis=1), np.maximum(np.sum(hardPREDs, axis=1), eps))
    prec = np.mean(precInd)

    recInd = np.divide(np.sum(np.multiply(hardPREDs, GTs), axis=1), np.maximum(np.sum(GTs, axis=1), eps))
    rec = np.mean(recInd)

    f1Ind = np.divide(2 * np.multiply(precInd, recInd), np.maximum(precInd + recInd, eps))
    f1 = 2 * prec * rec / (prec + rec)

    retrievedInd = (np.sum(hardPREDs * GTs, axis=1) > 0).astype(int)
    retrieved = np.sum(retrievedInd > 0)

    return prec, rec, f1, retrieved, f1Ind, precInd, recInd
