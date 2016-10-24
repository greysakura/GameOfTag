import numpy as np


class resultPR:
    def __init__(self):
        prec = 0
        rec = 0
        f1 = 0
        retrieved = 0
        f1Ind = None
        precInd = None
        recInd = None


def evaluatePR(GTs, PREDs, topK, typeIn):

    outresultPR = resultPR()

    eps = 2.2204e-16
    GTs = (GTs > 0)
    GTs = GTs.astype(float)

    hardPREDs = np.zeros(PREDs.shape, float)

    for n in range(GTs.shape[1]):
        confidence = PREDs[:, n]
        si = np.argsort(-confidence).flatten().tolist()[0:topK]
        hardPREDs[si, n] = 1

    if typeIn == 'tag':
        precInd = np.divide(np.sum(np.multiply(hardPREDs, GTs), axis=1), np.maximum(np.sum(hardPREDs, axis=1), eps))
        prec = np.mean(precInd)

        recInd = np.divide(np.sum(np.multiply(hardPREDs, GTs), axis=1), np.maximum(np.sum(GTs, axis=1), eps))
        rec = np.mean(recInd)

    elif typeIn == 'image':
        precInd = np.divide(np.sum(np.multiply(hardPREDs, GTs), axis=0), np.maximum(np.sum(hardPREDs, axis=0), eps))
        prec = np.mean(precInd)

        recInd = np.divide(np.sum(np.multiply(hardPREDs, GTs), axis=0), np.maximum(np.sum(GTs, axis=0), eps))
        rec = np.mean(recInd)

    else:
        # fprintf('error type input! please set type as tag or image! \n');
        print 'error type input! please set type as tag or image!'
        return

    f1Ind = 2 * np.divide(np.multiply(precInd, recInd), np.maximum(
        precInd + recInd, eps))

    f1 = 2 * prec * rec / (prec+rec)

    retrievedInd = np.where(np.sum(np.multiply(hardPREDs, GTs), axis=1) > 0)[0].tolist()
    retrieved = len(retrievedInd)

    outresultPR.prec = prec
    outresultPR.rec = rec
    outresultPR.f1 = f1
    outresultPR.retrieved = retrieved
    outresultPR.f1Ind = f1Ind
    outresultPR.precInd = precInd
    outresultPR.recInd = recInd

    # return prec, rec, f1, retrieved, f1Ind, precInd, recInd
    return outresultPR
