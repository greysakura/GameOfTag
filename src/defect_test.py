import numpy as np
import numpy.ma as ma
from numpy.random import random_integers
from numpy import random

def incomplete_and_defect(MatIn):
    ## MatIn:  m * n
    ## m: number of instances
    ## n: number of labels
    ## 2 steps.

    incomplete_ratio = 0.2
    defect_ratio = 0.03

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
            ToBe1 = max(1, int(defect_ratio * NumOf0))
            random.shuffle(Location_0s.transpose())  # Need to transpose!
            MatIn[i, :][Location_0s[:, 0:ToBe1]] = [1] * ToBe1


    return MatIn

def defect_on_ratio(MatIn):

    ## MatIn:  m * n
    ## m: number of instances
    ## n: number of labels
    ## for each instance, turn at least one tag of "1" into "0".

    defect_ratio = 0.03

    # print 'MatIn: ', MatIn, '\n'

    ## Find where label == 1
    for i in range(MatIn.shape[0]):
        # Locate those 0s
        Location = np.asarray(np.where(MatIn[i, :] == 0))

        ## How many tags of 0 exist?
        NumOf0 = len(Location[0])

        if NumOf0 > 0:
            ## How many of 0s need to be changed into 1?
            ToBe1 = max(1, int(defect_ratio * NumOf0))

            random.shuffle(Location.transpose())  # Need to transpose!
            MatIn[i, :][Location[:, 0:ToBe1]] = [1] * ToBe1
    return MatIn

if __name__ == '__main__':
    if __name__ == "__main__":
        # AAA = np.asarray(range(12)).reshape((3, 4))
        AAA = np.random.random_integers(0, 1, (4, 5))
        print 'Origin: \n', AAA, '\n'
        # ## Try use mask. Numpy
        # mask = ((AAA > 2) + (AAA > 7)) > 0
        # mx = ma.array(AAA, mask=mask)
        #
        # CCC = mx.compressed()
        # DDD = random_integers(0, 1, CCC.shape)
        #
        # print AAA
        # print mask
        # # print AAA[np.where((AAA > 4) & (AAA < 7))]
        # GCC = AAA[np.where((AAA > 2) & (AAA < 7))]
        # # print random.shuffle(np.where((AAA > 4) & (AAA < 7)))
        # TTT = np.asarray(np.where((AAA > 2) & (AAA < 7)))
        # print np.where((AAA > 2) & (AAA < 7))
        # random.shuffle(TTT.transpose())
        # print TTT[:, 0:2]
        #
        # print 2.0 / 75
        # print [1] * 10
        # AAA = defect_on_ratio(AAA)
        AAA = incomplete_and_defect(AAA)
        print 'Now: \n', AAA, '\n'
