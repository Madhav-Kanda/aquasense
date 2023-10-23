import numpy as np

# 1 is not converged, 0 is converged
def is_converged(splits, store, abs_thres = 1E-7, max_splits = 100000, rate = 0.05):
    if (len(splits) == 1):
        return False

    last_supp, last_sense = store[-2]
    this_supp, this_sense = store[-1]
    if (abs(np.sum(last_supp * last_sense) - np.sum(this_supp * this_sense)) / np.sum(last_supp * last_sense) < rate):
        return True
    # if abs(last_supp * last_supp - this_supp * this_sense) < abs_thres:
    #     return True
    if splits[-1] > max_splits:
        return False

    return False


            