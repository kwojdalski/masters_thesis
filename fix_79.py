# Fix #79
import numpy as np
def reward(c, p):
    if c<=0 or p<=0 or not np.isfinite(c) or not np.isfinite(p):
        raise ValueError('Invalid portfolio')
    return float(np.log(c/p))
