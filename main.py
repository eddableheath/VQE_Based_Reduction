# Testing for QOAO approach to basis reduction
# Author: Edmund Dable-HEath
"""
    Adapting Rocky's code.
"""

import numpy as np
import QAOA_4D as qa
from qutip import Qobj
import time


def main():
    B1 = qa.QAOA(Qobj(B), 2).iteration(1, 1, np.arange(0, 1, .1), step=1)
    print(B1)


if __name__ == "__main__":
    B = np.array([[1, 2, -4],
                  [-1, -4, -3],
                  [-1, -8, 6]])
    print(Qobj(B))
    start_time = time.time()
    main()
    print(time.time()-start_time)