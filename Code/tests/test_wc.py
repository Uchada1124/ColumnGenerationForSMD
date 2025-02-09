import numpy as np

from utils.wc import calc_w_C

def test_calc_w_C():
    A_plus = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ])
    A_minus = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    D_plus = np.sum(A_plus, axis=1)
    D_minus = np.abs(np.sum(A_minus, axis=1))
    C = [0, 1, 2]
    lambda_val = 0.5

    result = calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val)
    assert result == 0
