import numpy as np

def calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val):
    """
    コミュニティ C の重み w_C を計算する

    Parameters:
    - C: コミュニティ
    - A_plus: 正の隣接行列
    - A_minus: 負の隣接行列
    - D_plus: 正の次数
    - D_minus: 負の次数
    - lambda_val: パラメータ

    Returns:
    - w_C: コミュニティ C の重み
    """
    # コミュニティのサイズ
    size_C = len(C)

    # CとCの直積
    CC = np.ix_(C, C)

    # 正のエッジ
    sum_a_plus = np.sum(A_plus[CC])
    sum_d_plus = np.sum(D_plus[C])
    plus = (2 * sum_a_plus - 2 * (1 - lambda_val) * sum_d_plus)

    # 負のエッジ
    sum_a_minus = np.sum(A_minus[CC])
    sum_d_minus = np.sum(D_minus[C])
    minus = (2 * sum_a_minus - 2 * lambda_val * sum_d_minus)

    # w_C の計算
    w_C = (plus - minus) / size_C

    return w_C
