from utils.wc import calc_w_C
from utils.lps_with_partition import LPSWithPartition
from utils.ap_milp import AP_MILP
from utils.partition import generate_partition

def column_generation_with_partition(
        vertices, A_plus, A_minus, D_plus, D_minus, lambda_val):
    '''
    分割数制約付きの列生成法

    Parameters:
    - vertices: 頂点のリスト
    - A_plus: 正の隣接行列
    - A_minus: 負の隣接行列
    - D_plus: 正の次数
    - D_minus: 負の次数
    - lambda_val: パラメータ

    Returns:
    '''
    # k=2 についてtest
    print("k=2")
    k1 = 2
    
    init_partitions = [generate_partition(vertices, k1)]
    w_C_dict = {}
    for partition in init_partitions:
        for C in partition:
            frozen_C = frozenset(C)
            if frozen_C not in w_C_dict:
                w_C_dict[frozen_C] = calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val)
    S = list(w_C_dict.keys())

    lps = LPSWithPartition(S, w_C_dict, vertices)
    lps.add_partition_constr(k1)
    lps.solve_model()
    lps.debag_print_model()

    # k=3 についてtest
    print("k=3")
    k2 = 3

    init_partitions = [generate_partition(vertices, k2)]
    w_C_dict = {}
    for partition in init_partitions:
        for C in partition:
            frozen_C = frozenset(C)
            if frozen_C not in w_C_dict:
                w_C_dict[frozen_C] = calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val)
    S = list(w_C_dict.keys())

    lps.add_partition_constr(k2)
    lps.solve_model()
    lps.debag_print_model()