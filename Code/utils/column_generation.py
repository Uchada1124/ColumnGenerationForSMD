from utils.wc import calc_w_C
from utils.lps import LPS
# from utils.ap_milp import AP_MILP
from utils.ap_qp import AP_MILP

def column_generation(vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, init_partitions): 
    '''
    列生成法

    Parameters:
    - vertices: 頂点のリスト
    - A_plus: 正の隣接行列
    - A_minus: 負の隣接行列
    - D_plus: 正の次数
    - D_minus: 負の次数
    - lambda_val: パラメータ
    - init_partitions: 初期の分割集合

    Returns:
    - 
    '''
    # 初期化
    w_C_dict = {}
    for partition in init_partitions:
        for C in partition:
            frozen_C = frozenset(C)
            if frozen_C not in w_C_dict:
                w_C_dict[frozen_C] = calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val)
    S = list(w_C_dict.keys())

    lps = LPS(S, w_C_dict, vertices)

    # print(w_C_dict)
    # print(S)
    
    # # LPSの解説
    # lps = LPS(S, w_C_dict, vertices)
    # # lps.print_lps()

    # lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_lps()
    # # lps.print_lps()

    # new_w_C_dict = {}
    # frozen_C = frozenset([0, 1, 2])
    # new_w_C_dict[frozen_C] = calc_w_C(
    #         list(frozen_C), A_plus, A_minus, D_plus, D_minus, lambda_val
    #     )
    # new_S = list(new_w_C_dict.keys())

    # print(new_w_C_dict)
    # print(new_S)

    # lps.update_lps(new_S, new_w_C_dict)
    # lps.print_lps()

    # lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_lps()
    # lps.print_lps()

    cnt = 0
    while(cnt <= 5):
        # LP(S)を解く, 最適値, 主問題の解, 双対問題の解を得る. 
        lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_lps()

        # AP-MILPを解く ap_milp_opt, ap_milp_sol
        ap_milp = AP_MILP(
            vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, lps_dual_sol
        )
        ap_milp_opt, ap_milp_sol = ap_milp.solve_ap_milp()

        # 終了条件 ap_milp_opt <= 0 なら stop
        if (ap_milp_opt <= 0): break

        # ap_milp_solよりS, w_C_dictの更新
        new_w_C_dict = {}
        frozen_C = frozenset(u for u, x_val in ap_milp_sol["x_u"].items() if x_val > 0.9)
        if frozen_C not in w_C_dict:
            new_w_C_dict[frozen_C] = calc_w_C(
                list(frozen_C), A_plus, A_minus, D_plus, D_minus, lambda_val
            )
        new_S = list(new_w_C_dict.keys())

        # LPSを更新
        lps.update_lps(new_S, new_w_C_dict)

        # 結果の出力
        print(f"cnt : {cnt}")
        print(f"LPS Objective Value : {lps_opt}")
        print(f"LPS Primal Solution : {lps_primal_sol}")
        print(f"AP-MILP Objective Value : {ap_milp_opt}")
        print(f"AP-MILP Primal Solution : {ap_milp_sol['x_u']}")

        # if (cnt % 10 == 0): lps.print_lps()
        lps.print_lps()
        ap_milp.print_ap_milp()

        cnt+=1

    lps.print_lps()