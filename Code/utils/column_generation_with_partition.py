from utils.wc import calc_w_C
from utils.lps import LPS
from utils.ap_milp_with_partition import AP_MILPWithPartition

def column_generation_with_partition(
        vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, init_partitions):
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

    lps = LPS(
        vertices=vertices, 
        A_plus=A_plus, 
        A_minus=A_minus, 
        D_plus=D_plus, 
        D_minus=D_minus, 
        lambda_val=lambda_val, 
        init_partitions=init_partitions
        )

    ap_milp = AP_MILPWithPartition(vertices, A_plus, A_minus, D_plus, D_minus, lambda_val)

    cnt = 0
    lps_opt_list = []
    cg_opt = 0
    cg_sol = {}
    S = []

    while(True):
        # LP(S)を解く, 最適値, 主問題の解, 双対問題の解を得る. 
        lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_model()
        lps_opt_list.append(lps_opt)

        # AP-MILPを解く ap_milp_opt, ap_milp_sol
        ap_milp.add_lps_dual_sol(lps_dual_sol)
        ap_milp_opt, ap_milp_sol = ap_milp.solve_model()

        # 終了条件 ap_milp_opt <= 0 なら stop
        if (ap_milp_opt <= 0):
            # 最終結果の保存
            cg_opt = lps_opt
            cg_sol = lps_primal_sol

            break

        # ap_milp_solよりS, w_C_dictの更新
        frozen_C = frozenset(u for u, x_val in ap_milp_sol["x_u"].items() if x_val == 1.0)

        # LPSを更新
        S = lps.update_model(frozen_C)

        # カウントの更新
        cnt+=1

        # DEBUG
        print(f"反復回数: {cnt}")
        print(f"最適値: {lps_opt}")

    return cg_opt, cg_sol, lps_opt_list, S, cnt

    # # TEST 列生成の生成する集合の要素数を固定する
    # # LP(S)を解く, 最適値, 主問題の解, 双対問題の解を得る. 
    # lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_model()
    # lps_opt_list.append(lps_opt)

    # # AP-MILPを解く ap_milp_opt, ap_milp_sol
    # ap_milp.add_lps_dual_sol(lps_dual_sol)

    # # DEBUG AP-MILP の制約の更新の確認
    # ap_milp.update_model_for_k(len(vertices) - 1)
    # ap_milp.debag_print_model()
    
    # ap_milp.update_model_for_k(len(vertices) - 2)
    # ap_milp.debag_print_model()
