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
    cloumns = {} # 追加する予定の列を保存する辞書   

    while(cnt < 1):
        # LP(S)を解く, 最適値, 主問題の解, 双対問題の解を得る. 
        lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_model()
        lps_opt_list.append(lps_opt)

        # AP-MILPを解く ap_milp_opt, ap_milp_sol
        ap_milp.add_lps_dual_sol(lps_dual_sol)

        # 2 ~ 頂点総数までの分割制約でAP-MILPを解く
        for k in range(2, len(vertices) + 1):
            ap_milp.update_partition_constr(k)
            ap_milp_opt, ap_milp_sol = ap_milp.solve_model()

            # ap_milp_opt が正なら列に追加する
            if (ap_milp_opt > 10e-6):
                frozen_C = frozenset(u for u, x_val in ap_milp_sol["x_u"].items() if x_val == 1.0)
                cloumns[k] = (frozen_C)

        # 終了条件 cloumns が空なら stop
        if len(cloumns) == 0:
            break

        # LPSを更新
        for k in cloumns.keys():
            S = lps.update_model(cloumns[k])

        # DEBUG
        print(f"cnt: {cnt}")
        print(f"lps_opt: {lps_opt}")
        print(f"lps_primal_sol: {lps_primal_sol}")
        print(f"colums: {cloumns}")
        print(f"S: {S}")

        ap_milp.write_model(f"./data/output_data/ap_milp_{cnt}.lp")

        # cloumnsの初期化
        cloumns = {}

        cnt += 1

    # # TEST 列生成の生成する集合の要素数を固定する
    # # LP(S)を解く, 最適値, 主問題の解, 双対問題の解を得る. 
    # lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_model()
    # lps_opt_list.append(lps_opt)

    # # AP-MILPを解く ap_milp_opt, ap_milp_sol
    # ap_milp.add_lps_dual_sol(lps_dual_sol)

    # # DEBUG AP-MILP の制約の更新の確認
    # ap_milp.update_partition_constr(len(vertices) - 1)
    # ap_milp.debag_print_model()
    
    # ap_milp.update_partition_constr(len(vertices) - 2)
    # ap_milp.debag_print_model()