from utils.wc import calc_w_C
from utils.lps import LPS
from utils.ap_milp import AP_MILP

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
    - cg_opt: 最終的な最適値
    - cg_sol: 最終的な解
    - lps_opt_list: 各イテレーションのLPSの最適値リスト
    - S: 最終的な列集合
    - cnt: 反復回数
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

    ap_milp = AP_MILP(vertices, A_plus, A_minus, D_plus, D_minus, lambda_val)

    cnt = 0
    lps_opt_list = []
    cg_opt = 0
    cg_sol = {}

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
        new_w_C_dict = {}
        frozen_C = frozenset(u for u, x_val in ap_milp_sol["x_u"].items() if x_val == 1.0)
        if frozen_C not in w_C_dict:
            new_w_C_dict[frozen_C] = calc_w_C(
                list(frozen_C), A_plus, A_minus, D_plus, D_minus, lambda_val
            )
        new_S = list(new_w_C_dict.keys())
        S += new_S

        # LPSを更新
        lps.update_model(new_S, new_w_C_dict)

        # カウントの更新
        cnt+=1

    return cg_opt, cg_sol, lps_opt_list, S, cnt