import numpy as np
from mip import Model, xsum, maximize, minimize, BINARY, CONTINUOUS

import utils.graph_util as graph_util

def calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    """
    各クラスター C に対して重み w_C を計算する関数
    """
    C = np.array(C)
    
    # クラスター内の正負の隣接行列と次数行列を取得
    A_plus_C = adj_matrix_positive[np.ix_(C, C)]
    A_minus_C = adj_matrix_negative[np.ix_(C, C)]
    D_plus_C = degree_matrix_positive[C, C]
    D_minus_C = degree_matrix_negative[C, C]

    size_C = len(C)

    # 重み w_C を計算
    w_C = (
        (2 * np.sum(A_plus_C)) / size_C
        - (2 * (1 - lambda_val) * np.sum(D_plus_C)) / size_C
        - (2 * np.sum(A_minus_C)) / size_C
        + (2 * lambda_val * np.sum(D_minus_C)) / size_C
    )

    return w_C

def solve_lp_s(partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    """
    制約付き線形計画問題 LP(S) を解く関数
    """
    model = Model(sense=maximize, solver_name="CBC")

    # 全てのコミュニティとそれに対応する重みを計算
    all_communities = []
    w_c_values = []
    for partition in partitions:
        for C in partition:
            w_c = calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
            w_c_values.append(w_c)
            all_communities.append(C)

    # z 変数を作成（バイナリ変数）
    z_vars = [model.add_var(name=f"z_{i}", var_type=BINARY) for i in range(len(all_communities))]

    # 目的関数を定義
    model.objective = xsum(w_c_values[i] * z_vars[i] for i in range(len(all_communities)))

    # 制約を追加（各頂点がちょうど1つのクラスターに属するようにする）
    num_vertices = adj_matrix_positive.shape[0]
    for u in range(num_vertices):
        model += xsum(z_vars[i] for i, C in enumerate(all_communities) if u in C) == 1

    # 最適化を実行
    model.optimize()

    # 解を取得
    solution = [z_var.x for z_var in z_vars]
    objective_value = model.objective_value

    return objective_value, solution, all_communities

def solve_ld_s(partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    """
    双対問題 LD(S) を解く関数
    """
    model = Model(sense=minimize, solver_name="CBC")

    # 全てのコミュニティとその重みを計算
    all_communities = []
    w_c_values = []
    for partition in partitions:
        for C in partition:
            w_c = calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
            w_c_values.append(w_c)
            all_communities.append(C)

    # 双対変数 y を作成
    num_vertices = adj_matrix_positive.shape[0]
    y_vars = [model.add_var(name=f"y_{u}") for u in range(num_vertices)]

    # 目的関数を定義
    model.objective = xsum(y_vars[u] for u in range(num_vertices))

    # 制約を追加（双対条件を満たすようにする）
    for i, C in enumerate(all_communities):
        model += xsum(y_vars[u] for u in C) >= w_c_values[i]

    # 最適化を実行
    model.optimize()

    # 解を取得
    solution = [y_var.x for y_var in y_vars]
    objective_value = model.objective_value

    return objective_value, solution

def solve_ap_qp_milp(vertices, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative, lambda_val, solution_ld):
    """
    AP-QP の整数線形計画を解く関数
    """
    edges = graph_util.generate_edges(adj_matrix_positive, adj_matrix_negative)
    
    model = Model(sense=maximize, solver_name="CBC")

    # 変数の定義
    x_vars = {u: model.add_var(var_type=BINARY, name=f"x_{u}") for u in vertices}
    alpha_vars = {u: model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"alpha_{u}") for u in vertices}
    w_vars = {(u, v): model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"w_{u}_{v}") for u, v in edges}
    s_var = model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name="s")

    # 目的関数を構築
    model.objective = (
        4 * xsum(w_vars[u, v] for u, v in edges if adj_matrix_positive[u, v] > 0)
        - 2 * (1 - lambda_val) * xsum(degree_matrix_positive[u, u] * alpha_vars[u] for u in vertices)
        - 4 * xsum(w_vars[u, v] for u, v in edges if adj_matrix_negative[u, v] > 0)
        + 2 * lambda_val * xsum(degree_matrix_negative[u, u] * alpha_vars[u] for u in vertices)
        - xsum(solution_ld[u] * x_vars[u] for u in vertices)
    )

    # 制約を追加
    for u in vertices:
        model += s_var - (1 - x_vars[u]) <= alpha_vars[u], f"alpha_lower_bound_{u}"
        model += alpha_vars[u] <= x_vars[u], f"alpha_upper_bound_{u}"

    model += xsum(alpha_vars[u] for u in vertices) == 1, "alpha_sum_constraint"

    for u, v in edges:
        model += w_vars[u, v] <= alpha_vars[u], f"w_alpha_u_{u}_{v}"
        model += w_vars[u, v] <= alpha_vars[v], f"w_alpha_v_{u}_{v}"
        model += alpha_vars[u] - (2 - x_vars[u] - x_vars[v]) <= w_vars[u, v], f"w_alpha_{u}_constraint_{u}_{v}"
        model += alpha_vars[v] - (2 - x_vars[u] - x_vars[v]) <= w_vars[u, v], f"w_alpha_{v}_constraint_{u}_{v}"

    # 最適化を実行
    model.optimize()

    # 解を取得
    solution = {u: x_vars[u].x for u in vertices}
    objective_value = model.objective_value

    return objective_value, solution, s_var.x

def column_generation(vertices, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative, lambda_val, initial_partitions):
    """
    列生成法
    """
    # Step 1: 初期の列集合 S を設定する. 
    S = initial_partitions

    while True:
        # Step 2: 線形計画問題 LP(S), LD(S) を解く. LD(S)の解き方を相談する. 
        objective_value_lp, solution_lp, all_communities = solve_lp_s(
            partitions=S,
            adj_matrix_positive=adj_matrix_positive,
            adj_matrix_negative=adj_matrix_negative,
            degree_matrix_positive=degree_matrix_positive,
            degree_matrix_negative=degree_matrix_negative,
            lambda_val=lambda_val
        )
        print("LP(S) Objective Value:", objective_value_lp)
    
        objective_value_ld, solution_ld = solve_ld_s(
            partitions=S,
            adj_matrix_positive=adj_matrix_positive,
            adj_matrix_negative=adj_matrix_negative,
            degree_matrix_positive=degree_matrix_positive,
            degree_matrix_negative=degree_matrix_negative,
            lambda_val=lambda_val
        )
        print("LD(S) Objective Value:", objective_value_ld)

        # Step 3: 補助問題 AP-QP を解く. 
        objective_value_ap_qp, solution_ap_qp, _ = solve_ap_qp_milp(
            vertices=vertices,
            adj_matrix_positive=adj_matrix_positive,
            adj_matrix_negative=adj_matrix_negative,
            degree_matrix_positive=degree_matrix_positive,
            degree_matrix_negative=degree_matrix_negative,
            lambda_val=lambda_val,
            solution_ld=solution_ld
        )
        print("AP-QP Objective Value:", objective_value_ap_qp)

        # Step 4: 終了条件を確認する. 
        if objective_value_ap_qp <= 0:
            print("Column Generation terminated: no column with positive reduced cost.")
            break

        # Step 5: S を更新する. 
        new_column = [u for u in vertices if solution_ap_qp[u] > 0.5]
        print(f"New column added to S: {new_column}")
        S.append([new_column])  # 新しいパーティションを追加する. 

    return S, objective_value_lp, solution_lp, all_communities