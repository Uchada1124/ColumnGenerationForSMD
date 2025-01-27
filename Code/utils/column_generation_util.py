import numpy as np
from mip import Model, xsum, maximize, BINARY, CONTINUOUS, Column

import utils.graph_util as graph_util


def calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    """
    各クラスター C に対して重み w_C を計算する関数
    """
    C = np.array(C)
    A_plus_C = adj_matrix_positive[np.ix_(C, C)]
    A_minus_C = adj_matrix_negative[np.ix_(C, C)]
    D_plus_C = degree_matrix_positive[C, C]
    D_minus_C = degree_matrix_negative[C, C]
    size_C = len(C)

    # 各項の計算
    term1 = 2 * np.sum(A_plus_C)
    term2 = -2 * (1 - lambda_val) * np.sum(D_plus_C)
    term3 = 2 * np.sum(A_minus_C)
    term4 = -2 * lambda_val * np.sum(D_minus_C)
    w_C = (term1 + term2 - (term3 + term4)) / size_C

    print(C, term1, term2, term3, term4, w_C)

    return w_C

def calc_w_c_dict(partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    """
    全てのパーティションに対して, Key : C, Value ; w_C とする辞書を計算する関数
    """
    w_c_dict = {}
    for partition in partitions:
        for C in partition:
            w_c = calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
            w_c_dict[tuple(C)] = w_c
    return w_c_dict

def create_lp_s(partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    """
    LP(S) モデルを構築する関数
    """
    model = Model(sense=maximize, solver_name="CBC")
    # model.solver.set_verbose(False)

    w_c_dict = calc_w_c_dict(
        partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative
    )

    z_vars = {C: model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"z_{C}") for C in w_c_dict.keys()}

    model.objective = xsum(w_c_dict[C] * z_vars[C] for C in w_c_dict.keys())

    num_vertices = adj_matrix_positive.shape[0]
    constraints = {}
    for u in range(num_vertices):
        constr = model.add_constr(xsum(z_vars[C] for C in w_c_dict.keys() if u in C) == 1)
        constraints[u] = constr

    return model, z_vars, constraints

def add_column_to_lp_s(model, constraints, w_c, community, z_vars):
    """
    LPモデルに新しい列(変数と制約)を追加する関数
    """
    column = Column([1 if u in community else 0 for u, _ in enumerate(constraints)], constraints)

    z_new = model.add_var(var_type=CONTINUOUS, lb=0, ub=1, obj=w_c, column=column, name=f"z_{community}")

    z_vars[community] = z_new

    return z_new


def solve_lp_s(model, z_vars, constraints):
    """
    LP(S) モデルを解き、双対変数と結果を取得する関数
    """
    model.optimize()

    solution = {C: z_var.x for C, z_var in z_vars.items()}
    objective_value = model.objective_value

    dual_variables = {u: constraints[u].pi for u in constraints.keys()}
    
    for u, constr in constraints.items():
        print(f"Constraint {u}: Dual Variable = {constr.pi}")

    return objective_value, solution, dual_variables

def solve_ap_qp_milp(vertices, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative, lambda_val, dual_variables):
    """
    AP-QP の整数線形計画を解く関数
    """
    edges = graph_util.generate_edges(adj_matrix_positive, adj_matrix_negative)

    model = Model(sense=maximize, solver_name="CBC")
    model.solver.set_verbose(False)

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
        - xsum(dual_variables[u] * x_vars[u] for u in vertices)
    )

    # 制約を追加
    for u in vertices:
        model += s_var - (1 - x_vars[u]) <= alpha_vars[u], f"alpha_lower_bound_{u}"
        model += alpha_vars[u] <= s_var, f"alpha_upper_bound_s{u}"
        model += alpha_vars[u] <= x_vars[u], f"alpha_upper_bound_x{u}"

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

    print(model.objective)
    for c in model.constrs:
        print(c)

    return objective_value, solution, s_var.x

def solve_dual_lp(vertices, all_communities, w_c_dict):
    """
    双対問題 (Dual Problem) を解く関数
    """
    from mip import Model, xsum, minimize, CONTINUOUS

    model = Model(sense=minimize, solver_name="CBC")
    model.solver.set_verbose(False)

    y_vars = {u: model.add_var(var_type=CONTINUOUS, name=f"y_{u}") for u in vertices}

    model.objective = xsum(y_vars[u] for u in vertices)

    for C in all_communities:
        model.add_constr(xsum(y_vars[u] for u in C) >= w_c_dict[C], name=f"dual_constr_{C}")

    model.optimize()

    y_solution = {u: y_vars[u].x for u in vertices}
    objective_value = model.objective_value

    print(model.objective)
    for c in model.constrs:
        print(c)
    print("LP(D) Objective Value:", objective_value)
    print("LP(D) Solution:", y_solution) 

    return y_solution, objective_value

def column_generation(vertices, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative, lambda_val, initial_partitions):
    """
    列生成法を実行する関数
    """
    # モデルの初期化
    model, z_vars, constraints = create_lp_s(
        initial_partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative
    )

    S = initial_partitions

    while True:
        # LP(S)を解く
        objective_value_lp, solution_lp, dual_variables = solve_lp_s(model, z_vars, constraints)
        print(model.objective)
        for c in model.constrs:
            print(c)
        print("LP(S) Objective Value:", objective_value_lp)
        print("LP(S) Solution:", solution_lp)
        print("LP(S) Dual Variables:", dual_variables)

        # 双対問題を解いて検証
        w_c_dict = calc_w_c_dict(S, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
        solve_dual_lp(vertices, list(w_c_dict.keys()), w_c_dict)

        # AP-QPを解く
        objective_value_ap_qp, solution_ap_qp, _ = solve_ap_qp_milp(
            vertices=vertices,
            adj_matrix_positive=adj_matrix_positive,
            adj_matrix_negative=adj_matrix_negative,
            degree_matrix_positive=degree_matrix_positive,
            degree_matrix_negative=degree_matrix_negative,
            lambda_val=lambda_val,
            dual_variables = dual_variables
        )
        print("AP-QP Objective Value:", objective_value_ap_qp)

        # x_uを目的関数に代入してみて正負を確認
        # 終了条件を確認
        if objective_value_ap_qp <= 0:
            print("Column Generation terminated: no column with positive reduced cost.")
            break

        # 新しい列を追加
        new_community = tuple(u for u in vertices if solution_ap_qp[u] > 0.9)
        w_c = calc_w_c(new_community, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
        z_new = add_column_to_lp_s(model, constraints, w_c, new_community, z_vars)

        S.append([new_community])

    return S, objective_value_lp, solution_lp, list(z_vars.keys())
