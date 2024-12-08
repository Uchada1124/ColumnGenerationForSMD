import numpy as np
from mip import Model, xsum, maximize, minimize, BINARY, CONTINUOUS

import utils.graph_util as graph_util
import utils.partition_util as partition_util

def calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    C = np.array(C)
    
    A_plus_C = adj_matrix_positive[np.ix_(C, C)]
    A_minus_C = adj_matrix_negative[np.ix_(C, C)]
    D_plus_C = degree_matrix_positive[C, C]
    D_minus_C = degree_matrix_negative[C, C]

    size_C = len(C)

    w_C = (
        (2 * np.sum(A_plus_C)) / size_C
        - (2 * (1 - lambda_val) * np.sum(D_plus_C)) / size_C
        - (2 * np.sum(A_minus_C)) / size_C
        + (2 * lambda_val * np.sum(D_minus_C)) / size_C
    )

    return w_C

def solve_lp_s(partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    model = Model(sense=maximize, solver_name="CBC")

    all_communities = []
    w_c_values = []
    for partition in partitions:
        for C in partition:
            w_c = calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
            w_c_values.append(w_c)
            all_communities.append(C)

    z_vars = [model.add_var(name=f"z_{i}", var_type=BINARY) for i in range(len(all_communities))]

    model.objective = xsum(w_c_values[i] * z_vars[i] for i in range(len(all_communities)))

    num_vertices = adj_matrix_positive.shape[0]
    for u in range(num_vertices):
        model += xsum(z_vars[i] for i, C in enumerate(all_communities) if u in C) == 1

    model.optimize()

    solution = [z_var.x for z_var in z_vars]
    objective_value = model.objective_value

    return objective_value, solution, all_communities

def solve_ld_s(partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    model = Model(sense=minimize, solver_name="CBC")

    all_communities = []
    w_c_values = []
    for partition in partitions:
        for C in partition:
            w_c = calc_w_c(C, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
            w_c_values.append(w_c)
            all_communities.append(C)

    num_vertices = adj_matrix_positive.shape[0]
    y_vars = [model.add_var(name=f"y_{u}") for u in range(num_vertices)]

    model.objective = xsum(y_vars[u] for u in range(num_vertices))

    for i, C in enumerate(all_communities):
        model += xsum(y_vars[u] for u in C) >= w_c_values[i]

    model.optimize()

    solution = [y_var.x for y_var in y_vars]
    objective_value = model.objective_value

    return objective_value, solution

def generate_edges(adj_matrix_positive, adj_matrix_negative):
    num_vertices = adj_matrix_positive.shape[0]
    edges = [
        (u, v)
        for u in range(num_vertices)
        for v in range(u + 1, num_vertices)
        if adj_matrix_positive[u, v] > 0 or adj_matrix_negative[u, v] > 0
    ]
    return edges

def solve_ap_qp_milp(vertices, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative, lambda_val):
    edges = generate_edges(adj_matrix_positive, adj_matrix_negative)
    
    model = Model(sense=maximize, solver_name="CBC")

    # 変数の定義
    x_vars = {u: model.add_var(var_type=BINARY, name=f"x_{u}") for u in vertices}
    alpha_vars = {u: model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"alpha_{u}") for u in vertices}
    w_vars = {(u, v): model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"w_{u}_{v}") for u, v in edges}
    s_var = model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name="s")

    # 目的関数の構築
    model.objective = (
        4 * xsum(w_vars[u, v] for u, v in edges if adj_matrix_positive[u, v] > 0)  # 正のエッジの寄与
        - 2 * (1 - lambda_val) * xsum(degree_matrix_positive[u, u] * alpha_vars[u] for u in vertices)  # 正の次数
        - 4 * xsum(w_vars[u, v] for u, v in edges if adj_matrix_negative[u, v] > 0)  # 負のエッジの寄与
        + 2 * lambda_val * xsum(degree_matrix_negative[u, u] * alpha_vars[u] for u in vertices)  # 負の次数
        - xsum(alpha_vars[u] for u in vertices)  # alpha のコスト
    )

    # 制約の追加
    for u in vertices:
        model += s_var - (1 - x_vars[u]) <= alpha_vars[u], f"alpha_lower_bound_{u}"
        model += alpha_vars[u] <= x_vars[u], f"alpha_upper_bound_{u}"

    model += xsum(alpha_vars[u] for u in vertices) == 1, "alpha_sum_constraint"

    for u, v in edges:
        model += w_vars[u, v] <= alpha_vars[u], f"w_alpha_u_{u}_{v}"
        model += w_vars[u, v] <= alpha_vars[v], f"w_alpha_v_{u}_{v}"
        model += alpha_vars[u] - (2 - x_vars[u] - x_vars[v]) <= w_vars[u, v], f"w_alpha_constraint_{u}_{v}"

    # s の範囲制約
    model += s_var >= 0, "s_lower_bound"
    model += s_var <= 1, "s_upper_bound"

    # 最適化
    model.optimize()

    # 解の取得
    solution = {u: x_vars[u].x for u in vertices}
    objective_value = model.objective_value

    return objective_value, solution, s_var.x


def main():
    # Parameters for graph generation
    num_nodes = 10
    edge_prob = 0.4
    
    # Generate the signed graph
    (nodes, adj_matrix_positive, adj_matrix_negative,
     degree_matrix_positive, degree_matrix_negative, graph) = graph_util.generate_signed_graph(num_nodes, edge_prob)

    # Generate multiple unique random partitions
    vertices = list(range(num_nodes))
    num_partitions = 3
    num_samples = 5
    partitions = partition_util.generate_unique_partitions(vertices, num_partitions, num_samples)

    # Solve LP(S)
    lambda_val = 0.5
    objective_value_lp, solution_lp, all_communities = solve_lp_s(
        partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative
    )
    print("LP(S) Objective Value:", objective_value_lp)
    print("Dual Variables (y):", solution_lp)

    # Solve LD(S)
    objective_value_ld, solution_ld = solve_ld_s(
        partitions, lambda_val, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative
    )
    print("LD(S) Objective Value:", objective_value_ld)
    print("Dual Variables (y):", solution_ld)
    
    # Solve AP-QP MILP
    objective_value, solution, s_value = solve_ap_qp_milp(
        vertices=nodes,
        adj_matrix_positive=adj_matrix_positive,
        adj_matrix_negative=adj_matrix_negative,
        degree_matrix_positive=degree_matrix_positive,
        degree_matrix_negative=degree_matrix_negative,
        lambda_val=lambda_val
    )

    print("Objective Value:", objective_value)
    print("Solution for x:", solution)
    print("Value of s:", s_value)
    
    # Print and plot results
    print("Objective Value:", objective_value_lp)
    result_partition = []
    for i, z in enumerate(solution_lp):
        if z > 0.5: 
            print(f"z_{i} (for community {all_communities[i]}): {z}")
            result_partition.append(all_communities[i])
    graph_util.plot_partitioned_graph(graph, result_partition)

if __name__ == '__main__':
    main()