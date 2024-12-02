import numpy as np
from mip import Model, xsum, maximize, BINARY

import utils.graph_util as graph_util
import utils.partition_util as partition_util

def calc_w_c(C, lamda, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    C = np.array(C)
    
    A_plus_C = adj_matrix_positive[np.ix_(C, C)]
    A_minus_C = adj_matrix_negative[np.ix_(C, C)]
    D_plus_C = degree_matrix_positive[C, C]
    D_minus_C = degree_matrix_negative[C, C]

    size_C = len(C)

    w_C = (
        (2 * np.sum(A_plus_C)) / size_C
        - (2 * (1 - lamda) * np.sum(D_plus_C)) / size_C
        - (2 * np.sum(A_minus_C)) / size_C
        + (2 * lamda * np.sum(D_minus_C)) / size_C
    )

    return w_C

def solve_lp_s(partitions, lamda, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative):
    model = Model(sense=maximize, solver_name="CBC")

    all_communities = []
    w_c_values = []
    for partition in partitions:
        for C in partition:
            w_c = calc_w_c(C, lamda, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative)
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
    lamda = 0.5
    objective_value, solution, all_communities = solve_lp_s(
        partitions, lamda, adj_matrix_positive, adj_matrix_negative, degree_matrix_positive, degree_matrix_negative
    )
    
    # Print and plot results
    print("Objective Value:", objective_value)
    result_partition = []
    for i, z in enumerate(solution):
        if z > 0.5: 
            print(f"z_{i} (for community {all_communities[i]}): {z}")
            result_partition.append(all_communities[i])
    graph_util.plot_partitioned_graph(graph, result_partition)

if __name__ == '__main__':
    main()