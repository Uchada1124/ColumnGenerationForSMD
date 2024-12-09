import utils.graph_util as graph_util
import utils.partition_util as partition_util
import utils.column_generation_util as cg_util

def main():
    # グラフ生成のパラメータを設定する. 
    num_nodes = 10
    edge_prob = 0.4

    # 符号付ネットワークを生成する. 
    (nodes, adj_matrix_positive, adj_matrix_negative,
     degree_matrix_positive, degree_matrix_negative, graph) = graph_util.generate_signed_graph(num_nodes, edge_prob)

    # 初期の分割集合を生成する. 
    vertices = list(range(num_nodes))
    num_partitions = 3
    num_samples = 5
    initial_partitions = partition_util.generate_unique_partitions(vertices, num_partitions, num_samples)

    # ハイパーパラメータを設定する. 
    lambda_val = 0.5

    # 列生成法を実行する. 
    final_partitions, final_objective_value, final_solution, all_communities = cg_util.column_generation(
        vertices=vertices,
        adj_matrix_positive=adj_matrix_positive,
        adj_matrix_negative=adj_matrix_negative,
        degree_matrix_positive=degree_matrix_positive,
        degree_matrix_negative=degree_matrix_negative,
        lambda_val=lambda_val,
        initial_partitions=initial_partitions
    )

    # 結果を出力する. 
    print("Final Partitions:", final_partitions)
    print("Final Objective Value:", final_objective_value)
    print("Final Solution:", final_solution)
    print("All Communities", all_communities)
    result_partition = []
    for i, z in enumerate(final_solution):
        if z > 0.5: 
            print(f"z_{i} (for community {all_communities[i]}): {z}")
            result_partition.append(all_communities[i])
    graph_util.plot_partitioned_graph(graph, result_partition)

if __name__ == '__main__':
    main()
