import utils.graph_util as graph_util
import utils.partition_util as partition_util
import utils.column_generation_util as cg_util

def main():
    # グラフ生成のパラメータを設定する. 
    block_sizes = [5, 7, 8]
    p_in = 0.8
    p_out = 0.1
    
    # 符号付ネットワークを生成する. 
    (nodes, adj_matrix_positive, adj_matrix_negative,
     degree_matrix_positive, degree_matrix_negative, graph) = graph_util.generate_signed_graph(block_sizes, p_in, p_out)
    
    print("Positive Adjacency Matrix (A+):\n", adj_matrix_positive)
    print("Negative Adjacency Matrix (A-):\n", adj_matrix_negative)
    print("Positive Degree Matrix (D+):\n", degree_matrix_positive)
    print("Negative Degree Matrix (D-):\n", degree_matrix_negative)
    
    graph_util.plot_signed_graph(graph)

    # 初期の分割集合を生成する. 
    num_nodes = sum(block_sizes)
    vertices = list(range(num_nodes))
    initial_partitions = [partition_util.generate_singleton(vertices)]

    # initial_partitions = [
    # [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19]],
    # partition_util.generate_singleton(vertices)
    # ]

    # ハイパーパラメータを設定する. 
    lambda_val = 1.0

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

    # # 結果を出力する. 
    # print("Final Partitions:", final_partitions)
    # print("Final Objective Value:", final_objective_value)
    # print("Final Solution:", final_solution)
    # print("All Communities:", all_communities)

    # # 解から結果のパーティションを生成
    # result_partition = []
    # for community, z_value in final_solution.items():
    #     if z_value > 0.9:
    #         print(f"z (for community {community}): {z_value}")
    #         result_partition.append(community)

    # # プロット
    # graph_util.plot_partitioned_graph(graph, result_partition)

if __name__ == '__main__':
    main()
