import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph
from utils.partition import generate_singleton
from utils.column_generation import column_generation

def main():
    # 隣接行列のデータを読込む
    Adj = read_csv_as_numpy("./data/01_Slovene_AdjMat.csv")

    # 隣接行列をもとにグラフを生成
    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)

    # データの確認
    print("頂点数:", len(vertices))
    print("正のエッジ数:", np.sum(A_plus) // 2)
    print("負のエッジ数:", np.sum(A_minus) // 2)
    print("D+ (正の次数):", D_plus)
    print("D- (負の次数):", D_minus)

    # グラフの可視化
    pos = nx.spring_layout(G)  # ノード配置
    edge_colors = ["red" if G[u][v]["sign"] == -1 else "blue" for u, v in G.edges()]
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color="lightgray", node_size=500)
    plt.title("Signed Network")
    plt.show()

    # 列生成法
    lambda_val = 0.5
    init_partitions = [generate_singleton(vertices)]
    cg_opt, cg_sol, lps_opt_list, S, cnt = column_generation(
        vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, init_partitions
    )

    # 結果の出力
    print("\n=== 列生成法の結果 ===")
    print(f"反復回数: {cnt}")
    print(f"最終目的関数値: {cg_opt:.4f}")
    print("最終解 (z_C):")
    for C, value in cg_sol.items():
        print(f"  {set(C)}: {value:.4f}")

    # LPSの最適値の推移をプロット
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(lps_opt_list)), lps_opt_list, marker='o')
    plt.xlabel("Count")
    plt.ylabel("LPS Objective Value")
    plt.title("LPS Objective Value Progression")
    plt.grid(True)
    plt.show()

    # 最終的な分割の可視化
    final_partition = [list(C) for C in cg_sol.keys() if cg_sol[C] > 0.1]
    color_map = ["lightgray"] * len(vertices)
    colors = ["blue", "green", "red", "orange", "purple", "pink", "yellow"]

    for idx, community in enumerate(final_partition):
        for node in community:
            color_map[node] = colors[idx % len(colors)]

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color=color_map, node_size=500)
    plt.title("Partitioned Signed Network")
    plt.show()

if __name__ == '__main__':
    main()
