import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph
from utils.wc import calc_w_C
from utils.column_generation import column_generation

def main():
    # 隣接行列のデータを読込む
    Adj = read_csv_as_numpy("./data/01_Slovene_AdjMat.csv")

    # 隣接行列をもとにグラフを生成
    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)

    # # データの確認
    # print("頂点数:", len(vertices))
    # print("正のエッジ数:", np.sum(A_plus) // 2)
    # print("負のエッジ数:", np.sum(A_minus) // 2)
    # print("D+ (正の次数):", D_plus)
    # print("D- (負の次数):", D_minus)

    # # グラフの可視化
    # pos = nx.spring_layout(G)  # ノード配置
    # edge_colors = ["red" if G[u][v]["sign"] == -1 else "blue" for u, v in G.edges()]
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color="lightgray", node_size=500)
    # plt.title("符号付きネットワークの可視化")
    # plt.show()

    # # w_Cの確認
    # print(A_plus)
    # print(A_minus)
    # lambda_val = 0.5
    # C = [0, 1, 2]
    # w_C = calc_w_C(
    #     C, A_plus, A_minus, D_plus, D_minus, lambda_val
    # )
    # print(w_C)

    lambda_val = 0.5
    init_partitions = [[[u] for u in vertices]]
    # print(init_partitions)
    column_generation(
        vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, init_partitions
    )

if __name__ == '__main__':
    main()
