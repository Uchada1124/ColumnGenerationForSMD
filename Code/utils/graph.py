import numpy as np
import networkx as nx

def generate_signed_graph(A):
    """
    符号付き隣接行列 A をもとに符号付きグラフを生成する

    Returns:
        G: networkx グラフ
        vertices: 頂点のリスト
        A_plus: 正の隣接行列
        A_minus: 負の隣接行列
        D_plus: 正の次数行列
        D_minus: 負の次数行列
    """
    num_nodes = A.shape[0]
    vertices = list(range(num_nodes))

    # A_plus, A_minus の計算
    A_plus = (A > 0).astype(int)
    A_minus = (A < 0).astype(int)

    # D_plus, D_minus の計算
    D_plus = np.sum(A_plus, axis=1)
    D_minus = np.sum(A_minus, axis=1)

    # networkx グラフを構築
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # 上三角行列だけを参照する
            if A_plus[i, j] == 1:
                G.add_edge(i, j, sign=1)
            elif A_minus[i, j] == 1:
                G.add_edge(i, j, sign=-1)

    return G, vertices, A_plus, A_minus, D_plus, D_minus