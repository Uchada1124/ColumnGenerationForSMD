import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_graph(G, title="Signed Graph"):
    """
    グラフを可視化する

    Parameters
    ----------
    G : networkx.Graph
        グラフ
    title : str
        グラフのタイトル
    """
    pos = nx.spring_layout(G)  # ノードの位置を自動配置
    edge_colors = ['red' if G[u][v]['sign'] > 0 else 'blue' for u, v in G.edges()]
    node_colors = ['lightgray'] * len(G.nodes())

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color=node_colors, node_size=500)
    plt.title(title)
    plt.show()

def plot_lps_objective(lps_opt_list):
    """
    LPSの最適値の推移をプロットする

    Parameters
    ----------
    lps_opt_list : list
        LPSの最適値のリスト
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(lps_opt_list)), lps_opt_list, marker='o')
    plt.xlabel("Count")
    plt.ylabel("LPS Objective Value")
    plt.title("LPS Objective Value Progression")
    plt.grid(True)
    plt.show()

def plot_partitioned_graph(G, cg_sol, title="Partitioned Signed Network"):
    """
    最終的な分割の可視化

    Parameters
    ----------
    G : networkx.Graph
        グラフ
    cg_sol : dict
        最適解
    title : str
        グラフのタイトル
    """
    pos = nx.spring_layout(G)  # ノードの位置を自動配置
    edge_colors = ['red' if G[u][v]['sign'] > 0 else 'blue' for u, v in G.edges()]
    vertices = list(G.nodes())
    final_partition = [list(C) for C in cg_sol.keys() if cg_sol[C] > 0.1]
    color_map = ["lightgray"] * len(vertices)
    
    # 動的に色を取得
    cmap = plt.get_cmap('tab10')
    num_colors = cmap.N

    for idx, community in enumerate(final_partition):
        for node in community:
            color_map[node] = cmap(idx % num_colors)

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color=color_map, node_size=500)
    plt.title(title)
    plt.show()