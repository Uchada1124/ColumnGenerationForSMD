import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps

import numpy as np
import networkx as nx

# 内側にも負のエッジもはやす

def generate_signed_graph(block_sizes, p_in=0.9, p_out=0.1):
    num_blocks = len(block_sizes)
    block_offsets = np.cumsum([0] + block_sizes[:-1])
    total_nodes = sum(block_sizes)

    adjacency_matrix_positive = np.zeros((total_nodes, total_nodes))
    adjacency_matrix_negative = np.zeros((total_nodes, total_nodes))

    for b in range(num_blocks):
        start, end = block_offsets[b], block_offsets[b] + block_sizes[b]
        for i in range(start, end):
            for j in range(i + 1, end):
                if np.random.rand() < p_in:
                    adjacency_matrix_positive[i, j] = 1
                    adjacency_matrix_positive[j, i] = 1

    for b1 in range(num_blocks):
        for b2 in range(b1 + 1, num_blocks):
            start1, end1 = block_offsets[b1], block_offsets[b1] + block_sizes[b1]
            start2, end2 = block_offsets[b2], block_offsets[b2] + block_sizes[b2]
            for i in range(start1, end1):
                for j in range(start2, end2):
                    if np.random.rand() < p_out:
                        adjacency_matrix_negative[i, j] = 1
                        adjacency_matrix_negative[j, i] = 1

    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))

    for i in range(total_nodes):
        for j in range(i + 1, total_nodes):
            if adjacency_matrix_positive[i, j] > 0:
                G.add_edge(i, j, sign=1)
            elif adjacency_matrix_negative[i, j] > 0:
                G.add_edge(i, j, sign=-1)

    degree_matrix_positive = np.diag(np.sum(adjacency_matrix_positive, axis=1))
    degree_matrix_negative = np.diag(np.sum(adjacency_matrix_negative, axis=1))

    return list(G.nodes()), adjacency_matrix_positive, adjacency_matrix_negative, degree_matrix_positive, degree_matrix_negative, G


def generate_edges(adj_matrix_positive, adj_matrix_negative):
    num_vertices = adj_matrix_positive.shape[0]
    edges = [
        (u, v)
        for u in range(num_vertices)
        for v in range(u + 1, num_vertices)
        if adj_matrix_positive[u, v] > 0 or adj_matrix_negative[u, v] > 0
    ]
    return edges

def plot_signed_graph(G):
    pos = nx.spring_layout(G)

    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == -1]

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")

    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color="red", label="Positive Edges")

    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color="blue", style="dashed", label="Negative Edges")

    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    plt.legend(loc="best")
    plt.title("Signed Graph Visualization")
    plt.show()

def plot_partitioned_graph(G, partition):
    pos = nx.spring_layout(G)
    colors = colormaps['tab10'](np.linspace(0, 1, len(partition)))

    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == -1]
    
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color="red", label="Positive Edges")
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color="blue", style="dashed", label="Negative Edges")

    for i, community in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[colors[i]], label=f"Community {i+1}", node_size=500)

    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")

    plt.legend(loc="best")
    plt.title("Partitioned Graph Visualization")
    plt.show()

def main():
    block_sizes = [5, 7, 8]
    p_in = 0.8
    p_out = 0.1

    (nodes, adj_matrix_positive, adj_matrix_negative,
     degree_matrix_positive, degree_matrix_negative, graph) = generate_signed_graph(block_sizes, p_in, p_out)

    print("Nodes:", nodes)
    print("Positive Adjacency Matrix (A+):\n", adj_matrix_positive)
    print("Negative Adjacency Matrix (A-):\n", adj_matrix_negative)
    print("Positive Degree Matrix (D+):\n", degree_matrix_positive)
    print("Negative Degree Matrix (D-):\n", degree_matrix_negative)

    edges = generate_edges(adj_matrix_positive, adj_matrix_negative)
    print("Edges:\n", edges)

    plot_signed_graph(graph)

    partition = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19]]
    plot_partitioned_graph(graph, partition)

if __name__ == "__main__":
    main()
