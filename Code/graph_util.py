import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_signed_graph(num_nodes, edge_prob=0.3):
    G = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=False)
    
    for u, v in G.edges():
        G[u][v]['sign'] = np.random.choice([-1, 1])
    
    adjacency_matrix_positive = nx.to_numpy_array(G, weight='sign')
    adjacency_matrix_positive[adjacency_matrix_positive < 0] = 0
    
    adjacency_matrix_negative = nx.to_numpy_array(G, weight='sign')
    adjacency_matrix_negative[adjacency_matrix_negative > 0] = 0
    adjacency_matrix_negative = np.abs(adjacency_matrix_negative)
    
    degree_matrix_positive = np.diag(np.sum(adjacency_matrix_positive, axis=1))
    degree_matrix_negative = np.diag(np.sum(adjacency_matrix_negative, axis=1))
    
    return list(G.nodes()), adjacency_matrix_positive, adjacency_matrix_negative, degree_matrix_positive, degree_matrix_negative, G

def plot_signed_graph(G):
    """
    Visualize a signed graph with colors indicating edge signs.
    
    Args:
        G (networkx.Graph): The graph to visualize.
    """
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
    colors = plt.cm.get_cmap('tab10', len(partition))

    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 1]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == -1]
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, edge_color="red", label="Positive Edges")
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, edge_color="blue", style="dashed", label="Negative Edges")
    
    for i, community in enumerate(partition):
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=[colors(i)], label=f"Community {i+1}", node_size=500)

    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
    
    plt.legend(loc="best")
    plt.title("Partitioned Graph Visualization")
    plt.show()

def main():
    num_nodes = 10
    edge_prob = 0.4
    (nodes, adj_matrix_positive, adj_matrix_negative,
     degree_matrix_positive, degree_matrix_negative, graph) = generate_signed_graph(num_nodes, edge_prob)
    
    print("Nodes:", nodes)
    print("Positive Adjacency Matrix (A+):\n", adj_matrix_positive)
    print("Negative Adjacency Matrix (A-):\n", adj_matrix_negative)
    print("Positive Degree Matrix (D+):\n", degree_matrix_positive)
    print("Negative Degree Matrix (D-):\n", degree_matrix_negative)
    
    plot_signed_graph(graph)

    partition = [[0, 2, 8], [1, 3, 5, 6], [4, 7, 9] ]
    plot_partitioned_graph(graph, partition)

if __name__ == "__main__":
    main()
