import numpy as np

from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph

def test_generate_signed_graph():
    """
    generate_signed_graph のテスト
    """
    Adj = read_csv_as_numpy("./data/test_data/01_Slovene_AdjMat.csv")

    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)

    # 頂点数
    assert len(vertices) == 10

    # 正のエッジ数
    assert np.sum(A_plus) // 2 == 18 

    # 負のエッジ数
    assert np.sum(A_minus) // 2 == 27 

    # 正の次数
    assert np.array_equal(D_plus, [4, 4, 4, 4, 3, 4, 3, 4, 4, 2])

    # 負の次数
    assert np.array_equal(D_minus, [5, 5, 5, 5, 6, 5, 6, 5, 5, 7])
