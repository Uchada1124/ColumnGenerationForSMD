import numpy as np

from utils.graph import generate_signed_graph

# ダミーの隣接行列を作成
TEST_MATRIX = np.array([
    [ 0.0, -1.0,  1.0],
    [-1.0,  0.0, -1.0],
    [ 1.0, -1.0,  0.0]
])

def test_generate_signed_graph():
    """
    generate_signed_graph のテスト
    """
    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(TEST_MATRIX)

    # 頂点数
    assert len(vertices) == 3

    # 正のエッジ数
    assert np.sum(A_plus) // 2 == 1  # (0,2)

    # 負のエッジ数
    assert np.sum(A_minus) // 2 == 2  # (0,1), (1,2)

    # 正の次数
    assert np.array_equal(D_plus, [1, 0, 1])

    # 負の次数
    assert np.array_equal(D_minus, [1, 2, 1])

    # Graphのエッジ数
    assert len(G.edges) == 3

    # エッジの符号が正しく設定されているか
    assert G[0][1]["sign"] == -1
    assert G[0][2]["sign"] == 1
    assert G[1][2]["sign"] == -1
