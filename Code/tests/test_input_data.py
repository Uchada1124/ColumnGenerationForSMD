import numpy as np

from utils.input_data import read_csv_as_numpy

# ダミーの隣接行列を作成
TEST_MATRIX = np.array([
    [ 0.0, -1.0,  1.0],
    [-1.0,  0.0, -1.0],
    [ 1.0, -1.0,  0.0]
])

def test_read_csv_as_numpy(tmp_path):
    """
    read_csv_as_numpy のテスト
    """
    csv_content = "0.0,-1.0,1.0\n-1.0,0.0,-1.0\n1.0,-1.0,0.0\n"
    file = tmp_path / "test.csv"
    file.write_text(csv_content)

    matrix = read_csv_as_numpy(file)

    expected = np.array([
        [ 0.0, -1.0,  1.0],
        [-1.0,  0.0, -1.0],
        [ 1.0, -1.0,  0.0]
    ])

    assert np.array_equal(matrix, expected)
