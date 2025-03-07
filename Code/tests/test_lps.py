import pytest
import numpy as np
from mip import OptimizationStatus
from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph
from utils.wc import calc_w_C
from utils.lps import LPS

@pytest.fixture
def lps_instance():
    """LPS インスタンスを作成する (テストデータ使用)"""
    Adj = read_csv_as_numpy("./data/test_data/01_Slovene_AdjMat.csv")
    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)
    lambda_val = 0.5

    init_partitions = [[[u] for u in vertices]]
    w_C_dict = {
        frozenset(C): calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val)
        for partition in init_partitions for C in partition
    }
    S = list(w_C_dict.keys())

    lps = LPS(S, w_C_dict, vertices)

    return lps, w_C_dict, S, vertices, A_plus, A_minus, D_plus, D_minus, lambda_val

def test_lps_init(lps_instance):
    """LPSの初期化が正しく行われるか"""
    lps, w_C_dict, S, vertices, _, _, _, _, _ = lps_instance

    expected_w_C_dict = {
        frozenset({0}): np.float64(1.0),
        frozenset({1}): np.float64(1.0),
        frozenset({2}): np.float64(1.0),
        frozenset({3}): np.float64(1.0),
        frozenset({4}): np.float64(3.0),
        frozenset({5}): np.float64(1.0),
        frozenset({6}): np.float64(3.0),
        frozenset({7}): np.float64(1.0),
        frozenset({8}): np.float64(1.0),
        frozenset({9}): np.float64(5.0)
    }

    assert isinstance(lps, LPS)
    assert lps.w_C_dict == expected_w_C_dict
    assert set(lps.S) == set(expected_w_C_dict.keys())
    assert set(lps.vertices) == set(vertices)

def test_solve_lps(lps_instance):
    """LPSを解いたときの出力が期待値と一致するか"""
    lps, _, _, _, _, _, _, _, _ = lps_instance

    lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_lps()

    expected_optimal_value = 18.0
    expected_primal_sol = {
        frozenset({0}): 1.0, frozenset({1}): 1.0, frozenset({2}): 1.0, frozenset({3}): 1.0,
        frozenset({4}): 1.0, frozenset({5}): 1.0, frozenset({6}): 1.0, frozenset({7}): 1.0,
        frozenset({8}): 1.0, frozenset({9}): 1.0
    }
    expected_dual_sol = {
        0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 3.0,
        5: 1.0, 6: 3.0, 7: 1.0, 8: 1.0, 9: 5.0
    }

    assert lps.model.status == OptimizationStatus.OPTIMAL
    assert lps_opt == expected_optimal_value
    assert lps_primal_sol == expected_primal_sol
    assert lps_dual_sol == expected_dual_sol

def test_update_lps(lps_instance):
    """LPSの更新後の出力が期待値と一致するか"""
    lps, w_C_dict, S, vertices, A_plus, A_minus, D_plus, D_minus, lambda_val = lps_instance

    # 新しい列の追加
    new_w_C_dict = {
        frozenset({0, 1, 2}): np.float64(-0.3333333333333333)
    }
    new_S = list(new_w_C_dict.keys())

    lps.update_lps(new_S, new_w_C_dict)

    # 期待される更新後の w_C_dict
    expected_updated_w_C_dict = {
        **w_C_dict,
        **new_w_C_dict
    }
    
    # 期待される最適化結果
    expected_optimal_value = 18.0
    expected_primal_sol = {
        frozenset({0}): 1.0, frozenset({1}): 1.0, frozenset({2}): 1.0, frozenset({3}): 1.0,
        frozenset({4}): 1.0, frozenset({5}): 1.0, frozenset({6}): 1.0, frozenset({7}): 1.0,
        frozenset({8}): 1.0, frozenset({9}): 1.0, frozenset({0, 1, 2}): 0.0
    }
    
    lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_lps()

    # 更新後のチェック
    assert set(lps.w_C_dict.keys()) == set(expected_updated_w_C_dict.keys())
    assert lps.w_C_dict == expected_updated_w_C_dict
    assert set(lps.S) == set(expected_updated_w_C_dict.keys())

    # LPS解のチェック
    assert lps.model.status == OptimizationStatus.OPTIMAL
    assert lps_opt == expected_optimal_value
    assert lps_primal_sol == expected_primal_sol
