import numpy as np

from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph
from utils.partition import generate_singleton
from utils.wc import calc_w_C

def test_calc_w_C():
    Adj = read_csv_as_numpy("./data/test_data/01_Slovene_AdjMat.csv")

    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)

    lambda_val = 0.5
    init_partitions = [generate_singleton(vertices)]

    results = {}
    for partition in init_partitions:
        for C in partition:
            frozen_C = frozenset(C)
            if frozen_C not in results:
                results[frozen_C] = calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val)

    expected = {
        frozenset({0}): 1.0,
        frozenset({1}): 1.0,
        frozenset({2}): 1.0,
        frozenset({3}): 1.0,
        frozenset({4}): 3.0,
        frozenset({5}): 1.0,
        frozenset({6}): 3.0,
        frozenset({7}): 1.0,
        frozenset({8}): 1.0,
        frozenset({9}): 5.0
    }

    assert results == expected
