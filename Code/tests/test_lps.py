from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph
from utils.wc import calc_w_C
from utils.lps import LPS

# 隣接行列のデータを読込む
Adj = read_csv_as_numpy("./data/01_Slovene_AdjMat.csv")

# 隣接行列をもとにグラフを生成
G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)

lambda_val = 0.5
init_partitions = [[[u] for u in vertices]]

# LPSの解説

# 初期化
w_C_dict = {}
for partition in init_partitions:
    for C in partition:
        frozen_C = frozenset(C)
        if frozen_C not in w_C_dict:
            w_C_dict[frozen_C] = calc_w_C(C, A_plus, A_minus, D_plus, D_minus, lambda_val)
S = list(w_C_dict.keys())

lps = LPS(S, w_C_dict, vertices)

print(w_C_dict)
print(S)

lps = LPS(S, w_C_dict, vertices)
# lps.print_lps()

lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_lps()
# lps.print_lps()

new_w_C_dict = {}
frozen_C = frozenset([0, 1, 2])
new_w_C_dict[frozen_C] = calc_w_C(
        list(frozen_C), A_plus, A_minus, D_plus, D_minus, lambda_val
    )
new_S = list(new_w_C_dict.keys())

print(new_w_C_dict)
print(new_S)

lps.update_lps(new_S, new_w_C_dict)
lps.print_lps()

lps_opt, lps_primal_sol, lps_dual_sol = lps.solve_lps()
lps.print_lps()