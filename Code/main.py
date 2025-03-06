from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph
from utils.plot import plot_graph, plot_lps_objective, plot_partitioned_graph
from utils.partition import generate_singleton
from utils.column_generation import column_generation

def main():
    # 隣接行列のデータを読込む
    # Adj = read_csv_as_numpy("./data/test_data/01_Slovene_AdjMat.csv")
    Adj = read_csv_as_numpy("./data/test_data/02_GahukuGama_AdjMat.csv")

    # 隣接行列をもとにグラフを生成
    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)

    # # グラフの可視化
    plot_graph(G, title="Signed Graph")

    # 列生成法
    lambda_val = 0.5
    init_partitions = [generate_singleton(vertices)]
    cg_opt, cg_sol, lps_opt_list, S, cnt = column_generation(
        vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, init_partitions
    )

    # 結果の出力
    print("\n=== 列生成法の結果 ===")
    print(f"反復回数: {cnt}")
    print(f"最終目的関数値: {cg_opt:.4f}")
    print("最終解 (z_C):")
    for C, value in cg_sol.items():
        print(f"  {set(C)}: {value:.4f}")

    # LPSの最適値の推移をプロット
    plot_lps_objective(lps_opt_list)

    # 最終的な分割の可視化
    plot_partitioned_graph(G, cg_sol, title="Partitioned Signed Network")

if __name__ == '__main__':
    main()
