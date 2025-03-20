from utils.input_data import read_csv_as_numpy
from utils.graph import generate_signed_graph
from utils.plot import plot_graph, plot_lps_objective, plot_partitioned_graph
from utils.partition import generate_singleton
from utils.column_generation_with_partition import column_generation_with_partition

def main():
    # データセットの選択
    dataset = "01_Slovene_AdjMat.csv"
    # dataset = "02_GahukuGama_AdjMat.csv"
    print(f"\n=== データセット: {dataset} ===")

    # 隣接行列のデータを読込む
    Adj = read_csv_as_numpy(f"./data/test_data/{dataset}")

    # 隣接行列をもとにグラフを生成
    G, vertices, A_plus, A_minus, D_plus, D_minus = generate_signed_graph(A=Adj)

    # # グラフの可視化
    # plot_graph(G, title=f"Signed Graph - {dataset}")

    # パラメータ設定
    lambda_val = 0.5

    # 列生成法（分割数制約付き）の実行
    results = column_generation_with_partition(
        vertices, A_plus, A_minus, D_plus, D_minus, 
        lambda_val, init_partitions=[generate_singleton(vertices)]
    )

    # # 結果の出力と可視化
    # for k in k_values:
    #     result = results[k]
        
    #     print(f"\n=== k = {k} の結果 ===")
    #     print(f"反復回数: {result['cnt']}")
    #     print(f"最終目的関数値: {result['opt']:.4f}")
    #     print("最終解 (z_C):")
    #     for C, value in result['sol'].items():
    #         if value > 1e-6:  # 数値誤差を考慮
    #             print(f"  {set(C)}: {value:.4f}")

    #     # LPSの最適値の推移をプロット
    #     plot_lps_objective(
    #         result['lps_opt_list'], 
    #         title=f"Objective Value Progress (k={k})"
    #     )

    #     # 最終的な分割の可視化
    #     plot_partitioned_graph(
    #         G, result['sol'],
    #         title=f"Partitioned Signed Network (k={k})"
    #     )

if __name__ == '__main__':
    main()
