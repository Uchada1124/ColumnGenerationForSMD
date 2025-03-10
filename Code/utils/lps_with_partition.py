from utils.lps import LPS
from mip import xsum

class LPSWithPartition(LPS):
    def __init__(self, S, w_C_dict, vertices, k):
        """
        LPSWithPartitionの初期化

        Parameters:
        - S: 初期の列集合 (frozenset のリスト)
        - w_C_dict: 各列の重みを格納する辞書 {frozenset(C): w_C}
        - vertices: 頂点のリスト
        - k: 初期の分割数制約
        """
        super().__init__(S, w_C_dict, vertices)
        self.S = S
        self.k = k
        self.partition_constr = None
        self.add_partition_constr(k)

    def add_partition_constr(self, k):
        """
        分割数制約 (|P| <= k) を追加または更新
        """
        self.k = k

        # 既存の制約がある場合は削除
        if self.partition_constr is not None:
            try:
                self.model.remove(self.partition_constr)
                self.partition_constr = None  # 削除後はNoneにする
                # Sの更新が必要

            except Exception as e:
                print(f"Warning: Failed to remove partition constraint. Error: {e}")

        # 新しい制約を追加
        self.partition_constr = self.model.add_constr(
            xsum(self.z_C[C] for C in self.S) <= self.k,
            name="partition_constr"
        )
    
    def update_columns_and_partition(self, new_S, new_w_C_dict, new_k):
        """
        列を追加しつつ、分割数制約も更新するメソッド
        Parameters:
        - new_S: 新しい列集合 (frozenset のリスト)
        - new_w_C_dict: 新しい列に対応する重み辞書
        - new_k: 新しい分割数制約値
        """
        super().update_model(new_S, new_w_C_dict)  # 親クラスの列生成処理を実行
        self.update_partition_constr(new_k)  # 分割数制約の更新
