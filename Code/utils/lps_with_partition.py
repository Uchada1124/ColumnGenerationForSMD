from utils.lps import LPS
from mip import xsum

class LPSWithPartition(LPS):
    def add_partition_constr(self, k):
        """
        分割数制約 (|P| = k) を追加または更新
        Σ z_C <= k を追加
        
        Parameters:
        - k: 分割数の制約値
        """
        self.k = k
        
        # 既存の制約があれば削除
        if hasattr(self, 'partition_constr'):
            self.model.remove(self.partition_constr)
            
        # 新しい制約を追加
        self.partition_constr = self.model.add_constr(
            xsum(self.z_C[C] for C in self.S) <= self.k,
            name="partition_constr"
        )

    def update_model(self, new_S, new_w_C_dict):
        """
        新しい列を追加してモデルを更新
        
        Parameters:
        - new_S: 新しい列集合 (frozenset のリスト)
        - new_w_C_dict: 新しい列に対応する重み辞書
        """
        # 現在のkを保存
        current_k = self.k
        
        # 一時的に分割数制約を削除
        self.model.remove(self.partition_constr)
        
        # 親クラスのupdate_modelを呼び出し
        super().update_model(new_S, new_w_C_dict)
        
        # 分割数制約を再追加
        self.add_partition_constr(current_k)
