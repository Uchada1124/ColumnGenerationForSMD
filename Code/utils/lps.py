from mip import Model, xsum, maximize, CONTINUOUS, Column, OptimizationStatus

from utils.wc import calc_w_C

class LPS:
    def __init__(self, vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, init_partitions):
        """
        LP(S)の初期化

        Parameters:
        - S: 現在の列集合 (frozenset のリスト)
        - w_C_dict: 各列の重みを格納する辞書 {frozenset(C): w_C}
        - vertices: 頂点のリスト
        """
        self.vertices = vertices
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.D_plus = D_plus
        self.D_minus = D_minus
        self.lambda_val = lambda_val
        self.init_partitions = init_partitions

        self.model = Model(solver_name="CBC")
        self.model.solver.set_verbose(False)
        self.vertices = vertices
        self.w_C_dict = {}
        self.S = []
        self.init_S_w_C_dict()

        # 変数
        self.z_C = {C: self.model.add_var(var_type=CONTINUOUS, lb=0, name=f"z_{C}") for C in self.S}

        # 制約
        self.constraints = {}
        for u in vertices:
            self.constraints[u] = self.model.add_constr(
                xsum(self.z_C[C] for C in self.S if u in C) == 1
            )

        # 目的関数
        self.model.objective = maximize(xsum(self.w_C_dict[C] * self.z_C[C] for C in self.S))

    def init_S_w_C_dict(self):
        """
        self.S と self.w_C_dict を初期化
        """
        for partition in self.init_partitions:
            for C in partition:
                frozen_C = frozenset(C)
                if frozen_C not in self.w_C_dict:
                    self.w_C_dict[frozen_C] = calc_w_C(
                        C, self.A_plus, self.A_minus, self.D_plus, self.D_minus, self.lambda_val
                        )
        self.S = list(self.w_C_dict.keys())

    def solve_model(self):
        """
        LP(S) を解く
        Returns:
        - lps_opt: 最適値
        - lps_primal_sol: 主問題の解 {frozenset(C): 値}
        - lps_dual_sol: 双対問題の解 {頂点 u: 値}
        """
        self.model.optimize()

        # 最適値
        self.lps_opt = self.model.objective_value

        # 主問題の解
        self.lps_primal_sol = {C: self.z_C[C].x for C in self.S}

        # 双対問題の解
        self.lps_dual_sol = {u: self.constraints[u].pi for u in self.vertices}

        return self.lps_opt, self.lps_primal_sol, self.lps_dual_sol

    def update_model(self, frozen_C):
        """
        新しい列を追加してモデルを更新
        Parameters:
        - new_S: 新しい列集合 (frozenset のリスト)
        - new_w_C_dict: 新しい列に対応する重み辞書
        """
        new_w_C_dict = {}
        if frozen_C not in self.w_C_dict:
            new_w_C_dict[frozen_C] = calc_w_C(
                list(frozen_C), self.A_plus, self.A_minus, self.D_plus, self.D_minus, self.lambda_val
            )
        new_S = list(new_w_C_dict.keys())

        # S, w_C_dictの更新
        self.w_C_dict.update(new_w_C_dict)
        # self.S += new_S
        self.S = list(self.w_C_dict.keys())

        for C in new_S:
            # 列を構築
            column = Column(
                [self.constraints[u] for u in C],
                [1 for u in C]
            )

            # 新しい変数をモデルに追加
            z_new = self.model.add_var(
                var_type=CONTINUOUS,
                lb=0,
                obj=new_w_C_dict[C],
                column=column,
                name=f"z_{C}"
            )
            self.z_C[C] = z_new

        return self.S
    
    def debag_print_model(self):    
        print("\n=== LPS ===")

        print("S")
        print(self.S) 
        
        print("\nw_C_dict")
        print(self.w_C_dict)

        print("\nObjective Function:")
        print(self.model.objective)

        print("\nConstraints:")
        for c in self.model.constrs:
            print(c)

        print("\nVariables:")
        for v in self.model.vars:
            print(f"{v.name}: {v}")

        print("\nStatus:")
        print(self.model.status)

        if self.model.status == OptimizationStatus.OPTIMAL:
            print(f"Objective Value: {self.lps_opt}")

            print("Primal Solution (z_C):")
            for C, value in self.lps_primal_sol.items():
                print(f"  z_{C}: {value}")

            print("Dual Variables (y_u):")
            for u, value in self.lps_dual_sol.items():
                print(f"  y_{u}: {value}")