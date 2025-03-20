from mip import Model, xsum, maximize, CONTINUOUS, BINARY, OptimizationStatus

class AP_MILPWithPartition:
    def __init__(self, vertices, A_plus, A_minus, D_plus, D_minus, lambda_val):
        """
        AP-MILPの初期化
        """
        self.model = Model(solver_name="CBC")
        self.model.solver.set_verbose(False)
        self.vertices = vertices
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.D_plus = D_plus
        self.D_minus = D_minus
        self.lambda_val = lambda_val

        self.E_plus = []
        self.E_minus = []
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if A_plus[i, j] == 1:
                    self.E_plus.append((i, j))
                elif A_minus[i, j] == 1:
                    self.E_minus.append((i, j))
        self.E = self.E_plus + self.E_minus

        # 変数
        self.x_u = {u: self.model.add_var(var_type=BINARY, name=f"x_{u}") for u in self.vertices}
        self.z_uv = {(u, v): self.model.add_var(var_type=BINARY, name=f"z_{u}_{v}") for (u, v) in  self.E}
        self.k = 1
        self.inv_k = 1 / self.k

        # 制約
        for (u, v) in self.E:
            self.model.add_constr(self.x_u[u] + self.x_u[v] <= 1 + self.z_uv[(u, v)])
            self.model.add_constr(self.x_u[u] >= self.z_uv[(u, v)])
            self.model.add_constr(self.x_u[v] >= self.z_uv[(u, v)])
        self.partition_constr = self.model.add_constr(
            xsum(self.x_u[u] for u in self.vertices) == self.k,
            name="partition_constr"
        )

        # ベース項（双対変数なし）
        self.base_term = (
            4 * self.inv_k * xsum(self.z_uv[(u, v)] for (u, v) in  self.E_plus)
            -2 * self.inv_k * (1 - self.lambda_val) * xsum(self.D_plus[u] * self.x_u[u] for u in self.vertices)
            -4 * self.inv_k * xsum(self.z_uv[(u, v)] for u in self.vertices for (u, v) in self.E_minus)
            +2 * self.inv_k * self.lambda_val * xsum(self.D_minus[u] * self.x_u[u] for u in self.vertices)
        )

        self.dual_term = 0
        self.lps_dual_sol = None

    def add_lps_dual_sol(self, lps_dual_sol):
        """
        双対変数を目的関数に追加
        """
        self.lps_dual_sol = lps_dual_sol
        # 双対項
        self.dual_term = - self.inv_k * xsum(lps_dual_sol[v] * self.z_uv[(u, v)] for (u, v) in self.E)

        # 目的関数を設定
        self.model.objective = maximize(self.base_term + self.dual_term)

    def solve_model(self):
        """
        AP-MILPを解く
        Returns:
        - ap_milp_opt: 最適値
        - ap_milp_sol: 最適解
        """
        self.model.optimize()

        self.ap_milp_opt = self.model.objective_value
        self.ap_milp_sol = {
            "x_u": {u: self.x_u[u].x for u in self.vertices},
            "z_uv": {(u, v): self.z_uv[(u, v)].x for (u, v) in self.E},
        }

        return self.ap_milp_opt, self.ap_milp_sol

    def debag_print_model(self):
        print("\n=== AP-MILP ===")

        print("Objective Function:")
        print(self.model.objective)

        print("\nConstraint:")
        for constr in self.model.constrs:
            print(constr)

        print("\nStatus:")
        print(self.model.status)

        if self.model.status == OptimizationStatus.OPTIMAL:
            print(f"Objective Value: {self.ap_milp_opt}")

            print("Solution (x_u):")
            for u, value in self.ap_milp_sol["x_u"].items():
                print(f"  x_{u}: {value}")

    def update_model_for_k(self, k):
        """
        モデルの更新
        Parameters:
        - k: 制約として設定する集合の要素数, 0以上かつ頂点数未満である必要がある
        """
        self.k = k
        self.inv_k = 1 / self.k

        self.update_partition_constr()
        self.update_base_term()
        if self.lps_dual_sol is not None:
            self.add_lps_dual_sol(self.lps_dual_sol)

    def update_partition_constr(self):
        """
        x_u 列生成の際に生成される集合の要素数を固定する制約を追加または更新
        """
        # 既存の制約を削除
        if hasattr(self, "partition_constr"):
            try:
                self.model.remove(self.partition_constr)
            except Exception as e:
                print(f"Warning: Failed to remove partition constraint. Error: {e}")

        # 新しい制約を追加
        self.partition_constr = self.model.add_constr(
            xsum(self.x_u[u] for u in self.vertices) == self.k,
            name="partition_constr"
        )

    def update_base_term(self):
        """
        k の変更に応じてベース項（目的関数の一部）を更新
        """
        self.base_term = (
            4 * self.inv_k * xsum(self.z_uv[(u, v)] for (u, v) in self.E_plus)
            -2 * self.inv_k * (1 - self.lambda_val) * xsum(self.D_plus[u] * self.x_u[u] for u in self.vertices)
            -4 * self.inv_k * xsum(self.z_uv[(u, v)] for u in self.vertices for (u, v) in self.E_minus)
            +2 * self.inv_k * self.lambda_val * xsum(self.D_minus[u] * self.x_u[u] for u in self.vertices)
        )