from mip import Model, xsum, maximize, MAXIMIZE, CONTINUOUS, BINARY, OptimizationStatus

class AP_MILP:
    def __init__(self, vertices, A_plus, A_minus, D_plus, D_minus, lambda_val, lps_dual_sol):
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
        self.lps_dual_sol = lps_dual_sol

        # 変数
        self.x_u = {u: self.model.add_var(var_type=BINARY, name=f"x_{u}") for u in self.vertices}
        self.alpha_u = {u: self.model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"alpha_{u}") for u in self.vertices}
        self.s = self.model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"s")
        self.w_uv = {(u, v): self.model.add_var(var_type=CONTINUOUS, lb=0, ub=1, name=f"w_{u}_{v}")
                     for u in self.vertices for v in self.vertices if self.A_plus[u, v] > 0 or self.A_minus[u, v] > 0}
        
        # 制約
        for u in self.vertices:
            self.model.add_constr(self.s - (1 - self.x_u[u]) <= self.alpha_u[u])
            self.model.add_constr(self.alpha_u[u] <= self.s)
            self.model.add_constr(self.alpha_u[u] <= self.x_u[u])

        self.model.add_constr(xsum(self.alpha_u[u] for u in vertices) == 1)

        for u in self.vertices:
            for v in self.vertices: 
                if (self.A_plus[u, v] > 0 or self.A_minus[u, v] > 0):
                    self.model.add_constr(self.w_uv[(u, v)] <= self.alpha_u[u])
                    self.model.add_constr(self.w_uv[(u, v)] <= self.alpha_u[v])
                    self.model.add_constr(self.alpha_u[u] - (2 - self.x_u[u] - self.x_u[v]) <= self.w_uv[(u, v)])
                    self.model.add_constr(self.alpha_u[v] - (2 - self.x_u[u] - self.x_u[v]) <= self.w_uv[(u, v)])

        # 目的関数

        self.model.objective = maximize(
            4 * xsum(self.w_uv[(u, v)] for u in self.vertices for v in self.vertices if self.A_plus[u, v] > 0)
            -2 * (1 - self.lambda_val) * xsum(self.D_plus[u] * self.alpha_u[u] for u in self.vertices)
            -4 * xsum(self.w_uv[(u, v)] for u in self.vertices for v in self.vertices if self.A_minus[u, v] > 0)
            +2 * self.lambda_val * xsum(self.D_minus[u] * self.alpha_u[u] for u in self.vertices)
            - xsum(self.lps_dual_sol[u] * self.x_u[u] for u in vertices)
        )

    def solve_ap_milp(self):
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
            "alpha_u": {u: self.alpha_u[u].x for u in self.vertices},
            "s": self.s.x,
            "w_uv": {(u, v): self.w_uv[(u, v)].x for u, v in self.w_uv},
        }

        return self.ap_milp_opt, self.ap_milp_sol

    def print_ap_milp(self):
        print("\n=== AP-MILP ===")
        print("Objective Function:")
        print(self.model.objective)
        # print("\nConstraints:")
        # for c in self.model.constrs:
        #     print(c)
        # print("\nVariables:")
        # for v in self.model.vars:
        #     print(f"{v.name}: {v}")
        print("\nStatus:")
        print(self.model.status)
        if self.model.status == OptimizationStatus.OPTIMAL:
            print(f"Objective Value: {self.ap_milp_opt}")
            print("Solution (x_u):")
            for u, value in self.ap_milp_sol["x_u"].items():
                print(f"  x_{u}: {value}")
            # print("Solution (alpha_u):")
            # for u, value in self.ap_milp_sol["alpha_u"].items():
            #     print(f"  alpha_{u}: {value}")
            # print(f"s: {self.ap_milp_sol['s']}")
            # print("Solution (w_uv):")
            # for (u, v), value in self.ap_milp_sol["w_uv"].items():
            #     print(f"  w_{u}_{v}: {value}")