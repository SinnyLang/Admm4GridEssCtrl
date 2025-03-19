import numpy as np
from scipy.optimize import minimize
# code from deepseek

# 参数设置
num_nodes = 3  # 节点数量
num_lines = 3  # 线路数量
rho = 1.0  # ADMM惩罚参数
max_iter = 100  # 最大迭代次数
epsilon = 1e-4  # 收敛阈值

# 节点数据
P_gen = np.array([0.5, 0.3, 0.2])  # 发电量
P_load = np.array([0.4, 0.2, 0.1])  # 负荷
Q_gen = np.array([0.1, 0.05, 0.05])  # 无功发电量
Q_load = np.array([0.05, 0.02, 0.01])  # 无功负荷
V = np.ones(num_nodes)  # 电压幅值
theta = np.zeros(num_nodes)  # 相位角

# 线路数据
G = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # 电导
B = np.array([[0, -2, -2], [-2, 0, -2], [-2, -2, 0]])  # 电纳
S_max = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])  # 线路容量

# 初始化变量
P_ij = np.zeros((num_nodes, num_nodes))  # 有功功率流
Q_ij = np.zeros((num_nodes, num_nodes))  # 无功功率流
z_P = np.zeros((num_nodes, num_nodes))  # 全局有功功率流
z_Q = np.zeros((num_nodes, num_nodes))  # 全局无功功率流
u_P = np.zeros((num_nodes, num_nodes))  # 对偶变量（有功）
u_Q = np.zeros((num_nodes, num_nodes))  # 对偶变量（无功）

# 保存上一次的全局变量
z_P_prev = np.zeros((num_nodes, num_nodes))
z_Q_prev = np.zeros((num_nodes, num_nodes))

# 潮流方程
def power_flow(V, theta, G, B, i, j):
    P_ij = V[i] * V[j] * (G[i, j] * np.cos(theta[i] - theta[j]) + B[i, j] * np.sin(theta[i] - theta[j]))
    Q_ij = V[i] * V[j] * (G[i, j] * np.sin(theta[i] - theta[j]) - B[i, j] * np.cos(theta[i] - theta[j]))
    return P_ij, Q_ij

# 本地优化目标函数
def local_optimization(x, i, z_P, z_Q, u_P, u_Q, rho):
    P_gen_i, Q_gen_i, V_i, theta_i = x
    obj = 0.5 * P_gen_i**2  # 假设成本函数为二次函数
    for j in range(num_nodes):
        if j != i:
            P_ij, Q_ij = power_flow(V, theta, G, B, i, j)
            obj += u_P[i, j] * (P_ij - z_P[i, j]) + u_Q[i, j] * (Q_ij - z_Q[i, j])
            obj += (rho / 2) * ((P_ij - z_P[i, j])**2 + (Q_ij - z_Q[i, j])**2)
    return obj

# 本地优化约束
def constraints(x, i):
    P_gen_i, Q_gen_i, V_i, theta_i = x
    cons = []
    # 功率平衡约束
    P_balance = P_gen_i - P_load[i]
    Q_balance = Q_gen_i - Q_load[i]
    for j in range(num_nodes):
        if j != i:
            P_ij, Q_ij = power_flow(V, theta, G, B, i, j)
            P_balance -= P_ij
            Q_balance -= Q_ij
    cons.append({'type': 'eq', 'fun': lambda x: P_balance})
    cons.append({'type': 'eq', 'fun': lambda x: Q_balance})
    # 电压和相位角约束
    cons.append({'type': 'ineq', 'fun': lambda x: x[2] - 0.95})  # V_min = 0.95
    cons.append({'type': 'ineq', 'fun': lambda x: 1.05 - x[2]})  # V_max = 1.05
    cons.append({'type': 'ineq', 'fun': lambda x: x[3] + np.pi / 6})  # theta_min = -pi/6
    cons.append({'type': 'ineq', 'fun': lambda x: np.pi / 6 - x[3]})  # theta_max = pi/6
    return cons

# ADMM迭代
for k in range(max_iter):
    # 保存上一次的全局变量
    z_P_prev = z_P.copy()
    z_Q_prev = z_Q.copy()

    # 本地优化
    for i in range(num_nodes):
        x0 = np.array([P_gen[i], Q_gen[i], V[i], theta[i]])
        res = minimize(local_optimization, x0, args=(i, z_P, z_Q, u_P, u_Q, rho),
                       constraints=constraints(x0, i), method='SLSQP')
        P_gen[i], Q_gen[i], V[i], theta[i] = res.x

    # 全局协调
    for i in range(num_nodes):
        for j in range(num_nodes):
            if j != i:
                P_ij[i, j], Q_ij[i, j] = power_flow(V, theta, G, B, i, j)
                z_P[i, j] = np.clip(P_ij[i, j] + u_P[i, j] / rho, -S_max[i, j], S_max[i, j])
                z_Q[i, j] = np.clip(Q_ij[i, j] + u_Q[i, j] / rho, -S_max[i, j], S_max[i, j])

    # 对偶变量更新
    for i in range(num_nodes):
        for j in range(num_nodes):
            if j != i:
                u_P[i, j] += rho * (P_ij[i, j] - z_P[i, j])
                u_Q[i, j] += rho * (Q_ij[i, j] - z_Q[i, j])

    # 检查收敛条件
    primal_residual = np.linalg.norm(P_ij - z_P) + np.linalg.norm(Q_ij - z_Q)
    dual_residual = rho * (np.linalg.norm(z_P - z_P_prev) + np.linalg.norm(z_Q - z_Q_prev))
    if primal_residual < epsilon and dual_residual < epsilon:
        print(f"Converged at iteration {k}")
        break

# 输出结果
print("P_gen:", P_gen)
print("Q_gen:", Q_gen)
print("V:", V)
print("theta:", theta)
print("P_ij:", P_ij)
print("Q_ij:", Q_ij)