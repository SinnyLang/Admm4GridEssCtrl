import numpy as np
from scipy.optimize import minimize

import makeYbus
from case14 import case14  # 导入 IEEE 14 节点数据


# 加载 IEEE 14 节点数据
ppc = case14()

# 提取数据
baseMVA = ppc["baseMVA"]  # 系统基准功率
bus = ppc["bus"]  # 节点数据
gen = ppc["gen"]  # 发电机数据
branch = ppc["branch"]  # 线路数据

# 参数设置
num_nodes = bus.shape[0]  # 节点数量
num_lines = branch.shape[0]  # 线路数量
rho = 1.0  # ADMM惩罚参数
max_iter = 100  # 最大迭代次数
epsilon = 1e-4  # 收敛阈值

# 初始化变量
P_gen = gen[:, 1] / baseMVA  # 发电机有功出力（标幺值）
Q_gen = gen[:, 2] / baseMVA  # 发电机无功出力（标幺值）
V = bus[:, 7]  # 节点电压幅值（标幺值）
theta = np.radians(bus[:, 8])  # 节点电压相角（弧度）

# todo make type of Ybus from csc_matrix to ndarray
Ybus, _, _ = makeYbus.makeYbus(baseMVA, bus, branch)

# 线路参数
G = np.zeros((num_nodes, num_nodes))  # 电导矩阵
B = np.zeros((num_nodes, num_nodes))  # 电纳矩阵
for line in branch:
    fbus = int(line[0]) - 1  # 起始节点（Python索引从0开始）
    tbus = int(line[1]) - 1  # 终止节点
    r = line[2]  # 电阻
    x = line[3]  # 电抗
    G[fbus, tbus] = r / (r**2 + x**2)
    B[fbus, tbus] = -x / (r**2 + x**2)
    G[tbus, fbus] = G[fbus, tbus]
    B[tbus, fbus] = B[fbus, tbus]

# 初始化 ADMM 变量
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
    P_balance = P_gen_i - bus[i, 2] / baseMVA  # 负荷有功
    Q_balance = Q_gen_i - bus[i, 3] / baseMVA  # 负荷无功
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
                z_P[i, j] = np.clip(P_ij[i, j] + u_P[i, j] / rho, -branch[i, 5] / baseMVA, branch[i, 5] / baseMVA)
                z_Q[i, j] = np.clip(Q_ij[i, j] + u_Q[i, j] / rho, -branch[i, 5] / baseMVA, branch[i, 5] / baseMVA)

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
print("发电机出力：")
for i in range(num_nodes):
    print(f"节点 {i + 1}: P_gen = {P_gen[i] * baseMVA:.2f} MW, Q_gen = {Q_gen[i] * baseMVA:.2f} MVAR")

print("\n节点电压：")
for i in range(num_nodes):
    print(f"节点 {i + 1}: Vm = {V[i]:.4f} pu, Va = {np.degrees(theta[i]):.4f} deg")

print("\n线路潮流：")
for i in range(num_lines):
    fbus = int(branch[i, 0]) - 1
    tbus = int(branch[i, 1]) - 1
    P_from = P_ij[fbus, tbus] * baseMVA
    Q_from = Q_ij[fbus, tbus] * baseMVA
    P_to = P_ij[tbus, fbus] * baseMVA
    Q_to = Q_ij[tbus, fbus] * baseMVA
    print(f"线路 {fbus + 1} -> {tbus + 1}: P_from = {P_from:.2f} MW, Q_from = {Q_from:.2f} MVAR, "
          f"P_to = {P_to:.2f} MW, Q_to = {Q_to:.2f} MVAR")