import numpy as np
from scipy.optimize import minimize
# code from gpt

# === 参数设置 ===
num_nodes = 5  # 节点数
num_lines = num_nodes * (num_nodes - 1) // 2  # 假设全连接网络
rho = 1.0  # ADMM 参数
max_iter = 20  # 迭代次数
S_max = np.ones((num_nodes, num_nodes)) * 5  # 线路功率上限
V_min, V_max = 0.95, 1.05  # 电压限制
theta_max = np.pi / 4  # 相角差限制

# === 变量初始化 ===
P = np.zeros(num_nodes)  # 节点有功功率
Q = np.zeros(num_nodes)  # 节点无功功率
V = np.ones(num_nodes)  # 假设初始电压为 1.0
theta = np.zeros(num_nodes)  # 初始相角
P_ij = np.zeros((num_nodes, num_nodes))  # 线路有功功率
Q_ij = np.zeros((num_nodes, num_nodes))  # 线路无功功率

Y_P = np.zeros(num_nodes)  # 拉格朗日乘子（有功）
Y_Q = np.zeros(num_nodes)  # 拉格朗日乘子（无功）

# 线路参数（模拟）
G = np.random.rand(num_nodes, num_nodes)  # 电导矩阵
B = np.random.rand(num_nodes, num_nodes)  # 电纳矩阵


# === 目标函数 ===
def objective(x):
    P, Q, V, theta = x[:num_nodes], x[num_nodes:2 * num_nodes], x[2 * num_nodes:3 * num_nodes], x[3 * num_nodes:]
    cost = np.sum(P ** 2 + Q ** 2)  # 目标：最小化损耗
    penalty = rho / 2 * np.sum((P - np.sum(P_ij, axis=1) + Y_P / rho) ** 2)
    penalty += rho / 2 * np.sum((Q - np.sum(Q_ij, axis=1) + Y_Q / rho) ** 2)
    return cost + penalty


# === 约束条件 ===
def constraints(x):
    P, Q, V, theta = x[:num_nodes], x[num_nodes:2 * num_nodes], x[2 * num_nodes:3 * num_nodes], x[3 * num_nodes:]
    cons = []

    # 节点功率平衡约束
    for i in range(num_nodes):
        cons.append({'type': 'eq', 'fun': lambda x, i=i: P[i] - np.sum(P_ij[i, :])})
        cons.append({'type': 'eq', 'fun': lambda x, i=i: Q[i] - np.sum(Q_ij[i, :])})

    # 线路功率限制
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cons.append(
                    {'type': 'ineq', 'fun': lambda x, i=i, j=j: S_max[i, j] ** 2 - (P_ij[i, j] ** 2 + Q_ij[i, j] ** 2)})

    # 电压约束
    for i in range(num_nodes):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: V[i] - V_min})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: V_max - V[i]})

    # 相角约束
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: theta_max - abs(theta[i] - theta[j])})

    return cons


# === ADMM 迭代 ===
for k in range(max_iter):
    print(f"ADMM Iteration {k + 1}")

    # Step 1: 更新 P, Q, V, theta
    x0 = np.hstack([P, Q, V, theta])  # 初始值
    res = minimize(objective, x0, constraints=constraints(x0), method='SLSQP', options={'disp': False})
    P, Q, V, theta = \
        (res.x[:num_nodes], res.x[num_nodes:2 * num_nodes], res.x[2 * num_nodes:3 * num_nodes], res.x[3 * num_nodes:])

    # Step 2: 更新 P_ij, Q_ij
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                P_ij[i, j] = (P[i] + P[j]) / 2  # 近似分配
                Q_ij[i, j] = (Q[i] + Q[j]) / 2

    # Step 3: 更新拉格朗日乘子
    Y_P += rho * (P - np.sum(P_ij, axis=1))
    Y_Q += rho * (Q - np.sum(Q_ij, axis=1))

    print(f"  P: {P}, Q: {Q}, V: {V}, Theta: {theta}")

# === 最终结果 ===
print("优化完成")
print(f"最终有功功率 P: {P}")
print(f"最终无功功率 Q: {Q}")
