import numpy as np
import cvxpy as cp
# todo 后续需要注意    cvxpy 无法计算cos和sin，可采用线性近似和二阶锥近似
#  code from gpt

# 参数设定
num_nodes = 5  # 节点数
rho = 1.0  # ADMM 参数
max_iter = 20  # 迭代次数
S_max = np.ones((num_nodes, num_nodes)) * 5  # 线路功率上限
V_min, V_max = 0.95, 1.05  # 电压限制
theta_max = np.pi / 4  # 相角差限制

# 变量定义
P = cp.Variable(num_nodes)  # 节点有功功率
Q = cp.Variable(num_nodes)  # 节点无功功率
V = cp.Variable(num_nodes)  # 节点电压
theta = cp.Variable(num_nodes)  # 节点相角

Z_P = cp.Variable((num_nodes, num_nodes))  # 线路有功
Z_Q = cp.Variable((num_nodes, num_nodes))  # 线路无功

Y_P = np.zeros(num_nodes)  # 拉格朗日乘子（有功）
Y_Q = np.zeros(num_nodes)  # 拉格朗日乘子（无功）

# 线路参数（随机模拟）
G = np.random.rand(num_nodes, num_nodes)  # 电导矩阵
B = np.random.rand(num_nodes, num_nodes)  # 电纳矩阵

# 目标函数（交易成本 + 潮流损耗）
f_x = cp.sum_squares(P) + cp.sum_squares(Q)

for k in range(max_iter):
    # Step 1: 更新 P, Q, V, theta
    obj_x = cp.Minimize(f_x +
        (rho / 2) * cp.sum_squares(P - cp.sum(Z_P, axis=1) + Y_P / rho) +
        (rho / 2) * cp.sum_squares(Q - cp.sum(Z_Q, axis=1) + Y_Q / rho) +
        cp.sum_squares(V @ V.T * G * cp.cos(theta - theta.T) + V @ V.T * B * cp.sin(theta - theta.T) - Z_P) +
        cp.sum_squares(V @ V.T * G * cp.sin(theta - theta.T) - V @ V.T * B * cp.cos(theta - theta.T) - Z_Q)
    )
    constraints_x = [
        V_min <= V, V <= V_max,
        cp.abs(theta - theta.T) <= theta_max
    ]
    prob_x = cp.Problem(obj_x, constraints_x)
    prob_x.solve()

    # Step 2: 更新线路功率 (P_{ij}, Q_{ij})
    obj_z = cp.Minimize(
        (rho / 2) * cp.sum_squares(P.value - cp.sum(Z_P, axis=1) + Y_P / rho) +
        (rho / 2) * cp.sum_squares(Q.value - cp.sum(Z_Q, axis=1) + Y_Q / rho)
    )
    constraints_z = [
        cp.norm(cp.hstack([Z_P, Z_Q]), axis=0) <= S_max
    ]
    prob_z = cp.Problem(obj_z, constraints_z)
    prob_z.solve()

    # Step 3: 更新拉格朗日乘子
    Y_P += rho * (P.value - np.sum(Z_P.value, axis=1))
    Y_Q += rho * (Q.value - np.sum(Z_Q.value, axis=1))

print(f"Optimal P: {P.value}, Optimal Q: {Q.value}")
