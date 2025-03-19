import numpy as np
from scipy.optimize import minimize
from case14 import case14  # Importing the IEEE 14-bus case data

# === Load IEEE 14-bus data ===
ppc = case14()  # Get IEEE 14 bus data

# Extract buses, generators, and branches from the case data
buses = ppc["bus"]
branches = ppc["branch"]
generators = ppc["gen"]

# === Set ADMM Parameters ===
num_nodes = len(buses)  # Number of buses
rho = 1.0  # ADMM penalty parameter
max_iter = 20  # Max number of ADMM iterations
V_min, V_max = 0.94, 1.06  # Voltage limits
theta_max = np.pi / 4  # Max phase angle difference

# === Initialize Variables ===
P = np.zeros(num_nodes)  # Active power at each bus
Q = np.zeros(num_nodes)  # Reactive power at each bus
V = np.ones(num_nodes)  # Voltage magnitude at each bus
theta = np.zeros(num_nodes)  # Voltage phase angle at each bus
P_ij = np.zeros((num_nodes, num_nodes))  # Active power flow between buses
Q_ij = np.zeros((num_nodes, num_nodes))  # Reactive power flow between buses


# === Objective Function for ADMM ===
def objective(x):
    P, Q, V, theta = x[:num_nodes], x[num_nodes:2 * num_nodes], x[2 * num_nodes:3 * num_nodes], x[3 * num_nodes:]
    cost = np.sum(P ** 2 + Q ** 2)  # Objective: minimize power loss
    penalty = rho / 2 * np.sum((P - np.sum(P_ij, axis=1)) ** 2)
    penalty += rho / 2 * np.sum((Q - np.sum(Q_ij, axis=1)) ** 2)
    return cost + penalty


# === Constraints for ADMM ===
def constraints(x):
    P, Q, V, theta = x[:num_nodes], x[num_nodes:2 * num_nodes], x[2 * num_nodes:3 * num_nodes], x[3 * num_nodes:]
    cons = []

    # Power balance constraints for each bus (active and reactive power)
    for i in range(num_nodes):
        cons.append({'type': 'eq', 'fun': lambda x, i=i: P[i] - np.sum(P_ij[i, :])})
        cons.append({'type': 'eq', 'fun': lambda x, i=i: Q[i] - np.sum(Q_ij[i, :])})

    # Line power flow limits: ensure flows do not exceed capacity
    # todo define constant S_max for every line
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cons.append(
                    {'type': 'ineq', 'fun': lambda x, i=i, j=j: S_max[i, j] ** 2 - (P_ij[i, j] ** 2 + Q_ij[i, j] ** 2)})

    # Voltage limits for each bus
    for i in range(num_nodes):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: V[i] - V_min})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: V_max - V[i]})

    # Voltage phase angle limits between buses
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: theta_max - abs(theta[i] - theta[j])})

    return cons


# === ADMM Iterations ===
for k in range(max_iter):
    print(f"ADMM Iteration {k + 1}")

    # Step 1: Optimize node powers (P, Q, V, theta)
    x0 = np.hstack([P, Q, V, theta])  # Initial guess
    res = minimize(objective, x0, constraints=constraints(x0), method='SLSQP', options={'disp': False})
    P, Q, V, theta = res.x[:num_nodes], res.x[num_nodes:2 * num_nodes], res.x[2 * num_nodes:3 * num_nodes], res.x[
                                                                                                            3 * num_nodes:]

    # Step 2: Update power flow between buses (P_ij, Q_ij)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                P_ij[i, j] = (P[i] + P[j]) / 2  # Approximate power flow
                Q_ij[i, j] = (Q[i] + Q[j]) / 2

    # Step 3: Update Lagrange multipliers
    Y_P += rho * (P - np.sum(P_ij, axis=1))
    Y_Q += rho * (Q - np.sum(Q_ij, axis=1))

    print(f"  P: {P}, Q: {Q}, V: {V}, Theta: {theta}")

# === Display Final Results ===
print("Optimization completed.")
print(f"Optimal Active Power P: {P}")
print(f"Optimal Reactive Power Q: {Q}")
