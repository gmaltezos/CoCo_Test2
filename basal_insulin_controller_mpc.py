from cvxpy.reductions.solvers.defines import QP_SOLVERS
import cvxpy as cp
import numpy as np

T = 10
t = 1
xkm1= [0,0,0,0,0,0,0,0,250]
previous_ba = 0

def MPC(model, x0, tau, time_index, inputs_, xss, uss, dss):
    global t,T, xkm1, previous_ba
    
    # Return previous input if no input needs to be recalculated
    if (t == T):
        t=1
    else:
        t+=1
        return previous_ba, dss
    
    # Advance the system to the point of the input delay
    x0 = np.array(x0)
    for i in range(tau):
        A_, B_, C_ = model.get_state_equations(1, xss)
        meals_ = model.get_discretized_meals(time_index, tau, 1)
        I_ = B_[:,0]
        M_ = B_[:,1]
        x0 = A_ @ x0 + I_ * (inputs_[time_index+i]) + M_ * meals_[i] + C_

    # Get state equations
    A, B, C = model.get_state_equations(T, xss)
    I = B[:,0]
    M = B[:,1]

    # Define variables of the optimization problem
    K = 25
    n_x = 9
    x = cp.Variable((n_x, K + 1))
    n_i = 1
    i = cp.Variable((n_i,K))

    # Get future meals
    meals = model.get_discretized_meals(time_index+tau, (K)*T, T)

    # Define cost
    cost = 0
    Q = np.diag([0, 0.00001, 0, 0, 0, 0.000001, 0.000001, 0.000001, 0.000003])
    r = 20

    # Add constraints
    constr_hard = []
    u_lim = 0.04
    for k in range(K):
        cost +=  cp.quad_form(x[:, k] - xss,Q) + r*cp.sum_squares(i[0,k])
        constr_hard += [x[:, k + 1] == A @ x[:, k] + I * (i[0,k]+uss) + M * meals[k]  + C, i[0,k]+uss >= 0.0, i[0,k]+uss <= u_lim, x[8, k] >= 70]

    # Add initial and terminal state constraint
    constr_hard = constr_hard + [x[:, 0] == x0]#, cp.abs(x[8, K] - 140) <= 80]#, cp.abs(x[8, K]-120) <= 150, cp.abs(x[0, K]-120) <= 150]
    
    # Solve optimization problem
    problem = cp.Problem(cp.Minimize(cost), constr_hard)
    problem.solve(solver=cp.SCS, verbose = False)

    # Update return variables
    r_i = i
    r_x = x
    previous_ba = max(r_i[0,0].value + uss, 0)

    return previous_ba, dss 