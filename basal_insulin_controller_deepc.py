from cvxpy.reductions.solvers.defines import QP_SOLVERS
import cvxpy as cp
import numpy as np
from random import randint

from Heq import K, n_past, T

t = 1
xkm1= [0,0,0,0,0,0,0,0,250]
previous_ba = 0

x_obs = 2


H120 = np.load("DeePC_H120matrix.npy")
M_pf120 = np.load("DeePC_Hm120matrix.npy")
ncol = M_pf120.shape[1]
I_p120 = np.load("DeePC_Hu120matrix.npy")[:n_past]
I_f120 = np.load("DeePC_Hu120matrix.npy")[n_past:]
X_p120 = np.load("DeePC_Hd120matrix.npy")[:n_past*x_obs]
X_f120 = np.load("DeePC_Hd120matrix.npy")[n_past*x_obs:]


meal = 0
meals = []
inputs = []
states = []
i_past = [0]*n_past
x_past_np = np.zeros((n_past, x_obs))


def update_hist(new_x, new_u):
    global i_past, x_past_np
    # update past values of x and of i

    i_past[:n_past-1] = i_past[1:]
    i_past[n_past-1] = new_u
    
    x_past_np[:n_past-1] = x_past_np[1:]
    x_past_np[n_past-1] = new_x

def DeePC(model, x0, tau, time_index, inputs_, xss, uss, dss):
    global t,T, K, xkm1, previous_ba, meal, meals, inputs, states, n_past, i_past, x_past_np, x_obs, bo
    global H120, M_pf120, I_p120, I_f120, X_p120, X_f120
    
    # Return previous input if no input needs to be recalculated
    if (t == T):
        t=1
    else:
        t+=1
        return i_past[-1], dss

    # if we dont have enough past data, return
    if (time_index < n_past*T +5):
        update_hist(np.array([x0[5],x0[8]]), uss+randint(0, 150)/100000)
        return i_past[-1], dss
    
    # Get future and past meals
    m_all = model.get_discretized_meals(time_index-n_past*T, (K+n_past)*T, T)

    M_pf = M_pf120
    I_p = I_p120
    I_f = I_f120
    X_p = X_p120
    X_f = X_f120

    # define cvx variables
    g = cp.Variable((ncol,1))
    s = cp.Variable((n_past*x_obs,1))

    x = cp.Variable((x_obs*K, 1))

    n_i = 1
    i = cp.Variable((n_i*K, 1))

    # define constraints
    constr = [
        #meals
        M_pf@g == np.array(m_all).reshape(n_past+K,1),
        #inputs
        I_p@g == np.array([i_past]).reshape(n_past,1),
        I_f@g == i,
        #states
        X_p@g == x_past_np.reshape(n_past*x_obs,1) + s,
        x[0] == x0[5],
        x[1] == x0[8],
        X_f@g == x,
    ]
    cost = 0

    # cost and more constraints
    u_lim = 0.04

    q = 0.001
    xss = 120
    q = .15
    r = 18000

    for k in range(K):
        cost += q*cp.quad_form(x[k*x_obs:(k+1)*x_obs]-np.array([0]*(x_obs-1) + [xss]).reshape(x_obs,1), np.diag([0]*(x_obs-1) + [q])) + r*cp.sum_squares(i[k,0]-uss)
        constr += [i[k,0] >= 0.0, i[k,0] <= u_lim, x[(k+1)*x_obs-1,0] <= 175 + 5*(x0[8]>160), x[(k+1)*x_obs-1,0] >= 70]
    cost += .1*cp.norm(g,1) + 10_000*cp.norm(s,1)
    
    
    # Solve problem
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.MOSEK , verbose = False) 

    update_hist(np.array([x0[5],x0[8]]), i[0,0].value)
    return i_past[-1], dss 