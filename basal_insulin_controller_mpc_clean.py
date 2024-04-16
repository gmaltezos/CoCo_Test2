from cvxpy.reductions.solvers.defines import QP_SOLVERS
import cvxpy as cp
import numpy as np

T = 10
t = 1
xkm1= [0,0,0,0,0,0,0,0,250]
previous_ba = 0

def basal_insulin_controller_mpc(model, x0, tau, time_index, inputs_, meals_, xss, uss, to, dss):
# def basal_insulin_handler_clean(A, B, C, x0, tau, me_index, meals, xss, uss, to, dss):
    # return max(0,uss+(0.0005)*(x0[8]-xss[8])), dss
    global t,T, xkm1, previous_ba
    if x0[0]>180:
        return 0.025, dss
    if x0[0]<70:
        return 0, dss
    
    if (t == T):
        t=1
    else:
        t+=1
        return previous_ba, dss
    

    x0 = np.array(x0)
    for i in range(tau):
        A_, B_, C_, M_ = model.get_state_equations(1, x0)
        x0 = A_ @ x0 + B_ * (inputs_[time_index+i]) + M_ * meals_[time_index+i] + C_
        # x0 = model.step(time_index, meals_[time_index+i], inputs_[time_index+i])



    b = dss.basal_handler_params['default_rate']

    A, B, C, M = model.get_state_equations(T, x0)
    # print(A)
    # print(B)
    # print(C)
    # print("###############################")
    K = 25

    meal_seen = False
    meals = np.zeros((K+1, ))
    for k in range(K*T):
        meals[k//T] += meals_[time_index+tau+k]
        if meals_[time_index+k]>0:
            meal_seen = True

    # if meal_seen == True:
    #     print("meals:")
    #     print(meals_[time_index-1:time_index+K*T-1])
        # print(meals)



    n_x = 9
    x = cp.Variable((n_x, K + 1))
    x_ = cp.Variable((n_x, K + 1))
    n_u = 1
    u = cp.Variable((n_u,K))
    u_ = cp.Variable((n_u,K))

    cost = 0
    cost_ = 0
    constr = []
    constr_hard = []

    u_lim = 0.03
    # Q = np.diag([0, 10000, 0, 0, 0, 0.0001, 0.0001, 0.0001, 1*(x0[0]<0)])
    # Q = np.diag([0]*8+[0.00001])
    Q = np.diag([0, 0.00001, 0, 0, 0, 0.000001, 0.000001, 0.000001]+[0.000001])
    Q_ = np.diag([0, 0.00001, 0, 0, 0, 0.000001, 0.000001, 0.000001]+[0.000001])
    r = 1
    if meal_seen:
        xss[8] = 110
        # Q /= 1/2
        # Q_/= 1/2
        Q[8,8] = .000003
        Q_[8,8]= .000003
        r = 5
    if (x0[8]>xkm1[8]+5):
        Q[8,8] = .000002
        Q_[8,8]= .000002
    xkm1 = x0

    for k in range(K//2):
        # assert(meals[k]==0)
        # print(M*meals[k])
        cost +=  cp.quad_form(x[:, k] - xss,Q) + r*cp.sum_squares(u[0,k])
        cost_ +=  cp.quad_form(x_[:, k] - xss,Q_) + r*cp.sum_squares(u_[0,k])
        # cost +=  10*cp.sum_squares(x[:, k]-xss[8]) + r*cp.sum_squares(u[0,k])
        # meal_impact[:] = [0, [time_index+k]]
        constr += [x_[:, k + 1] == A @ x_[:, k] + B * (u_[0,k]+uss) * to + M * meals[k] + C, u_[0,k]+uss >= 0.0, u_[0,k]+uss <= u_lim, x_[8, k] >= 70, x[8, k] <= 180]
        constr_hard += [x[:, k + 1] == A @ x[:, k] + B * (u[0,k]+uss) * to + M * meals[k]  + C, u[0,k]+uss >= 0.0, u[0,k]+uss <= u_lim, x[8, k] <= 175 + 5*(x0[8]>160), x[8, k] >= 70]
    for k in range(K//2, K):
        cost +=  cp.quad_form(x[:, k] - xss,Q) + r*cp.sum_squares(u[0,k])
        cost_ +=  cp.quad_form(x_[:, k] - xss,Q_) + r*cp.sum_squares(u_[0,k])
        # cost +=  10*cp.sum_squares(x[:, k]-xss[8]) + r*cp.sum_squares(u[0,k])
        # meal_impact[:] = [0, [time_index+k]]
        constr += [x_[:, k + 1] == A @ x_[:, k] + B * (u_[0,k]+uss) * to + M * meals[k]  + C, u_[0,k]+uss >= 0.0, u_[0,k]+uss <= u_lim, x_[8, k] >= 70]
        constr_hard += [x[:, k + 1] == A @ x[:, k] + B * (u[0,k]+uss) * to + M * meals[k] + C, u[0,k]+uss >= 0.0, u[0,k]+uss <= u_lim, x[8, k] <= 200, x[8, k] >= 70]

    # Add initial and terminal state constraint
    # cost +=  cp.quad_form(x[:, K] - xss,Qf)
    constr1 = constr + [x_[:, 0] == x0]#, cp.abs(x[8, K] - 140) <= 80]#, cp.abs(x[8, K]-120) <= 150, cp.abs(x[0, K]-120) <= 150]
    constr_hard = constr_hard + [x[:, 0] == x0]#, cp.abs(x[8, K] - 140) <= 80]#, cp.abs(x[8, K]-120) <= 150, cp.abs(x[0, K]-120) <= 150]
    problem = cp.Problem(cp.Minimize(cost), constr_hard)
    problem.solve(solver=cp.SCS, verbose = False)
    r_u = u
    r_x = x
    if (problem.status == "infeasible"):
        #print("INFEASIBLE")
        problem = cp.Problem(cp.Minimize(cost_), constr1)
        problem.solve(solver=cp.SCS, verbose = False)
        r_u = u_
        r_x = x_

    

    # assert(problem.status == "infeasible")
    if (problem.status == "infeasible" or r_u[0,0].value == None):
        print("INFEASIBLE ################################################")
        assert(0)
        print("x0: ", x0)
        print(x[:,K].value)
        # assert(False)
        if x0[0]>140:
            return 0.035, dss
        return 0, dss
    
    # print("FEASIBLE")
    # print("x0")
    # print(r_x[:,0].value)
    # print("x1")
    # print(r_x[:,1].value)
    # # print("xK")
    # # print(x[:,K].value)
    # # for k in range(K+1):
    # #     print("x"+(str(k))+": ", end = "")
    # print(r_x[:,K].value)
    # print("u:")
    # for i in r_u[0,:].value:
    #     print(i, end = " ; ")
    # print()

    # assert(0)
    # assert(u[0,0].value<5)
    # print("u = ", r_u[0,0].value + uss)
    previous_ba = max(min(r_u[0,0].value + uss, 5), 0)
    # print("uss: ", uss)
    # assert(0)
    return previous_ba, dss 