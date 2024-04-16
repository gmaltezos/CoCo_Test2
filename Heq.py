from cvxpy.reductions.solvers.defines import QP_SOLVERS
import cvxpy as cp
import numpy as np
from random import randint, seed

T = 10
t = 1
xkm1= [0,0,0,0,0,0,0,0,250]
previous_ba = 0

seed(10)

meal = 0
meals = []
inputs = []
states = []
K = 15
n_past = 15

def get_H_eq(model, x0, tau, time_index, inputs_, meals_, xss, uss, to, dss):
    global t,T, xkm1, previous_ba, meal, meals, inputs, states

    # A, B, Co, M = model.get_state_equations(1, xss)
    # # C = np.array([0]*8 + [1])
    # # C = np.diag([0 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 1])
    # C = np.array([[0, 1] + [0]*7, [0]*8 + [1]])
    # accu = C
    # O = C
    # print("#past_data "+str(1)+": "+"rank = ", np.linalg.matrix_rank(O))
    # for i in range(1,11):
    #     accu = accu@A
    #     O = np.vstack((O, accu))
    #     # print(accu)
    #     print("#past_data "+str(i+1)+": "+"rank = ", np.linalg.matrix_rank(O))
    # assert(0)
        




    if (t == T):
        t=1
        meal += meals_[time_index]
        # print(meal)
        meals+=[meal*2]
        # previous_ba = max(randint(0, 10_000)/10_000_000,0)
        previous_ba = max(0,uss + np.random.normal(size=(1,1))[0,0]/1000*2)
        inputs+=[previous_ba]
        # states+=[np.array([x0[8]])]
        states+=[np.array([x0[5],x0[8]])]
        # states+=[np.array(x0)]
        meal = 0
    else:
        meal+=meals_[time_index]
        t+=1


    
    return previous_ba, dss
    
