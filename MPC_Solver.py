import Dynamics
import casadi 
from IPython.display import HTML
import Environment
import matplotlib.animation as animation
import numpy as np

def get_MPC_solver(params):
    T = params["Horizon"] # MPC horizon
    N = params["N"] # horion divides on N steps

    # boundary conditions
    U_MAX = params["U_MAX"]
    U_MIN = - U_MAX

    x_dim = params["x_dim"]
    u_dim = params["u_dim"]

    intg = params["integrator"]

    opti = casadi.Opti()
    x = opti.variable(x_dim*(N+1)) # all state history shape x_dim*(N+1)
    u = opti.variable(u_dim *N) # u history
    x0 = opti.parameter(x_dim) # initial x0 state
    xgoal = opti.parameter(x_dim*(N+1)) # goal state
    
    # opti.parameter is const, we will set up it latter
    # opti.variable will be variated by casadi methods
    
    weights = params["weights"]

    dx = x - xgoal # error
    weighted_sum = dx.T @ np.diag(weights * (N+1)) @dx 

    opti.minimize(weighted_sum) # try minimize error

    opti.subject_to(x[0:x_dim] == x0)
    for k in range(1,N+1):
        opti.subject_to(x[k*x_dim : (k+1)*x_dim] == \
                        intg(x0 = x[(k-1)*x_dim : k*x_dim],p = u[(k-1)*u_dim : k*u_dim])['xf'])

    for i in range(N):
            opti.subject_to(opti.bounded(U_MIN,u[i*u_dim],U_MAX))
    # constract solver from "opti" settings
    opti_options = {'print_in':False, 'print_out':False, 'print_time':False}
    solver_options = {'print_level' : 0}

    opti.solver('ipopt',opti_options,solver_options)
    mpc = opti.to_function("MPC",[x0,xgoal],[u],["x0","xgoal"],["u_opt"])        
    
    return mpc