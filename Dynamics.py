from casadi import *
import casadi as cs
from scipy import integrate
t = SX.sym('t')
l = SX.sym('l',2)
m = SX.sym('m',3)
phi = SX.sym('phi',3)
dphi = SX.sym('dphi',3)

G = SX.sym('g')
g = vcat([0,G])

u = SX.sym('u')
f = vcat([u,0,0])


P1 = vcat([phi[0],0])
P2 = vcat([l[0] * cos(phi[1]), l[0] * sin(phi[1])]) + P1
P3 = vcat([l[1] * cos(phi[2]), l[1] * sin(phi[2])]) + P2
P = horzcat(P1,P2,P3)

J1 = jacobian(P1,phi)
J2 = jacobian(P2,phi)
J3 = jacobian(P3,phi)
J = [J1,J2,J3]

T = SX([0])
U = SX([0])
for i in range(3):
    T += m[i]/2 *(J[i] @ dphi).T @ J[i] @ dphi
    U += m[i] * g.T @ P[:,i]

def diff_time(J,x,dx):
    J_dot = []
    for vect in horzsplit(J):
        J_dot.append(jacobian(vect,x) @ dx)
    J_dot = hcat(J_dot)
    return J_dot

J1_dot = diff_time(J1,phi,dphi)
J2_dot = diff_time(J2,phi,dphi)
J3_dot = diff_time(J3,phi,dphi)
J_dot = [J1_dot,J2_dot,J3_dot]

A = SX(3,3)
B = SX(3,3)
C = SX(3,3)
D = SX(3,2)
for i in range(3):
    A += m[i] * (J_dot[i].T @ J[i] + J[i].T @ J_dot[i])
    B += m[i] * (J[i].T @ J[i])
    C += m[i] * (J_dot[i].T @ J[i])
    D += m[i] * (J[i].T)
##### if you need friction assign mu ##########
mu = 0
Friction = cs.SX([[mu,0,0],[0,0,0],[0,0,0]])
###############################################
rhs = cs.solve(B,(f + (C - A - Friction) @ dphi - D @ g ))
rhs = cs.vertcat(dphi,rhs)

m_real = cs.SX([0.5,0.5,0.5])
l_real = cs.SX([1,1])
G_real = cs.SX([10])

rhs = solve(B,(f + (C - A) @ dphi - D @ g))
rhs = vertcat(dphi,rhs)

m_real = SX([1,1,0.5])
l_real = SX([1,1])
G_real = SX([10])

my_rhs = substitute([rhs],[m,l,G],[m_real,l_real,G_real])[0]
my_rhs = Function('rhs',[phi,dphi,u],[my_rhs])

energy = substitute([T + U],[m,l,G],[m_real,l_real,G_real])[0]
energy = Function('energy',[phi,dphi],[energy])

my_P = substitute([P],[m,l],[m_real,l_real])
my_P = Function('P',[phi],my_P)


def get_next_state(state, u, dt, normalize = True):
    next_state = integrate.odeint(lambda x, t: my_rhs.call([x[:3],x[3:],[u]])[0].T.full()[0] , state, [0,dt])[1]
    return next_state

def state_to_coords(state):
    return my_P.call([state[:3]])[0].full()
def get_energy(state):
     return energy.call([state[:3],state[3:]])[0].full()[0]
