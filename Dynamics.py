import casadi as cs
from scipy import integrate


l = cs.SX.sym('l',2)
m = cs.SX.sym('m',3)
phi = cs.SX.sym('phi',3)
dphi = cs.SX.sym('dphi',3)

G = cs.SX.sym('g')
g = cs.vcat([0,G])

u = cs.SX.sym('u')
f = cs.vcat([u,0,0])
P1 = cs.vcat([phi[0],0])
P2 = cs.vcat([l[0] * cs.cos(phi[1]), l[0] * cs.sin(phi[1])]) + P1
P3 = cs.vcat([l[1] * cs.cos(phi[2]), l[1] * cs.sin(phi[2])]) + P2
P = cs.horzcat(P1,P2,P3)

J1 = cs.jacobian(P1,phi)
J2 = cs.jacobian(P2,phi)
J3 = cs.jacobian(P3,phi)
J = [J1,J2,J3]

T = cs.SX([0])
U = cs.SX([0])
for i in range(3):
    T += m[i]/2 *(J[i] @ dphi).T @ J[i] @ dphi 
    U += m[i] * g.T @ P[:,i]

def diff_time(J,x,dx):
    J_dot = []
    for vect in cs.horzsplit(J):
        J_dot.append(cs.jacobian(vect,x) @ dx)
    J_dot = cs.hcat(J_dot)
    return J_dot

J1_dot = diff_time(J1,phi,dphi)
J2_dot = diff_time(J2,phi,dphi)
J3_dot = diff_time(J3,phi,dphi)
J_dot = [J1_dot,J2_dot,J3_dot]

A = cs.SX(3,3)
B = cs.SX(3,3)
C = cs.SX(3,3)
D = cs.SX(3,2)
for i in range(3):
    A += m[i] * (J_dot[i].T @ J[i] + J[i].T @ J_dot[i])
    B += m[i] * (J[i].T @ J[i])
    C += m[i] * (J_dot[i].T @ J[i])
    D += m[i] * (J[i].T)
rhs = cs.solve(B,(f + (C - A) @ dphi - D @ g))
rhs = cs.vertcat(dphi,rhs)

m_real = cs.SX([1,1,1])
l_real = cs.SX([1,1])
G_real = cs.SX([10])

my_rhs = cs.substitute([rhs],[m,l,G],[m_real,l_real,G_real])[0]
my_rhs = cs.Function('rhs',[phi,dphi,u],[my_rhs])

energy = cs.substitute([T + U],[m,l,G],[m_real,l_real,G_real])[0]
energy = cs.Function('energy',[phi,dphi],[energy])

my_P = cs.substitute([P],[m,l],[m_real,l_real])
my_P = cs.Function('P',[phi],my_P)

def get_next_state(state,u,dt):
    return integrate.odeint(lambda x,t: my_rhs.call([x[:3],x[3:],[u]])[0].T.full()[0] , state, [0,dt])[1]
def state_to_coords(state):
    return my_P.call([state[:3]])[0].full()
def get_energy(state):
     return energy.call([state[:3],state[3:]])[0].full()[0]
    