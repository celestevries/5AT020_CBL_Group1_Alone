import numpy as np
from qpsolvers import solve_qp

class MPCController:
    def __init__(self, env, Q, P, R, N=5):
        self.env = env
        self.N = N  # Prediction horizon
        
        self.Ae = np.vstack((np.hstack((env.ad, np.zeros((2,2)))),np.hstack((env.cd*env.ad, env.cd))))
        self.Be = np.vstack((env.bd,env.cd*env.bd))
        self.Ce = np.hstack((np.zeros((2,2)), np.eye(2)))
        
        # Initialize previous state and input vector
        self.x_old = np.zeros((env.ad.shape[0],1))
        self.u_old = np.zeros((self.Be.shape[1],1))

        # Prediction matrices
        self.n_xn = self.Ae.shape[0]  # Number of states in the augmented system
        self.n_du = self.Be.shape[1]  # Number of inputs in the augmented system
        self.n_y  = self.Ce.shape[0]  # Number of outputs in the augmented system
        self.Phi = np.zeros((self.N*self.n_y, self.n_xn))
        self.Gamma = np.zeros((self.N*self.n_y, self.N*self.n_du))
        for i in range(self.N):
            # Phi
            row_init = self.n_y*i;
            row_end  = row_init + self.n_y;
            self.Phi[row_init:row_end,:] = self.Ce @ np.linalg.matrix_power(self.Ae, i+1)

            # Gamma 
            for j in range(i+1):
                col_init = self.n_du*j;
                col_end  = col_init + self.n_du;
                self.Gamma[row_init:row_end,col_init:col_end] = self.Ce @ np.linalg.matrix_power(self.Ae, i-j) @ self.Be
        
        # Performance matrices
        Q = Q*np.eye(self.n_y)     # y-r (reference tracking error)
        P = P*Q                    # y_N-r_N (steady-state reference tracking error)
        R = R*np.eye(self.n_du)    # Δu (input difference)
        
        # Weight matrices
        self.Omega = np.kron(np.diag(np.hstack((np.ones(self.N-1),0))), Q) + \
                     np.kron(np.diag(np.hstack((np.zeros(self.N-1),1))), P)
        self.Psi   = np.kron(np.eye(self.N), R)

    def cost_function_matrices(self, yref):

        # Cost function matrices
        G = 2*(self.Psi + self.Gamma.T @ self.Omega @ self.Gamma)
        F = 2*self.Gamma.T @ self.Omega

        # Vector of reference values
        R_k = np.kron(np.ones((self.N, 1)), yref)

        # Positive semidefinite quadratic term matrix
        Q = (G+G.T)/2
        # Linear term vector
        q = F @ (self.Phi @ self.xn_old - R_k)

        return Q, q
    
    def restriction_matrices(self, umin, umax, dumin, dumax, ymin, ymax):

        C1 = np.kron(np.ones((self.N,1)), np.eye(self.n_du))
        C2 = np.tril(np.kron(np.ones((self.N,self.N)), np.eye(self.n_du)))
        # Input constraints
        M1 = np.vstack((-C2, 
                         C2))
        N1 = np.vstack(( np.kron(-umin, np.ones((self.N,1))) + C1 @ self.u_old, 
                         np.kron( umax, np.ones((self.N,1))) - C1 @ self.u_old))

        # Delta input constraints
        M2 = np.vstack((-np.eye(self.N*self.n_du), 
                         np.eye(self.N*self.n_du)))
        N2 = np.vstack((-np.kron( dumin, np.ones((self.N,1))), 
                         np.kron( dumax, np.ones((self.N,1)))))

        # Output constraints
        M3 = np.vstack((-self.Gamma, 
                         self.Gamma))
        N3 = np.vstack(( np.kron(-ymin, np.ones((self.N,1))) + self.Phi @ self.xn_old, 
                         np.kron( ymax, np.ones((self.N,1))) - self.Phi @ self.xn_old))
        
        M = np.vstack((M1, M2, M3))
        N = np.vstack((N1, N2, N3))

        return M, N
    
    def compute_input(self, x, y, yref, umax=600, dumax=1200, ymax=1.0000e+20):
        # Transform to column vectors
        x = np.array([x]).T
        y = np.array([y]).T
        yref = np.array([yref]).T

        # Define xn_old
        self.xn_old = np.vstack((x - self.x_old,    # measured states - previous states
                                  y))               # measured output
        self.x_old = x;

        # Input constraints
        umax  = np.array([[ umax], 
                          [ umax]])
        umin  = -umax
        # Input variation constraints
        dumax = np.array([[ dumax], 
                          [ dumax]])
        dumin = -dumax
        # Output constraints
        ymax = np.array([[ ymax], 
                         [ ymax]])
        ymin  = -ymax
        

        # Weights
        Q, q = self.cost_function_matrices(yref)
        M, n = self.restriction_matrices(umin, umax, dumin, dumax, ymin, ymax)
        
        # du = solve_qp(Q,q,M,n, solver="osqp")
        du = solve_qp(Q,q, solver="osqp")
        u  = np.array([du[0:2]]).T + self.u_old;
        self.u_old = u;

        return u.flatten()
    
class PIController:
    def __init__(self, Kp, Ki, Ts, umax):
        self.Kp = Kp                # Proportional gain
        self.Ki = Ki                # Integral gain
        self.Ts = Ts                # Sampling time [s]
        self.umax = umax            # Maximum input [V]
        self.error_sum = np.zeros(2)    # Initialization of error accumulation
        
    def reset(self):
        # Reset error accumulation
        self.error_sum = np.zeros(2)
        
    def control(self, reference, measured):
        # Define error
        error = reference - measured

        # Define error accumulation
        self.error_sum += error * self.Ts
        
        # Compute action "u"
        # u = Kp*e + Ki*Ts*Σe
        u = self.Kp * error + self.Ki * self.error_sum
        
        # Limit action
        u = np.clip(u, -self.umax, self.umax)
        
        return u