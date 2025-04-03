import numpy as np
import pinocchio as pin
from scipy.sparse import bmat, csc_matrix, triu
import osqp

class OSQPSolver:
    def __init__(self, model, dt=0.01, N=32, dQ_cost=0.01, R_cost=1e-5, QN_cost=100, regularize=True, eps=1):
        # model things
        self.model = model
        self.data = model.createData()

        # environment
        self.N = N  # num knotpoints
        self.dt = dt  # time step

        # properties
        self.nq = model.nq  # j_pos
        self.nv = model.nv  # j_vel
        self.nx = self.nq + self.nv  # state size
        self.nu = len(model.joints) - 1  # num controls
        self.nxu = self.nx + self.nu  # state + control size
        self.traj_len = (self.nx + self.nu)*self.N - self.nu  # length of trajectory

        # cost
        self.dQ_cost = dQ_cost  # joint velocity cost
        self.R_cost = R_cost  # torque cost
        self.QN_cost = QN_cost  # final joint position cost
        self.regularize = regularize  # regularize velocities and controls
        self.eps = eps  # regularization parameter
        
        # sparse matrix templates
        self.A = self.initialize_A()  # linearized dynamics
        self.l = np.zeros(self.N*self.nx) 
        self.P = self.initialize_P()
        self.g = np.zeros(self.traj_len)
        self.Pdata = np.zeros(self.P.nnz)
        self.Adata = np.zeros(self.A.nnz)

        self.osqp = osqp.OSQP()
        osqp_settings = {'verbose': False}
        self.osqp.setup(P=self.P, q=self.g, A=self.A, l=self.l, u=self.l, **osqp_settings)

        # temporary variables
        self.A_k = np.vstack([-1.0 * np.eye(self.nx), np.vstack([np.hstack([np.eye(self.nq), self.dt * np.eye(self.nq)]), np.ones((self.nq, 2*self.nq))])])
        self.B_k = np.vstack([np.zeros((self.nq, self.nq)), np.zeros((self.nq, self.nq))])
        self.cx_k = np.zeros(self.nx)

    def initialize_P(self):
        block = np.eye(self.nxu)
        block[:self.nq, :self.nq] = np.ones((self.nq, self.nq))
        bd = np.kron(np.eye(self.N), block)[:-self.nu, :-self.nu]
        return csc_matrix(triu(bd), shape=(self.traj_len, self.traj_len))
    
    def initialize_A(self):
        blocks = [[np.ones((self.nx,self.nx))] + [None] * (2*self.N)]
        for i in range(self.N-1):
            row = []
            # Add initial zeros if needed
            for j in range(2*i):
                row.append(None)  # None is interpreted as zero block
            # Add A, B, I
            row.extend([np.ones((self.nx,self.nx)), 2 * np.ones((self.nx,self.nu)), -1 * np.ones((self.nx,self.nx))])
            # Pad remaining with zeros
            while len(row) < 2*self.N + 1:
                row.append(None)
            blocks.append(row)

        return bmat(blocks, format='csc')
    
    def compute_dynamics_jacobians(self, q, v, u):
        d_dq, d_dv, d_du = pin.computeABADerivatives(self.model, self.data, q, v, u)
        self.A_k[self.nx + self.nq:, :self.nq] = d_dq * self.dt
        self.A_k[self.nx + self.nq:, self.nq:2*self.nq] = d_dv * self.dt + np.eye(self.nv)
        self.B_k[self.nq:, :] = d_du * self.dt
        
        a = self.data.ddq
        qnext = pin.integrate(self.model, q, v * self.dt)
        vnext = v + a * self.dt
        xnext = np.hstack([qnext, vnext])
        xcur = np.hstack([q,v])
        self.cx_k = xnext - self.A_k[self.nx:] @ xcur - self.B_k @ u

    def update_constraint_matrix(self, xu, xs):
        # Fast update of the existing CSC matrix
        self.l[:self.nx] = -1 * xs  # negative because top left is negative identity
        Aind = 0
        for k in range(self.N-1):
            xu_stride = (self.nx + self.nu)
            qcur = xu[k*xu_stride : k*xu_stride + self.nq]
            vcur = xu[k*xu_stride + self.nq : k*xu_stride + self.nx]
            ucur = xu[k*xu_stride + self.nx : (k+1)*xu_stride]
            
            self.compute_dynamics_jacobians(qcur, vcur, ucur)
            
            self.Adata[Aind:Aind+self.nx*self.nx*2]=self.A_k.T.reshape(-1)
            Aind += self.nx*self.nx*2
            self.Adata[Aind:Aind+self.nx*self.nu]=self.B_k.T.reshape(-1)
            Aind += self.nx*self.nu

            self.l[(k+1)*self.nx:(k+2)*self.nx] = -1.0 * self.cx_k
        self.Adata[Aind:] = -1.0 * np.eye(self.nx).reshape(-1)

    def update_cost_matrix(self, XU, eepos_g):
        Pind = 0
        for k in range(self.N):
            if k < self.N-1:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)]
            else:
                XU_k = XU[k*(self.nx + self.nu) : (k+1)*(self.nx + self.nu)-self.nu]
            eepos, deepos = self.d_eepos(XU_k[:self.nq])
            eepos_err = np.array(eepos.T) - eepos_g[k*3:(k+1)*3]
            
            # cost multipliers
            dQ_modified = self.dQ_cost if not self.regularize else self.dQ_cost * (1/(abs(np.linalg.norm(eepos_err)) + self.eps))
            R_modified = self.R_cost if not self.regularize else self.R_cost * (1/(abs(np.linalg.norm(eepos_err)) + self.eps))
            Q_modified = self.QN_cost if k==self.N-1 else 1

            joint_err = eepos_err @ deepos

            g_start = k*(self.nx + self.nu)
            self.g[g_start : g_start + self.nx] = np.vstack([
                Q_modified * joint_err.T,
                (dQ_modified * XU_k[self.nq:self.nx]).reshape(-1)
            ]).reshape(-1)

            phessian = np.outer(joint_err, joint_err)
            pos_costs = Q_modified * phessian[np.tril_indices_from(phessian)]
            self.Pdata[Pind:Pind+len(pos_costs)] = pos_costs
            Pind += len(pos_costs)
            self.Pdata[Pind:Pind+self.nv] = np.full(self.nv, dQ_modified)
            Pind+=self.nv
            if k < self.N-1:
                self.Pdata[Pind:Pind+self.nu] = np.full(self.nu, R_modified)
                Pind+=self.nu
                self.g[g_start + self.nx : g_start + self.nx + self.nu] = R_modified * XU_k[self.nx:self.nx+self.nu].reshape(-1)

    def setup_and_solve_qp(self, xu, xs, eepos_g):
        self.update_constraint_matrix(xu, xs)
        self.update_cost_matrix(xu, eepos_g)
        self.osqp.update(Px=self.Pdata)
        self.osqp.update(Ax=self.Adata)
        self.osqp.update(q=self.g, l=self.l, u=self.l)
        return self.osqp.solve()
    
    # End effector methods
    def eepos(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        return self.data.oMi[6].translation

    def d_eepos(self, q):
        eepos_joint_id = 6
        pin.computeJointJacobians(self.model, self.data, q)
        deepos = pin.getJointJacobian(self.model, self.data, eepos_joint_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        eepos = self.data.oMi[6].translation
        return eepos, deepos