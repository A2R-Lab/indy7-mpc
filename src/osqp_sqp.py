import numpy as np
import pinocchio as pin

class SQP_OSQP:
    def __init__(self, solver, stats=None):
        self.solver = solver
        self.stats = stats or {
            'qp_iters': {'values': [], 'unit': '', 'multiplier': 1},
            'linesearch_alphas': {'values': [], 'unit': '', 'multiplier': 1},
            'sqp_stepsizes': {'values': [], 'unit': '', 'multiplier': 1}
        }
    
    def eepos_cost(self, eepos_goals, XU):
        qcost = 0
        vcost = 0
        ucost = 0
        for k in range(self.solver.N):
            if k < self.solver.N-1:
                XU_k = XU[k*(self.solver.nx + self.solver.nu) : (k+1)*(self.solver.nx + self.solver.nu)]
                Q_modified = 1
            else:
                XU_k = XU[k*(self.solver.nx + self.solver.nu) : (k+1)*(self.solver.nx + self.solver.nu)-self.solver.nu]
                Q_modified = self.solver.QN_cost
            eepos = self.solver.eepos(XU_k[:self.solver.nq])
            eepos_err = eepos.T - eepos_goals[k*3:(k+1)*3]
            qcost += Q_modified * np.dot(eepos_err, eepos_err)
            vcost += self.solver.dQ_cost * np.dot(XU_k[self.solver.nq:self.solver.nx].reshape(-1), XU_k[self.solver.nq:self.solver.nx].reshape(-1))
            if k < self.solver.N-1:
                ucost += self.solver.R_cost * np.dot(XU_k[self.solver.nx:self.solver.nx+self.solver.nu].reshape(-1), XU_k[self.solver.nx:self.solver.nx+self.solver.nu].reshape(-1))
        return qcost, vcost, ucost

    def integrator_err(self, XU):
        err = 0
        for k in range(self.solver.N-1):
            xu_stride = (self.solver.nx + self.solver.nu)
            qcur = XU[k*xu_stride : k*xu_stride + self.solver.nq]
            vcur = XU[k*xu_stride + self.solver.nq : k*xu_stride + self.solver.nx]
            ucur = XU[k*xu_stride + self.solver.nx : (k+1)*xu_stride]

            a = pin.aba(self.solver.model, self.solver.data, qcur, vcur, ucur)
            qnext = pin.integrate(self.solver.model, qcur, vcur*self.solver.dt)
            vnext = vcur + a*self.solver.dt

            qnext_err = qnext - XU[(k+1)*xu_stride : (k+1)*xu_stride + self.solver.nq]
            vnext_err = vnext - XU[(k+1)*xu_stride + self.solver.nq : (k+1)*xu_stride + self.solver.nx]
            err += np.linalg.norm(qnext_err) + np.linalg.norm(vnext_err)
        return err

    def linesearch(self, XU, XU_fullstep, eepos_goals):
        mu = 10.0

        base_qcost, base_vcost, base_ucost = self.eepos_cost(eepos_goals, XU)
        integrator_err = self.integrator_err(XU)
        baseCV = integrator_err + np.linalg.norm(XU[:self.solver.nx] - XU[:self.solver.nx])
        basemerit = base_qcost + base_vcost + base_ucost + mu * baseCV
        diff = XU_fullstep - XU

        alphas = np.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125])
        fail = True
        for alpha in alphas:
            XU_new = XU + alpha * diff
            qcost_new, vcost_new, ucost_new = self.eepos_cost(eepos_goals, XU_new)
            integrator_err = self.integrator_err(XU_new)
            CV_new = integrator_err + np.linalg.norm(XU_new[:self.solver.nx] - XU[:self.solver.nx])
            merit_new = qcost_new + vcost_new + ucost_new + mu * CV_new
            exit_condition = (merit_new <= basemerit)

            if exit_condition:
                fail = False
                break

        alpha = 0.0 if fail else alpha
        self.stats['linesearch_alphas']['values'].append(alpha)
        return alpha
    
    def sqp(self, xcur, eepos_goals, XU):
        for qp in range(2):
            sol = self.solver.setup_and_solve_qp(XU, xcur, eepos_goals)
            
            alpha = self.linesearch(XU, sol.x, eepos_goals)
            if alpha == 0.0:
                continue

            step = alpha * (sol.x - XU)
            XU = XU + step
            
            stepsize = np.linalg.norm(step)
            self.stats['sqp_stepsizes']['values'].append(stepsize)

            if stepsize < 1e-3:
                break
        self.stats['qp_iters']['values'].append(qp+1)
        return XU
    
    def get_stats(self):
        return self.stats