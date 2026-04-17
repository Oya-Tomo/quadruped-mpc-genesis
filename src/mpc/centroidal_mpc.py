import casadi as ca
import numpy as np
import scipy.sparse as sp
from .com_trajectory import ComTraj
from .go2_robot_data import PinGo2Model
import time

COST_MATRIX_Q = np.diag([1, 1, 50,  10, 20, 1,  2, 2, 1,  1, 1, 1])
COST_MATRIX_R = np.diag([1e-5] * 12)
MU = 0.8
NX = 12
NU = 12

OPTS = {
    'warm_start_primal': True,
    'warm_start_dual': True,
    "osqp": {
        "eps_abs": 1e-4,
        "eps_rel": 1e-4,
        "max_iter": 1000,
        "polish": False,
        "verbose": True,
        'adaptive_rho': True,
        "check_termination": 10,
        'adaptive_rho_interval': 25,
        "scaling": 5,
        "scaled_termination": True
    }
}

SOLVER_NAME = "osqp"

class CentroidalMPC:
    def __init__(self, go2: PinGo2Model, traj: ComTraj):
        self.Q = COST_MATRIX_Q 
        self.R = COST_MATRIX_R 
        self.nvars = traj.N * NX + traj.N * NU
        self.solve_time = 0 
        self.update_time = 0
        self.N = traj.N

        self.I_block = ca.DM.eye(self.N * NX)
        ones_N_minus_1 = np.ones(self.N - 1)
        S_scipy = sp.kron(sp.diags([ones_N_minus_1], [-1]), sp.eye(NX))
        self.S_block = self._scipy_to_casadi(S_scipy)

        self.A_ineq_static = self._precompute_friction_matrix(traj)
        self.dyn_builder = self._create_dynamics_function()
        self._build_sparse_matrix(traj)

    def solve_QP(self, go2: PinGo2Model, traj: ComTraj, verbose: bool = False):
        t0 = time.perf_counter()

        g, A, lba, uba = self._update_sparse_matrix(traj)
        lbx, ubx = self._compute_bounds(traj)
        
        t1 = time.perf_counter()

        qp_args = {
            'h': self.H_const,
            'g': g,
            'a': A,
            'lba': lba, 
            'uba': uba, 
            'lbx': lbx, 
            'ubx': ubx
        }
        
        if hasattr(self, 'x_prev') and self.x_prev is not None:
            qp_args['x0'] = self.x_prev
            qp_args['lam_x0'] = self.lam_x_prev
            qp_args['lam_a0'] = self.lam_a_prev

        sol = self.solver(**qp_args)
        t2 = time.perf_counter()

        self.update_time = (t1 - t0) * 1e3
        self.solve_time = (t2 - t1) * 1e3

        self.x_prev = sol["x"]
        self.lam_x_prev = sol["lam_x"]
        self.lam_a_prev = sol["lam_a"]

        return sol

    def _compute_bounds(self, traj: ComTraj):
        fz_min = 10
        N = traj.N      
        nvars = self.nvars
        start_u = N * 12

        lbx_np = np.full((nvars, 1), -np.inf, dtype=float)
        ubx_np = np.full((nvars, 1),  np.inf, dtype=float)

        force_block = (np.arange(12)[:, None] + 12*np.arange(N)[None, :])
        force_idx   = start_u + force_block

        contact = np.asarray(traj.contact_table, dtype=bool).T  # (N, 4) -> wait, contact_table is (N, 4)?
        # Let's check gait.py: return contact_table -> shape is (N, 4)?
        # No, in gait.py compute_contact_table: contact_table = (phases < self.gait_duty).astype(np.int32)
        # phases shape: (4, N) so contact_table is (4, N)
        if contact.shape == (traj.N, 4):
            contact = contact.T

        leg_rows = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9,10,11]])
        swing = ~contact
        swing_xyz = np.repeat(swing[:, None, :], 3, axis=1)

        mask_12N = np.zeros((12, N), dtype=bool)
        mask_12N[leg_rows.reshape(-1), :] = swing_xyz.reshape(12, N)

        swing_idx = force_idx[mask_12N]
        lbx_np[swing_idx, 0] = 0.0
        ubx_np[swing_idx, 0] = 0.0

        fz_rows = np.array([2, 5, 8, 11])
        stance_idx_2d = force_idx[fz_rows[:, None], np.arange(N)[None, :]]
        stance_mask   = contact
        stance_idx    = stance_idx_2d[stance_mask]

        lbx_np[stance_idx, 0] = np.maximum(lbx_np[stance_idx, 0], fz_min)

        return ca.DM(lbx_np), ca.DM(ubx_np)
    
    def _build_sparse_matrix(self, traj: ComTraj):
        rows, cols, vals = [], [], []
        for k in range(self.N):
            base = k*NX
            for i in range(NX):
                if self.Q[i,i] != 0:
                    rows.append(base+i)
                    cols.append(base+i)
                    vals.append(2*self.Q[i,i])
        
        for k in range(self.N):
            base = self.N*NX + k*NU
            for i in range(NU):
                if self.R[i,i] != 0:
                    rows.append(base+i)
                    cols.append(base+i)
                    vals.append(2*self.R[i,i])
        
        self.H_const = ca.DM.triplet(rows, cols, ca.DM(vals), self.nvars, self.nvars)
        self.H_sp = self.H_const.sparsity()

        Ad_dm = ca.DM(traj.Ad)
        Bd_stacked_np = traj.Bd.reshape(self.N * NX, NU)
        Bd_seq_dm = ca.DM(Bd_stacked_np)
        
        A_init = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)
        self.A_sp = A_init.sparsity()

        qp = {'h': self.H_sp, 'a': self.A_sp}
        self.solver = ca.conic('S', SOLVER_NAME, qp, OPTS)

    def _update_sparse_matrix(self, traj: ComTraj):
        Ad_dm = ca.DM(traj.Ad) 
        Bd_stacked_np = traj.Bd.reshape(self.N * NX, NU)
        Bd_seq_dm = ca.DM(Bd_stacked_np)
        A_dm = self._assemble_A_matrix(Ad_dm, Bd_seq_dm)

        Q_mat = ca.DM(self.Q)
        x_ref_np = traj.compute_x_ref_vec() 
        x_ref_dm = ca.DM(x_ref_np)
        gx_mat = -2 * (Q_mat @ x_ref_dm)
        g_x = ca.vec(gx_mat)
        g = ca.vertcat(g_x, ca.DM.zeros(self.N*NU, 1))

        x0 = ca.DM(traj.initial_x_vec)
        gd = ca.DM(traj.gd)
        beq_first = Ad_dm @ x0 + gd
        beq_rest  = ca.repmat(gd, self.N-1, 1)
        beq = ca.vertcat(beq_first, beq_rest)

        n_ineq = 4 * 4 * self.N
        l_ineq = -ca.inf * ca.DM.ones(n_ineq, 1)
        u_ineq_np = np.inf * np.ones(n_ineq)
        
        ct = traj.contact_table
        if ct.shape == (traj.N, 4):
            ct = ct.T

        idx = 0
        for k in range(self.N):
            for leg in range(4):
                if ct[leg, k] == 1: 
                    u_ineq_np[idx:idx+4] = 0.0
                idx += 4
        
        u_ineq = ca.DM(u_ineq_np)

        lb = ca.vertcat(beq, l_ineq)
        ub = ca.vertcat(beq, u_ineq)

        return g, A_dm, lb, ub
     
    def _assemble_A_matrix(self, Ad, Bd):
        big_minus_Ad, big_minus_Bd = self.dyn_builder(Ad, Bd)
        term_Ad = self.S_block @ big_minus_Ad
        A_eq = ca.horzcat(self.I_block + term_Ad, big_minus_Bd)
        A_total = ca.vertcat(A_eq, self.A_ineq_static)
        return A_total
    
    def _create_dynamics_function(self):
        Ad_sym = ca.SX.sym('Ad', NX, NX)
        Bd_seq_sym = ca.SX.sym('Bd_seq', self.N * NX, NU)
        
        list_Ad = [-Ad_sym] * self.N
        list_Bd = []

        for k in range(self.N):
            idx_start = k * NX
            idx_end   = (k + 1) * NX
            Bk = Bd_seq_sym[idx_start:idx_end, :]
            list_Bd.append(-Bk)
        
        big_Ad = ca.diagcat(*list_Ad)
        big_Bd = ca.diagcat(*list_Bd)
        
        return ca.Function('dyn_builder', [Ad_sym, Bd_seq_sym], [big_Ad, big_Bd])

    def _precompute_friction_matrix(self, traj):
        rows, cols, vals = [], [], []
        baseU = self.N * NX 
        r0 = 0
        
        for k in range(self.N):
            uk0 = baseU + k * NU
            for leg in range(4):
                fx, fy, fz = 3*leg, 3*leg+1, 3*leg+2
                
                rows.extend([r0, r0]); cols.extend([uk0+fx, uk0+fz]); vals.extend([1.0, -MU]); r0+=1
                rows.extend([r0, r0]); cols.extend([uk0+fx, uk0+fz]); vals.extend([-1.0, -MU]); r0+=1
                rows.extend([r0, r0]); cols.extend([uk0+fy, uk0+fz]); vals.extend([1.0, -MU]); r0+=1
                rows.extend([r0, r0]); cols.extend([uk0+fy, uk0+fz]); vals.extend([-1.0, -MU]); r0+=1

        A_sp = sp.csc_matrix((vals, (rows, cols)), shape=(r0, self.nvars))
        return self._scipy_to_casadi(A_sp)

    @staticmethod
    def _scipy_to_casadi(M):
        M = M.tocsc()
        return ca.DM(ca.Sparsity(M.shape[0], M.shape[1], M.indptr, M.indices), M.data)
