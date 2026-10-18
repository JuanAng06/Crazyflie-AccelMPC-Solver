import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky_banded, cho_solve_banded


class AccelMPCSolver:
    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 H: int = 31, rho: float = 4096.0, scale_x: np.ndarray = None, scale_u: np.ndarray = None):
        """
        Khởi tạo bộ giải AccelMPC với vector quyết định đan xen w và cơ chế Full Scaling.
        """
        self.nx = A.shape[0]
        self.nu = B.shape[1]
        self.nxu = self.nx + self.nu  # Thuộc tính hỗ trợ cho test_admm_math
        self.H = H
        self.rho = rho
        self.Nvar = (self.H + 1) * self.nx + self.H * self.nu

        # =========================================================
        # 1. Thiết lập Ma trận Scaling (D) cho w
        # =========================================================
        if scale_x is None:
            scale_x = np.ones(self.nx)
        if scale_u is None:
            scale_u = np.ones(self.nu)

        D_diag = np.zeros(self.Nvar)
        for k in range(self.H):
            w_start = k * self.nxu
            D_diag[w_start: w_start + self.nx] = scale_x
            D_diag[w_start + self.nx: w_start + self.nxu] = scale_u
        D_diag[self.H * self.nxu:] = scale_x

        self.D = sp.diags(D_diag, format='csc')
        self.D_inv_diag = 1.0 / D_diag

        # =========================================================
        # 2. Dựng Ma trận P và M nguyên bản (chưa scale)
        # =========================================================
        P_blocks = []
        for _ in range(self.H):
            P_blocks.append(Q)
            P_blocks.append(R)
        P_blocks.append(Q)
        P_orig = sp.block_diag(P_blocks, format='csc')

        dyn_rows = self.H * self.nx
        M_dyn = sp.lil_matrix((dyn_rows, self.Nvar))
        for k in range(self.H):
            row_start = k * self.nx
            w_start = k * self.nxu
            M_dyn[row_start:row_start + self.nx, w_start:w_start + self.nx] = A
            M_dyn[row_start:row_start + self.nx, w_start + self.nx:w_start + self.nxu] = B
            next_x_start = w_start + self.nxu
            M_dyn[row_start:row_start + self.nx, next_x_start:next_x_start + self.nx] = -np.eye(self.nx)

        M_bound = sp.eye(self.Nvar, format='lil')
        M_orig = sp.vstack([M_dyn, M_bound], format='csc')
        self.nd = dyn_rows

        # =========================================================
        # 3. Áp dụng Full Scaling
        # =========================================================
        self.P = self.D @ P_orig @ self.D
        self.M = M_orig @ self.D

        # =========================================================
        # 4. Offline Pre-factorization (Cholesky Dạng Băng)
        # =========================================================
        K_sparse = self.P + self.rho * (self.M.T @ self.M)
        self.h = self._get_bandwidth(K_sparse)
        self.K_banded = self._sparse_to_banded(K_sparse, self.h, lower=True)
        self.L_banded = cholesky_banded(self.K_banded, lower=True)

        # =========================================================
        # 5. Khởi tạo các vector cho vòng lặp ADMM
        # =========================================================
        self.w = np.zeros(self.Nvar)
        self.y = np.zeros(self.M.shape[0])
        self.s = np.zeros(self.M.shape[0])
        self.l = np.full(self.M.shape[0], -np.inf)
        self.u = np.full(self.M.shape[0], np.inf)

        # Hàng động lực học luôn có l = u = 0
        self.l[:self.nd] = 0.0
        self.u[:self.nd] = 0.0

    def _get_bandwidth(self, A_sparse):
        # Ép kiểu sang COO để có thể truy cập mảng tọa độ row và col
        A_coo = A_sparse.tocoo()
        return (A_coo.row - A_coo.col).max()

    def _sparse_to_banded(self, A_sparse, h, lower=True):
        A_dense = A_sparse.toarray()
        n = A_dense.shape[0]
        ab = np.zeros((h + 1, n))
        if lower:
            for i in range(h + 1):
                ab[i, :n - i] = np.diag(A_dense, k=-i)
        return ab

    def solve(self, x0, x_ref, u_ref=None, max_iter=50, tol=1e-5):
        """Giải bài toán bằng ADMM với Full Scaling."""
        H, nx, nu = self.H, self.nx, self.nu
        x_ref = np.asarray(x_ref, float)
        if x_ref.ndim == 1:
            x_ref = np.tile(x_ref, (H + 1, 1))
        u_ref = np.zeros((H, nu)) if u_ref is None else np.asarray(u_ref, float)
        if u_ref.ndim == 1:
            u_ref = np.tile(u_ref, (H, 1))

        # Scale tham chiếu (w_ref_scaled)
        w_ref = self._pack(x_ref, u_ref)
        w_ref_scaled = w_ref * self.D_inv_diag
        self.c = -(self.P @ w_ref_scaled)

        # Cài đặt điều kiện đầu ở hệ tọa độ vật lý (chưa scale)
        self.l[self.nd:self.nd + nx] = x0
        self.u[self.nd:self.nd + nx] = x0

        b = self.M.T @ (self.rho * (self.y - self.s))
        it, r_prim, r_dual = 0, np.inf, np.inf

        for it in range(1, max_iter + 1):

            # w-update
            self.w = cho_solve_banded((self.L_banded, True), b - self.c)
            v = self.M @ self.w  # Quay về đơn vị vật lý

            # y-update
            y_prev = self.y.copy()
            self.y = np.clip(v + self.s, self.l, self.u)

            # s-update
            self.s = self.s + v - self.y

            # Cập nhật vế phải b chính xác cho bước tiếp theo
            b = self.M.T @ (self.rho * (self.y - self.s))

            if tol is not None:
                r_prim = np.abs(v - self.y).max()
                r_dual = np.abs(self.M.T @ (self.rho * (self.y - y_prev))).max()
                if r_prim < tol and r_dual < tol:
                    break

        # Unscale kết quả u0 đầu ra
        u0_tilde = self.w[nx:nx + nu]
        u0 = u0_tilde / self.D_inv_diag[nx:nx + nu]

        return u0.copy(), {"iters": it, "r_prim": r_prim, "r_dual": r_dual}

    def _pack(self, x_ref, u_ref):
        w = np.zeros(self.Nvar)
        for k in range(self.H):
            w_start = k * self.nxu
            w[w_start:w_start + self.nx] = x_ref[k]
            w[w_start + self.nx:w_start + self.nxu] = u_ref[k]
        w[self.H * self.nxu:] = x_ref[self.H]
        return w

    def reset_bounds(self):
        """Khôi phục l, u của phần ràng buộc hộp về +-inf"""
        self.l[self.nd:] = -np.inf
        self.u[self.nd:] = np.inf
        # Đảm bảo điều kiện đầu cũng được reset
        self.l[self.nd:self.nd + self.nx] = 0.0
        self.u[self.nd:self.nd + self.nx] = 0.0

    def set_obstacle_bounds(self, axis, side, position):
        """Ghi đè vector biên (chỉ áp dụng từ k=1 đến H) theo trục vật lý"""
        for k in range(1, self.H + 1):
            idx = self.nd + k * self.nxu + axis
            if side == 'upper':
                self.u[idx] = position
            elif side == 'lower':
                self.l[idx] = position

    def warm_start_shift(self):
        """Dịch chuyển biến y và s để warm-start cho bước thời gian tiếp theo."""
        stride = self.nxu

        # Shift y
        y_new = np.zeros_like(self.y)
        y_new[:self.nd - self.nx] = self.y[self.nx:self.nd]
        y_new[self.nd: -stride] = self.y[self.nd + stride:]
        self.y = y_new

        # Shift s
        s_new = np.zeros_like(self.s)
        s_new[:self.nd - self.nx] = self.s[self.nx:self.nd]
        s_new[self.nd: -stride] = self.s[self.nd + stride:]
        self.s = s_new