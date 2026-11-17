"""Giai đoạn 3 - tạo bộ giải AccelMPC của nhóm + chi phí cuối DARE (không sửa code của Thành viên A).

Vì sao cần DARE: trong repo, chi phí cuối Qf = Q nên chân trời chỉ 62 ms khiến drone gần như
"đứng yên" (điều khiển rất nhỏ). Thay khối cuối của P bằng nghiệm phương trình Riccati rời rạc
(P_inf) thì MPC hành xử như LQR vô hạn chân trời. Bộ giải vẫn là của nhóm, chỉ đổi P và phân rã
Cholesky.
"""
import os
import sys

import numpy as np
from scipy.linalg import solve_discrete_are, cholesky_banded

HERE = os.path.dirname(os.path.abspath(__file__))
def _find_solver_dir():
    """SOLVER_DIR (biến môi trường) > thư mục cha (khi file nằm trong repo) > thư mục anh em."""
    cands = [os.environ.get("SOLVER_DIR"), os.path.join(HERE, ".."),
             os.path.join(HERE, "..", "Crazyflie-AccelMPC-Solver")]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "src", "admm_mpc_solver.py")):
            return os.path.abspath(c)
    raise FileNotFoundError("Không tìm thấy src/admm_mpc_solver.py. Đặt: set SOLVER_DIR=<thư mục repo>")


SOLVER_DIR = _find_solver_dir()
sys.path.insert(0, SOLVER_DIR)
from src.admm_mpc_solver import AccelMPCSolver  # noqa: E402

Q = np.diag([100, 100, 100, 4, 4, 400, 4, 4, 4, 2, 2, 4]).astype(float)
R = 4.0 * np.eye(4)
SCALE_X = np.array([1e-2] * 3 + [0.5] * 3 + [0.3] * 3 + [20.0] * 3)


def make_solver(A, B, H=31, rho=4096.0, dare_terminal=True):
    S = AccelMPCSolver(A, B, Q, R, H=H, rho=rho, scale_x=SCALE_X, scale_u=np.ones(4))
    if dare_terminal:
        Qf = solve_discrete_are(A, B, Q, R)
        Dx = S.D.diagonal()[S.H * S.nxu:]
        last = slice(S.H * S.nxu, S.Nvar)
        P = S.P.tolil()
        P[last, last] = np.diag(Dx) @ Qf @ np.diag(Dx)
        S.P = P.tocsc()
        K = S.P + S.rho * (S.M.T @ S.M)
        S.h = S._get_bandwidth(K)
        S.K_banded = S._sparse_to_banded(K, S.h, lower=True)
        S.L_banded = cholesky_banded(S.K_banded, lower=True)
    return S
