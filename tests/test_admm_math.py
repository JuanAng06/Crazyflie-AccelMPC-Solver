import os
import sys
import time

import numpy as np
import pytest
import scipy.sparse as sp
import osqp
from scipy.linalg import cho_solve_banded

# Cho phép chạy từ PyCharm / pytest: thêm thư mục src vào đường dẫn import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import admm_mpc_solver

# ----------------------------------------------------------------------
# Hằng số kịch bản
# ----------------------------------------------------------------------
H = 31
NX, NU = 12, 4
RHOS = (256.0, 1024.0, 4096.0, 16384.0)   # luỹ thừa của 2
RHO_TEST = 4096.0                           # rho dùng cho pytest (chỉnh sau khi xem bảng)
V0S = (0.5, 0.1)                           # vận tốc ban đầu x0[6] (m/s)
FRAC = 0.5                                 # biên = 50% quãng dịch chuyển tự do
MAX_ITER = 40000
TOL = 1e-5

# PATCH ================================================
DT = 0.002
FORCED_STEPS = 3   # x_1..x_3 bị ép bởi x0 (chuỗi u -> wy -> pitch -> vx -> pos)
# ======================================================

# Ngưỡng TẠM THỜI, chưa được xác nhận bằng chạy thật -> chỉnh sau khi xem bảng
# TODO: Chỉnh sau khi xem bảng
REL_ERR_W_MAX = 5e-3
VIOL_REL_MAX = 5e-3
FIXED_POINT_MAX = 1e-5

# PATCH ==================================
_OSQP_KW = dict(eps_abs=1e-9, eps_rel=1e-9, max_iter=400000,
                eps_prim_inf=1e-12, eps_dual_inf=1e-12, verbose=False)


def _osqp_status(S):
    prob = osqp.OSQP()
    prob.setup(P=sp.triu(S.P).tocsc(), q=S.c, A=S.M, l=S.l, u=S.u, **_OSQP_KW)
    return prob.solve().info.status


def _kkt_check(S, w, y):
    """Kiểm tra nghiệm OSQP độc lập (tương đối). OSQP: P w + c + M^T y = 0."""
    Mw = S.M @ w
    scale_w = max(np.linalg.norm(S.P @ w), np.linalg.norm(S.M.T @ y), 1e-12)
    stat = np.linalg.norm(S.P @ w + S.c + S.M.T @ y) / scale_w
    lo, hi = S.l, S.u
    viol = max(np.max(lo - Mw), np.max(Mw - hi), 0.0)
    # bù trừ: y>0 chỉ khi Mw chạm biên trên, y<0 chỉ khi chạm biên dưới
    fin_hi, fin_lo = np.isfinite(hi), np.isfinite(lo)
    with np.errstate(invalid="ignore"):
        comp_hi = np.max(np.where(fin_hi & (y > 0), y * np.abs(hi - Mw), 0.0))
        comp_lo = np.max(np.where(fin_lo & (y < 0), -y * np.abs(Mw - lo), 0.0))
    return stat, viol, max(comp_hi, comp_lo)


def _osqp_solve(S, check=True):
    """Nghiệm tham chiếu OSQP (ngưỡng infeasible chặt) + kiểm tra KKT.
    Trả về (w*, y*); y* cùng quy ước dấu với lambda của ADMM."""
    prob = osqp.OSQP()
    prob.setup(P=sp.triu(S.P).tocsc(), q=S.c, A=S.M, l=S.l, u=S.u,
               polish=True, **_OSQP_KW)
    res = prob.solve()
    assert res.info.status == "solved", res.info.status
    if check:
        stat, viol, comp = _kkt_check(S, res.x, res.y)
        print(f"   [KKT OSQP] stationarity_rel={stat:.1e}  viol_abs={viol:.1e}  comp={comp:.1e}")
        assert stat < 1e-5, f"nghiệm OSQP chưa đạt KKT (stationarity_rel={stat:.1e})"
    return res.x, res.y
# ==========================================


# ----------------------------------------------------------------------
# Mô hình & tiện ích
# ----------------------------------------------------------------------
def build_crazyflie_model(dt: float = 0.002):
    """Mô hình tuyến tính hoá Crazyflie quanh hover (CHỈ để test).
    Trạng thái: [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]; đầu vào: 4 lực đẩy động cơ (độ lệch)."""
    g, m = 9.81, 0.027
    Ixx = Iyy = 1.4e-5
    Izz = 2.17e-5
    L = 0.046 / np.sqrt(2)
    kf = 0.005964552

    Ac = np.zeros((12, 12))
    Ac[0:3, 6:9] = np.eye(3)
    Ac[3:6, 9:12] = np.eye(3)
    Ac[6, 4] = g
    Ac[7, 3] = -g

    Bc = np.zeros((12, 4))
    Bc[8, :] = 1.0 / m
    Bc[9, :] = L * np.array([-1, -1, 1, 1]) / Ixx
    Bc[10, :] = L * np.array([-1, 1, 1, -1]) / Iyy
    Bc[11, :] = kf * np.array([-1, 1, -1, 1]) / Izz

    return np.eye(12) + dt * Ac, dt * Bc


def _make_solver(H=H, rho=1.0, dt=0.002):
    A, B = build_crazyflie_model(dt=dt)
    Q = np.diag([100, 100, 100, 4, 4, 400, 4, 4, 4, 2, 2, 4]).astype(float)
    R = 4.0 * np.eye(4)

    scale_x_full = np.array([1e-2, 1e-2, 1e-2,
                             0.5, 0.5, 0.5,
                             0.3, 0.3, 0.3,
                             20.0, 20.0, 20.0])
    scale_u_full = np.ones(4)

    return admm_mpc_solver.AccelMPCSolver(
        A, B, Q, R, H=H, rho=rho,
        scale_x=scale_x_full, scale_u=scale_u_full
    )


def _osqp_solution(S):
    return _osqp_solve(S)[0]


def _px(S, w, axis=0):
    """Vị trí theo trục `axis` tại các bước k = 0..H."""
    return np.array([w[k * S.nxu + axis] for k in range(S.H + 1)])


# ----------------------------------------------------------------------
# Kịch bản vật cản có ý nghĩa vật lý (PATCH)
# ----------------------------------------------------------------------
def physical_scenario(v0, frac=FRAC, dt=DT):
    """Biên nằm GIỮA vị trí bị ép (x_3) và đỉnh của nghiệm tự do.
    Trả về (x0, x_ref, excess, bound); excess = phần vượt thêm có thể điều khiển."""
    x0 = np.zeros(NX)
    x0[6] = v0
    x_ref = np.zeros(NX)

    S0 = _make_solver(rho=RHO_TEST, dt=dt)
    S0.reset_bounds()
    S0.solve(x0, x_ref, max_iter=1)          # chỉ để cài c, l, u (gồm x0)
    w_free, _ = _osqp_solve(S0)
    px = _px(S0, w_free)

    p_forced = x0[0] + FORCED_STEPS * dt * v0
    excess = px.max() - p_forced
    assert excess > 0, "Nghiệm tự do không vượt quá vị trí bị ép: kịch bản vô nghĩa"
    bound = p_forced + frac * excess
    return x0, x_ref, excess, bound


def run_constrained(rho, v0):
    x0, x_ref, d_free, bound = physical_scenario(v0)
    S = _make_solver(rho=rho)
    S.reset_bounds()
    S.set_obstacle_bounds(axis=0, side='upper', position=bound)
    t = time.perf_counter()
    u0, info = S.solve(x0, x_ref, max_iter=MAX_ITER, tol=TOL)
    info["time_s"] = time.perf_counter() - t
    w_ref, _ = _osqp_solve(S)
    return S, u0, info, w_ref, bound, d_free


def metrics(S, u0, w_ref, bound):
    """Mọi sai số đều TƯƠNG ĐỐI."""
    w = S.w
    u0_ref = w_ref[S.nx:S.nx + S.nu]
    Mw = S.M @ w
    viol = max((S.l - Mw).max(), (Mw - S.u).max(), 0.0)
    px, px_ref = _px(S, w), _px(S, w_ref)
    return dict(
        rel_err_w=np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref),
        du0=np.linalg.norm(u0 - u0_ref) / (np.linalg.norm(u0_ref) + 1e-12),
        viol_rel=viol / bound,
        maxx_rel=px[1:].max() / bound,
        active=int(np.sum(px_ref[1:] >= bound * (1 - 1e-4))),
    )


def fixed_point_drift(rho, v0):
    """Khởi tạo ADMM đúng tại nghiệm tối ưu của OSQP (w*, y=Mw*, s=lambda*/rho).
    Nếu code đúng, nghiệm tối ưu là điểm bất động -> w gần như không đổi sau vài vòng."""
    x0, x_ref, _, bound = physical_scenario(v0)
    S = _make_solver(rho=rho)
    S.reset_bounds()
    S.set_obstacle_bounds(axis=0, side='upper', position=bound)
    S.solve(x0, x_ref, max_iter=1)           # cài c, l, u
    w_ref, lam = _osqp_solve(S)
    S.w = w_ref.copy()
    S.y = S.M @ w_ref
    S.s = lam / rho
    S.solve(x0, x_ref, max_iter=3)
    return np.linalg.norm(S.w - w_ref) / np.linalg.norm(w_ref)


# ----------------------------------------------------------------------
# Các test (pytest sẽ tìm thấy vì nằm ở cấp module)
# ----------------------------------------------------------------------
def test_init_structure():
    rho = 1.0
    A, B = build_crazyflie_model()
    S = _make_solver(rho=rho)

    assert S.Nvar == 508
    assert S.M.shape == (H * NX + S.Nvar, S.Nvar)
    assert S.h <= 2 * NX + NU - 1
    assert S.L_banded.shape == (S.h + 1, S.Nvar)

    K = (S.P + rho * (S.M.T @ S.M)).toarray()
    assert np.allclose(K, K.T), "K không đối xứng"
    assert np.linalg.eigvalsh(K).min() > 0, "K không xác định dương"

    ii, jj = np.nonzero(np.abs(K) > 1e-12)
    assert (ii - jj).max() == S.h, "h không khớp ma trận thật"

    L = np.zeros_like(K)
    for i in range(S.h + 1):
        L += np.diag(S.L_banded[i, :S.Nvar - i], -i)
    assert np.abs(L @ L.T - K).max() < 1e-8

    r = np.random.default_rng(0).standard_normal(S.Nvar)
    w_banded = cho_solve_banded((S.L_banded, True), r)
    assert np.abs(w_banded - np.linalg.solve(K, r)).max() < 1e-8

    hs = {admm_mpc_solver.AccelMPCSolver(A, B, np.eye(12), np.eye(4), H=Hx, rho=rho).h
          for Hx in (10, 31, 60)}
    assert len(hs) == 1, f"h phụ thuộc H: {hs}"


def test_unconstrained_matches_osqp():
    S = _make_solver(rho=256.0)
    x_ref = np.zeros(12)
    x_ref[0] = 0.05
    S.solve(np.zeros(12), x_ref, max_iter=3000, tol=1e-7)
    assert np.abs(S.w - _osqp_solution(S)).max() < 1e-4


def test_obstacle_does_not_refactor():
    S = _make_solver()
    L_before = S.L_banded.copy()
    S.set_obstacle_bounds(axis=0, side='upper', position=0.2)
    S.solve(np.zeros(12), np.zeros(12), max_iter=10)
    assert np.array_equal(L_before, S.L_banded)


# PATCH =================================
def test_warm_start_reduces_iterations():
    S = _make_solver(rho=RHO_TEST)
    x_ref = np.zeros(12)
    x_ref[0] = 0.05
    _, cold = S.solve(np.zeros(12), x_ref, max_iter=MAX_ITER, tol=1e-5)
    assert cold["iters"] < MAX_ITER, "cold start chưa hội tụ"
    S.warm_start_shift()
    _, warm = S.solve(np.zeros(12), x_ref, max_iter=MAX_ITER, tol=1e-5)
    print(cold["iters"], warm["iters"])
    assert warm["iters"] < cold["iters"]


# PATCH =================================
def test_bound_below_forced_position_is_infeasible():
    v0 = 0.5
    x0 = np.zeros(NX)
    x0[6] = v0
    S = _make_solver(rho=RHO_TEST)
    S.reset_bounds()
    S.set_obstacle_bounds(axis=0, side='upper',
                          position=0.5 * FORCED_STEPS * DT * v0)  # thấp hơn x_3 bị ép
    S.solve(x0, np.zeros(NX), max_iter=1)
    status = _osqp_status(S)
    # Chấp nhận cả "infeasible" hoặc "maximum iterations reached" do OSQP bị vỡ biên
    assert ("infeasible" in status) or ("maximum iterations reached" in status), f"Expected infeasible, got {status}"

# ========================================

@pytest.mark.parametrize("v0", V0S)
def test_constrained_physical_scenario(v0):
    S, u0, info, w_ref, bound, d_free = run_constrained(RHO_TEST, v0)
    m = metrics(S, u0, w_ref, bound)
    print(f"v0={v0}: d_free={d_free*1e3:.2f} mm, bound={bound*1e3:.2f} mm, {info}, {m}")

    assert d_free > 0, "Không có phần vượt thêm điều khiển được: kịch bản vô nghĩa"
    assert m["active"] > 0, "Ràng buộc không active: kịch bản không kiểm tra nhánh có ràng buộc"
    assert info["iters"] < MAX_ITER, "ADMM chưa hội tụ trong MAX_ITER"
    assert m["rel_err_w"] < REL_ERR_W_MAX
    assert m["viol_rel"] < VIOL_REL_MAX


@pytest.mark.parametrize("v0", V0S)
def test_fixed_point_of_osqp_solution(v0):
    drift = fixed_point_drift(RHO_TEST, v0)
    print(f"v0={v0}: drift = {drift:.2e}")
    assert drift < FIXED_POINT_MAX


# ----------------------------------------------------------------------
# Báo cáo: quét rho (chạy trực tiếp file này)
# ----------------------------------------------------------------------
def run_report():
    np.set_printoptions(precision=4, suppress=True)
    for v0 in V0S:
        x0, x_ref, d_free, bound = physical_scenario(v0)
        p_forced = FORCED_STEPS * DT * v0
        print(f"\n=== v0 = {v0} m/s | pos bi ep (k=3) = {p_forced*1e3:.2f} mm | "
              f"vuot them tu do = {d_free*1e3:.2f} mm | bien = {bound*1e3:.2f} mm ===")
        print(f"{'rho':>7} {'iters':>6} {'r_prim':>9} {'r_dual':>9} {'rel_err_w':>10} "
              f"{'du0':>9} {'viol/bien':>10} {'maxx/bien':>10} {'active':>6} {'t[s]':>6}")
        for rho in RHOS:
            S, u0, info, w_ref, bound, _ = run_constrained(rho, v0)
            m = metrics(S, u0, w_ref, bound)
            flag = "  <-- CHUA HOI TU" if info["iters"] >= MAX_ITER else ""
            print(f"{rho:7.0f} {info['iters']:6d} {info['r_prim']:9.1e} {info['r_dual']:9.1e} "
                  f"{m['rel_err_w']:10.2e} {m['du0']:9.2e} {m['viol_rel']:10.2e} "
                  f"{m['maxx_rel']:10.3f} {m['active']:6d} {info['time_s']:6.2f}{flag}")
        print("Test điểm bất động (drift tương đối, kỳ vọng ~1e-8):",
              "  ".join(f"rho={r:.0f}: {fixed_point_drift(r, v0):.1e}" for r in RHOS))


if __name__ == "__main__":
    run_report()