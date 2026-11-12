"""Kiểm tra Giai đoạn 2 KHÔNG cần PyBullet (chỉ numpy + scipy + solver của nhóm).

Chạy:   python check_phase2.py
Kỳ vọng: in "TẤT CẢ KIỂM TRA ĐẠT" ở cuối.

Hai bước cuối dùng chính solver của nhóm (chưa có PyBullet) để chắc rằng biên vật cản thật
sự chặn được vị trí và vòng điều khiển chạy thông.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from solver_ext import SOLVER_DIR   # tự tìm repo (xem _find_solver_dir)
sys.path.insert(0, SOLVER_DIR)

from crazyflie_model import (build_linear_model, initial_state, state20_to_x12, HOVER_THRUST_PER_MOTOR,
                             u_dev_to_rpm, U_DEV_MIN, U_DEV_MAX)
from reference import Figure8Reference, PointReference
from obstacles import separating_bound, update_solver_bounds
from src.admm_mpc_solver import AccelMPCSolver

H, DT = 31, 0.002


def check_model():
    A, B = build_linear_model(DT)
    assert A.shape == (12, 12) and B.shape == (12, 4)
    x0 = initial_state(pos=(0.1, 0.2, 0.5))
    assert x0.shape == (12,) and np.allclose(x0[:3], [0.1, 0.2, 0.5])
    s20 = np.arange(20, dtype=float)
    x = state20_to_x12(s20)
    assert x.shape == (12,) and np.allclose(x, np.r_[0:3, 7:10, 10:13, 13:16])
    Ag, Bg = build_linear_model(DT, preset="gym")
    assert np.allclose(A, Ag) and not np.allclose(B, Bg)       # chỉ B (cánh tay, hệ số xoắn) đổi
    assert abs(u_dev_to_rpm(np.zeros(4))[0] - 14476) < 5       # RPM hover
    assert abs(u_dev_to_rpm(np.full(4, 10.0))[0] - 21714) < 5  # cắt ở RPM tối đa
    assert U_DEV_MIN < 0 < U_DEV_MAX
    print("[OK] 2.1 mô hình: A(12x12), B(12x4), x0(12), đổi 20->12 chiều, preset team/gym, RPM hover/tối đa")


def check_reference():
    ref = Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)
    W = ref.window(3.0, H, DT)
    assert W.shape == (H + 1, 12), W.shape
    # vận tốc phải khớp đạo hàm số của vị trí
    num_v = (W[1:, 0:3] - W[:-1, 0:3]) / DT
    assert np.abs(num_v - 0.5 * (W[1:, 6:9] + W[:-1, 6:9])).max() < 1e-3
    assert np.allclose(ref.state(0.0)[6:9], 0.0)               # khởi động từ vận tốc 0
    assert np.abs(ref.state(0.0)[0:3] - [0, 0, 0.5]).max() < 1e-12
    pts = ref.path_points(400)
    assert abs(pts[:, 0].max() - 0.5) < 1e-2 and abs(pts[:, 1].max() - 0.25) < 1e-2
    assert PointReference((1, 2, 3)).window(0, H, DT).shape == (H + 1, 12)
    print("[OK] 2.2 quỹ đạo số 8: xref (H+1)x12 = %s, vận tốc nhất quán, điểm vẽ đường bay OK"
          % (W.shape,))


def check_bounds_logic():
    half = np.array([0.05, 0.15, 0.3])
    # drone ở phía -x của vật cản đặt tại x=0.5 -> biên trên theo trục x
    b = separating_bound((0.5, 0, 0.5), half, (0.0, 0.0, 0.5), margin=0.1)
    assert b[0] == 0 and b[1] == "upper" and abs(b[2] - (0.5 - 0.05 - 0.1)) < 1e-12, b
    # drone ở phía +x -> biên dưới
    b = separating_bound((0.5, 0, 0.5), half, (1.0, 0.0, 0.5), margin=0.1)
    assert b[0] == 0 and b[1] == "lower" and abs(b[2] - (0.5 + 0.05 + 0.1)) < 1e-12, b
    # drone ở phía +y của vật cản -> trục y
    b = separating_bound((0.5, 0, 0.5), half, (0.5, 0.9, 0.5), margin=0.1)
    assert b[0] == 1 and b[1] == "lower", b
    # drone nằm trong vùng đệm -> None
    assert separating_bound((0.5, 0, 0.5), half, (0.5, 0.0, 0.5), margin=0.1) is None
    print("[OK] 2.3 logic biên vật cản: trục/phía chọn đúng, trả None khi ở trong vùng đệm")


def _make_solver():
    A, B = build_linear_model(DT)
    Q = np.diag([100, 100, 100, 4, 4, 400, 4, 4, 4, 2, 2, 4]).astype(float)
    R = 4.0 * np.eye(4)
    scale_x = np.array([1e-2] * 3 + [0.5] * 3 + [0.3] * 3 + [20.0] * 3)
    S = AccelMPCSolver(A, B, Q, R, H=H, rho=4096.0, scale_x=scale_x, scale_u=np.ones(4))
    return S, A, B


def _pred_x(S):
    """Vị trí x dự đoán (đơn vị vật lý) tại k = 1..H."""
    Mw = S.M @ S.w
    return np.array([Mw[S.nd + k * S.nxu] for k in range(1, S.H + 1)])


def check_obstacle_is_active():
    """Kịch bản vật lý giống test của Thành viên A: biên nằm giữa vị trí bị ép (k=3) và
    đỉnh của nghiệm tự do, nên ràng buộc thật sự active. Dùng update_solver_bounds."""
    v0, margin = 0.5, 0.05
    half = np.array([0.05, 0.3, 0.3])
    x0 = np.zeros(12); x0[6] = v0
    xref = np.zeros(12)

    S0, _, _ = _make_solver()
    S0.reset_bounds()
    S0.solve(x0, xref, max_iter=40000, tol=1e-5)
    px_free = _pred_x(S0)
    forced = 3 * DT * v0
    excess = px_free.max() - forced
    assert excess > 0
    bound = forced + 0.9 * excess      # biên nằm trong khoảng (vị trí bị ép, đỉnh tự do)

    S, _, _ = _make_solver()
    obs_pos = np.array([bound + half[0] + margin, 0.0, 0.0])    # => biên x đúng bằng `bound`
    applied, inside = update_solver_bounds(S, x0[0:3], [(obs_pos, half)], margin)
    assert applied == [(0, "upper", bound)] or (applied[0][0] == 0 and abs(applied[0][2] - bound) < 1e-12), applied
    _, info = S.solve(x0, xref, max_iter=40000, tol=1e-5)
    px = _pred_x(S)
    print("    biên = %.3f mm | đỉnh nghiệm tự do = %.3f mm | đỉnh có vật cản = %.3f mm | "
          "ADMM %d vòng" % (1e3 * bound, 1e3 * px_free.max(), 1e3 * px.max(), info["iters"]))
    assert px_free.max() > bound * 1.05, "ràng buộc không active: kịch bản vô nghĩa"
    assert px.max() <= bound * (1 + 1e-2), "vị trí dự đoán vượt biên vật cản quá 1 %"
    print("[OK] biên vật cản qua update_solver_bounds chặn được nghiệm (vượt biên %.2f %%)"
          % (100 * (px.max() / bound - 1)))
    if info["iters"] >= 40000:
        print("[CẢNH BÁO] ADMM CHƯA hội tụ (hết 40000 vòng, tol=1e-5) khi biên vật cản active. "
              "Biên chỉ được tôn trọng gần đúng - báo Thành viên A.")


def check_loop_smoke():
    """Chạy thử vòng điều khiển (API đúng, không lỗi) trên mô hình tuyến tính, 100 bước."""
    S, A, B = _make_solver()
    ref = Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)
    obs = (np.array([0.5, 0.0, 0.5]), np.array([0.05, 0.3, 0.3]))
    x = ref.state(0.0)
    times = []
    for i in range(100):
        update_solver_bounds(S, x[0:3], [obs], 0.05)
        t0 = time.perf_counter()
        u0, info = S.solve(x, ref.window(i * DT, H, DT), max_iter=200, tol=None)
        times.append(time.perf_counter() - t0)
        assert u0.shape == (4,) and np.all(np.isfinite(u0))
        x = A @ x + B @ u0
        S.warm_start_shift()
    print("    100 bước x 200 vòng ADMM: TB %.1f ms mỗi bước (%.3f ms mỗi vòng)"
          % (1e3 * np.mean(times), 1e3 * np.mean(times) / 200))
    print("[OK] vòng điều khiển chạy thông (cập nhật biên -> solve -> warm-start)")


if __name__ == "__main__":
    check_model()
    check_reference()
    check_bounds_logic()
    check_obstacle_is_active()
    check_loop_smoke()
    print("TẤT CẢ KIỂM TRA ĐẠT")
