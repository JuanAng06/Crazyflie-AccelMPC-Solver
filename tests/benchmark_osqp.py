"""Tuần 4 - Phần 2: Đối chiếu AccelMPCSolver với OSQP (CPU, Float64).

Đặt file này tại:  CSHTT_Project/tests/benchmark_osqp.py   (cạnh benchmark_solver.py)
Chạy (Anaconda Prompt, env drones, đứng ở thư mục gốc dự án):
    python tests\\benchmark_osqp.py --quick       # ~3 s mô phỏng, vài phút
    python tests\\benchmark_osqp.py               # 10 s mô phỏng, lâu hơn

Ý tưởng: tại MỖI bước điều khiển, cùng một QP (P, c, M, l, u của chính bộ giải) được giải bởi
  - "bóng" OSQP độ chính xác rất cao (eps nhỏ + polish)  -> nghiệm tham chiếu w*, u0*, f*
  - bộ điều khiển đang chạy (ADMM với K vòng cố định, hoặc OSQP với eps nới lỏng).
Từ đó đo: sai số u0, độ dưới tối ưu f(w) - f*, vi phạm ràng buộc, thời gian giải và chất lượng bám quỹ đạo.

Không sửa src/admm_mpc_solver.py. Cần: pip install osqp (đã có trong env drones).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import functools
import gc
import sys
import time
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("simulation", "src", ""):
    sys.path.insert(0, os.path.join(ROOT, sub))

OBS_CENTER = np.array([0.5, 0.0, 0.5])
OBS_HALF = np.array([0.05, 0.15, 0.3])
BIG = 1e30
print = functools.partial(print, flush=True)      # in ra ngay, không bị đệm


# ----------------------------------------------------------------------------------------------
# Các hàm lõi (không phụ thuộc module dự án) - dùng chung với phần kiểm thử
# ----------------------------------------------------------------------------------------------
class OsqpQP:
    """Bọc OSQP cho đúng QP mà `solver` đang giữ (solver.P, solver.M, solver.c, solver.l, solver.u)."""

    def __init__(self, eps_abs, eps_rel, max_iter=5000, polish=False, time_limit=0.05):
        self.kw = dict(eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter, polish=polish,
                       eps_prim_inf=1e-12, eps_dual_inf=1e-12, adaptive_rho=False,
                       verbose=False, warm_start=True)
        if time_limit and time_limit > 0:
            self.kw["time_limit"] = time_limit
        self.prob = None

    def solve(self, solver):
        import osqp
        q = np.asarray(solver.c, float).copy()
        l = np.clip(solver.l, -BIG, BIG)
        u = np.clip(solver.u, -BIG, BIG)
        if self.prob is None:
            self.prob = osqp.OSQP()
            self.prob.setup(sp.triu(solver.P, format="csc"), q, sp.csc_matrix(solver.M), l, u, **self.kw)
        else:
            self.prob.update(q=q, l=l, u=u)
        res = self.prob.solve()
        return np.array(res.x, float), res.info


def u0_from_w(solver, w):
    """w là nghiệm trong biến đã scale; u0 vật lý = w[nx:nx+nu] * D (D = 1/D_inv_diag)."""
    nx, nu = solver.nx, solver.nu
    return w[nx:nx + nu] / solver.D_inv_diag[nx:nx + nu]


def qp_obj(solver, w):
    return float(0.5 * w @ (solver.P @ w) + solver.c @ w)


def qp_violation(solver, w):
    v = solver.M @ w
    return float(max(0.0, np.max(solver.l - v), np.max(v - solver.u)))


def stat(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p99=float(np.percentile(a, 99)),
                max=float(a.max()))


# ----------------------------------------------------------------------------------------------
# Phần phụ thuộc dự án
# ----------------------------------------------------------------------------------------------
def load_project():
    from simulation.crazyflie_model import build_linear_model, U_DEV_MIN, U_DEV_MAX
    from simulation.reference import Figure8Reference
    from simulation.obstacles import update_solver_bounds, shape_reference
    from simulation.solver_ext import make_solver
    return SimpleNamespace(build_linear_model=build_linear_model, U_DEV_MIN=U_DEV_MIN, U_DEV_MAX=U_DEV_MAX,
                           Figure8Reference=Figure8Reference, update_solver_bounds=update_solver_bounds,
                           shape_reference=shape_reference, make_solver=make_solver)


def self_check(PR, A, B, H, dt):
    """Kiểm tra bộ giải và OSQP đang giải CÙNG một QP: ADMM chạy rất nhiều vòng phải ra gần nghiệm OSQP."""
    solver = PR.make_solver(A, B, H, dare_terminal=True)
    ref = PR.Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)
    x = np.asarray(ref.state(0.0), float).copy()
    solver.reset_bounds()
    W = ref.window(0.0, H, dt)
    solver.solve(x, W, max_iter=1, tol=None)                  # dựng c, l, u
    print("[self-check] đang giải OSQP tham chiếu ...")
    t0 = time.perf_counter()
    w_opt, info = OsqpQP(1e-8, 1e-8, 50000, True, time_limit=20.0).solve(solver)
    print("[self-check] OSQP xong sau %.2f s (%d vòng, trạng thái: %s)"
          % (time.perf_counter() - t0, info.iter, info.status))
    u_opt = u0_from_w(solver, w_opt)
    dus = []
    for iters in (5000, 50000):
        solver.y[:] = 0.0
        solver.s[:] = 0.0
        u0, _ = solver.solve(x, W, max_iter=iters, tol=None)
        w_admm = solver.w.copy()
        dus.append(float(np.abs(u0 - u_opt).max()))
        print("[self-check] ADMM %5d vòng vs OSQP(%s): |Δu0|max = %.2e | f(w_admm)-f* = %.2e | vi phạm ADMM %.2e"
              % (iters, info.status, dus[-1], qp_obj(solver, w_admm) - qp_obj(solver, w_opt),
                 qp_violation(solver, w_admm)))
    if dus[-1] > 1e-2 and dus[-1] > 0.5 * dus[0]:
        print("   CẢNH BÁO: |Δu0| không giảm khi tăng số vòng -> có thể solver.P/M khác hệ đã phân rã "
              "(kiểm tra solver_ext), hoặc ADMM hội tụ rất chậm ở rho lớn.")
    solver.y[:] = 0.0
    solver.s[:] = 0.0


def run_cl(PR, A, B, H, dt, ctrl, duration, margin, near_radius, ref_eps, shadow_every=5, n_skip=50):
    """Vòng kín trên plant tuyến tính. ctrl = ("admm", K) hoặc ("osqp", eps)."""
    kind, param = ctrl
    solver = PR.make_solver(A, B, H, dare_terminal=True)
    ref = PR.Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)
    x = np.asarray(ref.state(0.0), float).copy()
    shadow = OsqpQP(1e-7, 1e-7, 5000, False, time_limit=0.05)
    shadow_bad = 0
    osqp_ctrl = OsqpQP(param, param, 1000, False, time_limit=0.05) if kind == "osqp" else None
    urange = float(np.max(np.asarray(PR.U_DEV_MAX) - np.asarray(PR.U_DEV_MIN)))

    n = int(round(duration / dt))
    times, osqp_rt, osqp_it, du_l, gap_l, viol_l, bad = [], [], [], [], [], [], 0
    err = np.zeros(n)
    near = np.zeros(n, bool)
    mingap, tilt, diverged, done = 1e9, 0.0, False, 0

    T0 = time.time()
    tag = "%s %g" % (kind, param)
    gc.collect()
    gc.disable()
    try:
        for i in range(n):
            if i % 250 == 0:
                print("      [%s] bước %d/%d, đã chạy %.0f s" % (tag, i, n, time.time() - T0))
            t = i * dt
            W_ref = ref.window(t, H, dt)
            PR.update_solver_bounds(solver, x[0:3], [(OBS_CENTER, OBS_HALF)], margin,
                                    drone_vel=x[6:9], dt=dt, a_brake=0.5, a_floor=0.5)
            W_ref = PR.shape_reference(W_ref, x[0:3], [(OBS_CENTER, OBS_HALF)], margin)
            mingap = min(mingap, float(np.max(np.abs(x[0:3] - OBS_CENTER) - OBS_HALF)))

            if kind == "admm":
                t0 = time.perf_counter_ns()
                u0, _ = solver.solve(x, W_ref, max_iter=param, tol=None)
                t1 = time.perf_counter_ns()
                w_c = solver.w.copy()
            else:
                solver.solve(x, W_ref, max_iter=1, tol=None)        # chỉ để dựng c, l, u (không tính giờ)
                t0 = time.perf_counter_ns()
                w_c, oi = osqp_ctrl.solve(solver)
                t1 = time.perf_counter_ns()
                u0 = u0_from_w(solver, w_c)
                osqp_rt.append(1e3 * float(oi.run_time))
                osqp_it.append(int(oi.iter))
                bad += int(oi.status != "solved")
            times.append(t1 - t0)

            # --- nghiệm tham chiếu trên cùng QP (không tính giờ); chỉ làm mỗi shadow_every bước cho nhanh
            if i % shadow_every == 0:
                def shadow_ok(w_, si_):
                    return (si_.status in ("solved", "solved inaccurate") and np.isfinite(w_).all()
                            and np.abs(w_).max() < 1e6 and qp_violation(solver, w_) < 1e-3)
                w_opt, si = shadow.solve(solver)
                good = shadow_ok(w_opt, si)
                if not good:                                  # dựng lại OSQP từ đầu (bỏ warm-start hỏng) rồi thử lại
                    shadow.prob = None
                    w_opt, si = shadow.solve(solver)
                    good = shadow_ok(w_opt, si)
                    if not good:
                        shadow.prob = None
                if good:
                    du_l.append(float(np.abs(u0 - u0_from_w(solver, w_opt)).max()) / urange)
                    gap_l.append(qp_obj(solver, w_c) - qp_obj(solver, w_opt))
                    viol_l.append(qp_violation(solver, w_c))
                else:
                    shadow_bad += 1

            u0 = np.clip(u0, PR.U_DEV_MIN, PR.U_DEV_MAX)
            x = A @ x + B @ u0
            if kind == "admm":
                solver.warm_start_shift()

            xr = ref.state(t + dt)
            err[i] = np.linalg.norm(x[0:2] - xr[0:2])
            near[i] = np.linalg.norm(x[0:2] - OBS_CENTER[0:2]) < near_radius
            tilt = max(tilt, float(np.abs(x[3:5]).max()))
            done = i + 1
            if (not np.isfinite(x).all()) or abs(x[3]) > 1.2 or abs(x[4]) > 1.2:
                diverged = True
                break
    finally:
        gc.enable()

    err, near = err[:done], near[:done]
    sk = n_skip if done > n_skip + 10 else 0
    ts = stat(np.array(times[sk:]) / 1e6)
    if not du_l:                                   # không có nghiệm tham chiếu hợp lệ nào
        du_l, gap_l, viol_l = [np.nan], [np.nan], [np.nan]
    dus, gaps, viols = stat(du_l), stat(np.abs(gap_l)), stat(viol_l)
    res = dict(steps=done, diverged=diverged,
               mean_ms=ts["mean"], median_ms=ts["median"], p99_ms=ts["p99"], max_ms=ts["max"],
               du0_mean_pct=100 * dus["mean"], du0_max_pct=100 * dus["max"],
               subopt_median=gaps["median"], subopt_max=gaps["max"],
               viol_median=viols["median"], viol_max=viols["max"],
               rms_far_mm=1e3 * np.sqrt(np.mean(err[~near] ** 2)) if (~near).any() else np.nan,
               rms_near_mm=1e3 * np.sqrt(np.mean(err[near] ** 2)) if near.any() else np.nan,
               mingap_mm=1e3 * mingap, tilt_deg=float(np.degrees(tilt)),
               osqp_iter_mean=float(np.mean(osqp_it)) if osqp_it else np.nan,
               osqp_runtime_mean_ms=float(np.mean(osqp_rt)) if osqp_rt else np.nan,
               osqp_not_solved=bad, shadow_bad=shadow_bad, shadow_total=len(range(0, done, shadow_every)))
    return res


def describe_qp(solver, w, x, A, top=6):
    """In các hàng ràng buộc bị nghiệm ADMM vi phạm nhiều nhất, và các biên hộp hữu hạn tại k=1."""
    nx, nxu, H, nd = solver.nx, solver.nxu, solver.H, solver.nd
    v = solver.M @ w
    viol = np.maximum(0.0, np.maximum(solver.l - v, v - solver.u))

    def where(r):
        if r < nd:
            return "động lực k=%d, x[%d]" % (r // nx, r % nx)
        j = r - nd
        if j >= H * nxu:
            return "hộp k=%d, x[%d]" % (H, j - H * nxu)
        k, c = divmod(j, nxu)
        return "hộp k=%d, %s[%d]" % (k, "x" if c < nx else "u", c if c < nx else c - nx)

    print("     hàng bị ADMM vi phạm nhiều nhất:")
    for r in np.argsort(-viol)[:top]:
        print("        %-22s l=%-10.4g u=%-10.4g v=%-10.4g vi phạm %.2e" % (where(r), solver.l[r], solver.u[r], v[r], viol[r]))
    x_free = A @ x
    print("     biên hộp hữu hạn tại k=1 (x_free = A@x: giá trị nếu u=0):")
    for c in range(nx):
        j = nd + nxu + c
        lo, up = solver.l[j], solver.u[j]
        if np.isfinite(lo) or np.isfinite(up):
            print("        x[%d]: l=%-10.5g u=%-10.5g x_free=%.5g" % (c, lo, up, x_free[c]))


def debug_run(PR, A, B, H, dt, margin, K=50, n=60):
    """Chạy ngắn, in chẩn đoán: OSQP (đường update, như trong benchmark) vs OSQP mới hoàn toàn vs ADMM."""
    solver = PR.make_solver(A, B, H, dare_terminal=True)
    ref = PR.Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)
    x = np.asarray(ref.state(0.0), float).copy()
    persistent = OsqpQP(1e-6, 1e-6, 20000, False, time_limit=1.0)
    picks = {0, 1, 2, 10, 30, n - 1}
    detail = {30, n - 1}
    for i in range(n):
        t = i * dt
        W_ref = ref.window(t, H, dt)
        PR.update_solver_bounds(solver, x[0:3], [(OBS_CENTER, OBS_HALF)], margin,
                                drone_vel=x[6:9], dt=dt, a_brake=0.5, a_floor=0.5)
        W_ref = PR.shape_reference(W_ref, x[0:3], [(OBS_CENTER, OBS_HALF)], margin)
        u0, _ = solver.solve(x, W_ref, max_iter=K, tol=None)
        w_c = solver.w.copy()
        w_p, ip = persistent.solve(solver)
        if i in picks:
            w_f, if_ = OsqpQP(1e-6, 1e-6, 20000, False, time_limit=1.0).solve(solver)
            print("  bước %d | hàng l>u: %d | biên hữu hạn l/u: %d/%d | ADMM: vi phạm %.1e f=%.3e"
                  % (i, int(np.sum(solver.l > solver.u + 1e-12)), int(np.isfinite(solver.l).sum()),
                     int(np.isfinite(solver.u).sum()), qp_violation(solver, w_c), qp_obj(solver, w_c)))
            for name, w, inf in (("OSQP-update", w_p, ip), ("OSQP-mới", w_f, if_)):
                print("       %-11s %-18s vòng %-6d max|w|=%.1e vi phạm %.1e f=%.3e"
                      % (name, inf.status, inf.iter, np.abs(w).max(), qp_violation(solver, w), qp_obj(solver, w)))
            if i in detail:
                print("     x0: vị trí %s, vận tốc %s" % (np.round(x[0:3], 4), np.round(x[6:9], 4)))
                describe_qp(solver, w_c, x, A)
        u0 = np.clip(u0, PR.U_DEV_MIN, PR.U_DEV_MAX)
        x = A @ x + B @ u0
        solver.warm_start_shift()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--H", type=int, default=31)
    ap.add_argument("--Ks", default="10,20,50,100", help="ngân sách vòng lặp ADMM")
    ap.add_argument("--eps", default="1e-3,1e-5", help="dung sai OSQP khi làm bộ điều khiển")
    ap.add_argument("--ref_eps", type=float, default=1e-6, help="dung sai OSQP 'bóng' tham chiếu")
    ap.add_argument("--shadow_every", type=int, default=5, help="chỉ giải OSQP tham chiếu mỗi N bước")
    ap.add_argument("--skip_check", action="store_true", help="bỏ bước self-check")
    ap.add_argument("--debug", action="store_true", help="chạy ngắn, in chẩn đoán OSQP rồi thoát")
    ap.add_argument("--debug_n", type=int, default=60, help="số bước của chế độ --debug")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--near", type=float, default=0.35)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)
        except Exception:                                                        # noqa: BLE001
            pass

    duration = 3.0 if a.quick else a.duration
    os.makedirs(a.out, exist_ok=True)
    PR = load_project()
    A, B = PR.build_linear_model(a.dt, "gym")

    if not a.skip_check:
        self_check(PR, A, B, a.H, a.dt)

    if a.debug:
        print("\n== DEBUG: %d bước đầu ==" % a.debug_n)
        debug_run(PR, A, B, a.H, a.dt, a.margin, n=a.debug_n)
        return

    configs = [("admm", int(k)) for k in a.Ks.split(",")] + [("osqp", float(e)) for e in a.eps.split(",")]
    rows = []
    print("\n== Đối chiếu OSQP: H=%d, dt=%g, %.0f s mô phỏng ==" % (a.H, a.dt, duration))
    for ctrl in configs:
        r = run_cl(PR, A, B, a.H, a.dt, ctrl, duration, a.margin, a.near, a.ref_eps, a.shadow_every)
        name = "ADMM K=%d" % ctrl[1] if ctrl[0] == "admm" else "OSQP eps=%g" % ctrl[1]
        rows.append(dict(controller=ctrl[0], param=ctrl[1], H=a.H, dt=a.dt, **r))
        print("%-13s | %.2f ms (median %.2f, p99 %.2f) | Δu0 TB %.2f%% / max %.2f%% | f-f* median %.1e max %.1e | "
              "vi phạm max %.1e | RMS xa %.1f mm | gap min %.1f mm%s" % (
                  name, r["mean_ms"], r["median_ms"], r["p99_ms"], r["du0_mean_pct"], r["du0_max_pct"],
                  r["subopt_median"], r["subopt_max"], r["viol_max"], r["rms_far_mm"], r["mingap_mm"],
                  (" | OSQP %.0f vòng, run_time %.2f ms, %d bước không 'solved'" % (
                      r["osqp_iter_mean"], r["osqp_runtime_mean_ms"], r["osqp_not_solved"])
                   if ctrl[0] == "osqp" else "")), flush=True)
        print("      nghiệm tham chiếu không hợp lệ: %d/%d lần" % (r["shadow_bad"], r["shadow_total"]))

    path = os.path.join(a.out, "benchmark_osqp.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nĐã ghi:", path)


if __name__ == "__main__":
    main()