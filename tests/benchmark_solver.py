"""Tuần 4 - Phần 1: Benchmark AccelMPCSolver (CPU, Float64).

Đặt file này tại:  CSHTT_Project/tests/benchmark_solver.py
Chạy (Anaconda Prompt, env drones, đứng ở thư mục gốc dự án):
    python tests\\benchmark_solver.py --quick        # chạy thử 3 s mô phỏng (~1 phút)
    python tests\\benchmark_solver.py                # chạy đầy đủ 10 s (vài phút)

Thí nghiệm:
  A) Ngân sách cố định (tol=None), K in {10,20,50,100}, warm-start vs cold-start.
  B) Dừng theo dung sai (tol), trần max_iter: so số vòng lặp trung bình warm vs cold.

Vòng kín dùng plant TUYẾN TÍNH x+ = A x + B u (cùng mô hình với bộ giải), nên chạy nhanh,
lặp lại được, không cần PyBullet. RMSE ở đây chỉ phản ánh chất lượng bộ giải
(không có sai lệch mô hình) -> khác với RMSE trong PyBullet của Thành viên B.

Không sửa src/admm_mpc_solver.py. Chỉ dùng: make_solver, solver.solve, solver.warm_start_shift,
solver.y, solver.s (y, s là hai biến mang trạng thái warm-start).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")      # phải đặt TRƯỚC khi import numpy
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import gc
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("simulation", "src", ""):
    sys.path.insert(0, os.path.join(ROOT, sub))

from crazyflie_model import build_linear_model, U_DEV_MIN, U_DEV_MAX      # noqa: E402
from reference import Figure8Reference                                     # noqa: E402
from obstacles import update_solver_bounds, shape_reference                # noqa: E402
from solver_ext import make_solver                                         # noqa: E402

# Cùng cấu hình vật cản với run_simulation.py (mặc định --obstacle_x 0.5, --margin 0.10)
OBS_CENTER = np.array([0.5, 0.0, 0.5])
OBS_HALF = np.array([0.05, 0.15, 0.3])


def stats_ms(t_ns):
    t = np.asarray(t_ns, float) / 1e6
    return dict(mean=t.mean(), std=t.std(), median=np.median(t),
                p99=np.percentile(t, 99), max=t.max())


def run_closed_loop(A, B, H, dt, max_iter, tol, warm, duration, margin, near_radius, n_skip=50):
    solver = make_solver(A, B, H, dare_terminal=True)
    ref = Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)
    x = np.asarray(ref.state(0.0), float).copy()
    assert x.shape == (A.shape[0],), "ref.state() không trả về vector %d chiều: %s" % (A.shape[0], x.shape)

    n = int(round(duration / dt))
    times, iters, conv, rp, rd = [], [], [], [], []
    err = np.zeros(n)
    near = np.zeros(n, bool)
    mingap, tilt, diverged, done = 1e9, 0.0, False, 0

    gc.collect()
    gc.disable()
    try:
        for i in range(n):
            t = i * dt
            W_ref = ref.window(t, H, dt)
            update_solver_bounds(solver, x[0:3], [(OBS_CENTER, OBS_HALF)], margin,
                                 drone_vel=x[6:9], dt=dt, a_brake=0.5, a_floor=0.5)
            W_ref = shape_reference(W_ref, x[0:3], [(OBS_CENTER, OBS_HALF)], margin)
            mingap = min(mingap, float(np.max(np.abs(x[0:3] - OBS_CENTER) - OBS_HALF)))

            if not warm:                      # cold-start: xoá trạng thái kép/phụ trợ
                solver.y[:] = 0.0
                solver.s[:] = 0.0

            t0 = time.perf_counter_ns()
            u0, info = solver.solve(x, W_ref, max_iter=max_iter, tol=tol)
            t1 = time.perf_counter_ns()
            times.append(t1 - t0)
            iters.append(info["iters"])
            conv.append(tol is not None and info["iters"] < max_iter)
            rp.append(info["r_prim"])
            rd.append(info["r_dual"])

            u0 = np.clip(u0, U_DEV_MIN, U_DEV_MAX)
            x = A @ x + B @ u0
            if warm:
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
    s = stats_ms(times[n_skip:] if done > n_skip + 10 else times)
    far_rms = 1e3 * np.sqrt(np.mean(err[~near] ** 2)) if (~near).any() else np.nan
    near_rms = 1e3 * np.sqrt(np.mean(err[near] ** 2)) if near.any() else np.nan
    return dict(steps=done, diverged=diverged,
                mean_ms=s["mean"], std_ms=s["std"], median_ms=s["median"],
                p99_ms=s["p99"], max_ms=s["max"],
                mean_iters=float(np.mean(iters)),
                pct_converged=100.0 * float(np.mean(conv)) if tol is not None else np.nan,
                mean_r_prim=float(np.mean(rp)) if tol is not None else np.nan,
                mean_r_dual=float(np.mean(rd)) if tol is not None else np.nan,
                rms_far_mm=far_rms, rms_near_mm=near_rms,
                max_err_mm=1e3 * float(err.max()), mingap_mm=1e3 * mingap,
                tilt_deg=float(np.degrees(tilt))), np.array(times, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--H", type=int, default=31)
    ap.add_argument("--Ks", default="10,20,50,100")
    ap.add_argument("--duration", type=float, default=10.0, help="thời gian mô phỏng [s]")
    ap.add_argument("--quick", action="store_true", help="chỉ mô phỏng 3 s để chạy thử")
    ap.add_argument("--tols", default="100,10,1",
                    help="các dung sai cho thí nghiệm B (r_dual đã nhân rho=4096 nên tol nhỏ khó đạt)")
    ap.add_argument("--no_priority", action="store_true", help="không nâng mức ưu tiên tiến trình (Windows)")
    ap.add_argument("--tag", default="", help="hậu tố tên file raw (để so sánh giữa các lần chạy)")
    ap.add_argument("--cap", type=int, default=200, help="trần số vòng lặp cho thí nghiệm B")
    ap.add_argument("--skip_tol", action="store_true", help="bỏ thí nghiệm B")
    ap.add_argument("--skip_A", action="store_true", help="bỏ thí nghiệm A (chỉ chạy B, ghi ra benchmark_tol.csv)")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--near", type=float, default=0.35, help="bán kính [m] coi là 'đoạn gần vật cản'")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()

    if os.name == "nt" and not a.no_priority:       # giảm nhiễu hệ điều hành: ưu tiên cao cho tiến trình này
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            ok_p = k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)   # HIGH_PRIORITY_CLASS
            print("Mức ưu tiên cao:", "bật" if ok_p else "không bật được (lỗi %d)" % ctypes.get_last_error())
        except Exception as e:                                                  # noqa: BLE001
            print("Không nâng được mức ưu tiên:", e)

    duration = 3.0 if a.quick else a.duration
    Ks = [int(k) for k in a.Ks.split(",")]
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    A, B = build_linear_model(a.dt, "gym")

    rows = []

    def do(exp, warm, max_iter, tol):
        tag = "%s %-4s K=%-3d" % (exp, "warm" if warm else "cold", max_iter)
        res, times = run_closed_loop(A, B, a.H, a.dt, max_iter, tol, warm, duration, a.margin, a.near)
        row = dict(exp=exp, mode="warm" if warm else "cold", H=a.H, dt=a.dt,
                   max_iter=max_iter, tol=("" if tol is None else tol), **res)
        rows.append(row)
        suffix = ("" if tol is None else "_tol%g" % tol) + a.tag
        np.save(os.path.join(a.out, "raw", "times_%s_%s_K%d%s.npy" % (exp, row["mode"], max_iter, suffix)), times)
        print("%s | mean %.2f / median %.2f ms (std %.2f, p99 %.2f, max %.2f) | iters %.1f | RMS xa %.1f / gần %.1f mm | "
              "gap min %.1f mm | tilt %.1f° %s" % (
                  tag, res["mean_ms"], res["median_ms"], res["std_ms"], res["p99_ms"], res["max_ms"], res["mean_iters"],
                  res["rms_far_mm"], res["rms_near_mm"], res["mingap_mm"], res["tilt_deg"],
                  "| MẤT ỔN ĐỊNH sau %d bước" % res["steps"] if res["diverged"] else ""), flush=True)
        if tol is not None:
            print("      residual cuối bước: r_prim TB %.2e | r_dual TB %.2e | dừng sớm ở %.1f%% số bước" % (
                res["mean_r_prim"], res["mean_r_dual"], res["pct_converged"]), flush=True)

    if not a.skip_A:
        print("== Thí nghiệm A: ngân sách cố định (tol=None), H=%d, dt=%g, %.0f s mô phỏng ==" % (a.H, a.dt, duration))
        for warm in (True, False):
            for K in Ks:
                do("A", warm, K, None)

    if not a.skip_tol:
        for tol in [float(v) for v in a.tols.split(",")]:
            print("\n== Thí nghiệm B: dừng theo tol=%g, trần %d vòng ==" % (tol, a.cap))
            for warm in (True, False):
                do("B", warm, a.cap, tol)

    # --- Hồi quy thời gian = a + b*K (warm, thí nghiệm A): tách overhead mỗi lần gọi và chi phí mỗi vòng ---
    # Dùng MEDIAN (bền vững với các cú giật hiếm gặp); in thêm bản dùng mean để đối chiếu.
    ok = [r for r in rows if r["exp"] == "A" and r["mode"] == "warm" and not r["diverged"]]
    if len(ok) >= 2:
        k = np.array([r["max_iter"] for r in ok], float)
        for name, key in (("median", "median_ms"), ("mean", "mean_ms")):
            m = np.array([r[key] for r in ok])
            b, a0 = np.polyfit(k, m, 1)
            print("\nHồi quy (warm, theo %s): thời gian ≈ %.3f ms + K × %.1f µs" % (name, a0, 1e3 * b))
            print("  -> %.1f µs/vòng = %.2f µs/(vòng·bước dự báo) [bài báo: FPGA 3.33-3.69, MCU 52.3]"
                  % (1e3 * b, 1e3 * b / a.H))
            print("  -> overhead mỗi lần gọi solve: %.3f ms | K tối đa trong chu kỳ %.1f ms: %d"
                  % (a0, 1e3 * a.dt, max(0, int((1e3 * a.dt - a0) / b))))

    path = os.path.join(a.out, "benchmark_tol.csv" if a.skip_A else "benchmark_warm_cold.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nĐã ghi:", path)


if __name__ == "__main__":
    main()