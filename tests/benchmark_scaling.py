"""Tuần 4 - Phần 3: Khả năng mở rộng theo chân trời dự báo H (CPU, Float64).

Đặt file này tại:  CSHTT_Project/tests/benchmark_scaling.py
Chạy (Anaconda Prompt, env drones, đứng ở thư mục gốc dự án):
    python tests\\benchmark_scaling.py --quick     # ít bước hơn, chạy thử
    python tests\\benchmark_scaling.py             # đầy đủ

Với mỗi H in {15, 31, 63, 127} đo:
  - N_var, độ rộng băng h, dung lượng L (MB)
  - OFFLINE: tổng thời gian make_solver (dựng P, M, phân rã) và riêng bước cholesky_banded
  - ONLINE: thời gian giải (median, P99) ở K=10 và K=50, warm-start, dọc quỹ đạo số 8 (không cần vòng kín)
  - Suy ra: µs/vòng, µs/(vòng·bước dự báo), K tối đa trong chu kỳ 2 ms và 1 ms, độ dốc log-log theo N_var
Kỳ vọng từ bài báo: h gần như không đổi, thời gian online mỗi vòng tăng ~tuyến tính theo N_var.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
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


def bench_online(solver, get_xW, n, K, n_skip=30):
    """Giải n bước liên tiếp với warm-start, trả về mảng thời gian [ms] (đã bỏ n_skip bước đầu)."""
    solver.reset_bounds()
    solver.y[:] = 0.0
    solver.s[:] = 0.0
    times = []
    gc.collect()
    gc.disable()
    try:
        for i in range(n):
            x, W = get_xW(i)
            t0 = time.perf_counter_ns()
            solver.solve(x, W, max_iter=K, tol=None)
            t1 = time.perf_counter_ns()
            solver.warm_start_shift()
            times.append(t1 - t0)
    finally:
        gc.enable()
    return np.array(times[n_skip:], float) / 1e6


def median_time(fn, repeats):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--Hs", default="15,31,63,127")
    ap.add_argument("--K_lo", type=int, default=10)
    ap.add_argument("--K_hi", type=int, default=50)
    ap.add_argument("--n", type=int, default=400, help="số bước giải mỗi cấu hình")
    ap.add_argument("--repeats", type=int, default=3, help="số lần lặp đo offline")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    a = ap.parse_args()
    if a.quick:
        a.n, a.repeats = 150, 2

    if os.name == "nt":
        try:
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)
        except Exception:                                                        # noqa: BLE001
            pass

    from scipy.linalg import cholesky_banded
    from simulation.crazyflie_model import build_linear_model
    from simulation.reference import Figure8Reference
    from simulation.solver_ext import make_solver

    os.makedirs(a.out, exist_ok=True)
    A, B = build_linear_model(a.dt, "gym")
    ref = Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)

    rows = []
    print("H, dt=%g | N_var, h | offline: make_solver / cholesky [ms] | online K=%d / K=%d: median (p99) [ms] | "
          "µs/vòng, µs/(vòng·bước)" % (a.dt, a.K_lo, a.K_hi))
    for H in [int(v) for v in a.Hs.split(",")]:
        t_make = 1e3 * median_time(lambda: make_solver(A, B, H, dare_terminal=True), a.repeats)
        solver = make_solver(A, B, H, dare_terminal=True)
        t_chol = 1e3 * median_time(lambda: cholesky_banded(solver.K_banded, lower=True), a.repeats)

        def get_xW(i, H=H):
            t = 1.0 + i * a.dt
            return np.asarray(ref.state(t), float), ref.window(t, H, a.dt)

        lo = bench_online(solver, get_xW, a.n, a.K_lo)
        hi = bench_online(solver, get_xW, a.n, a.K_hi)
        m_lo, m_hi = float(np.median(lo)), float(np.median(hi))
        b = (m_hi - m_lo) / (a.K_hi - a.K_lo)            # ms mỗi vòng
        a0 = m_lo - a.K_lo * b                           # overhead mỗi lần gọi [ms]
        kmax = lambda period_ms: max(0, int((period_ms - a0) / b)) if b > 0 else 0   # noqa: E731
        row = dict(H=H, dt=a.dt, Nvar=int(solver.Nvar), bandwidth=int(solver.h),
                   L_MB=solver.L_banded.nbytes / 1e6,
                   offline_make_ms=t_make, offline_chol_ms=t_chol,
                   median_lo_ms=m_lo, p99_lo_ms=float(np.percentile(lo, 99)),
                   median_hi_ms=m_hi, p99_hi_ms=float(np.percentile(hi, 99)),
                   us_per_iter=1e3 * b, us_per_iter_step=1e3 * b / H, overhead_ms=a0,
                   Kmax_2ms=kmax(2.0), Kmax_1ms=kmax(1.0))
        rows.append(row)
        print("H=%-4d | N=%-5d h=%-3d L=%.2f MB | %8.1f / %8.1f | %6.2f (%.2f) / %6.2f (%.2f) | %.1f, %.2f | "
              "Kmax 2ms=%d, 1ms=%d" % (H, row["Nvar"], row["bandwidth"], row["L_MB"], t_make, t_chol,
                                       m_lo, row["p99_lo_ms"], m_hi, row["p99_hi_ms"],
                                       row["us_per_iter"], row["us_per_iter_step"],
                                       row["Kmax_2ms"], row["Kmax_1ms"]), flush=True)

    if len(rows) >= 2:
        n_var = np.array([r["Nvar"] for r in rows], float)
        for name, key in (("online K=%d" % a.K_hi, "median_hi_ms"), ("offline cholesky", "offline_chol_ms"),
                          ("offline make_solver", "offline_make_ms")):
            slope = np.polyfit(np.log(n_var), np.log([r[key] for r in rows]), 1)[0]
            print("Độ dốc log-log theo N_var (%s): %.2f  (1.0 = tuyến tính, 2.0 = bình phương)" % (name, slope))

    path = os.path.join(a.out, "benchmark_scaling.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nĐã ghi:", path)


if __name__ == "__main__":
    main()