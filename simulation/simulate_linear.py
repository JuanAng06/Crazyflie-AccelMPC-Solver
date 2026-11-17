"""Giai đoạn 3 - "song sinh số": vòng kín chỉ numpy (KHÔNG cần PyBullet).
Chạy:  python simulate_linear.py            (số 8 + vật cản, 10 s)
       python simulate_linear.py --no_obstacle
Kỳ vọng: không va chạm (khoảng cách nhỏ nhất > 70 mm), tilt nhỏ, in thống kê vòng lặp ADMM.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crazyflie_model import build_linear_model, U_DEV_MIN, U_DEV_MAX
from reference import Figure8Reference, TakeoffReference
from obstacles import Obstacle, update_solver_bounds, shape_reference
from solver_ext import make_solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--no_obstacle", action="store_true")
    ap.add_argument("--preset", default="gym")
    ap.add_argument("--takeoff", type=float, default=0.0)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    H = 31
    A, B = build_linear_model(a.dt, a.preset)
    S = make_solver(A, B, H)
    ref = Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)
    obs = Obstacle(shape="box", center=(0.5, 0.0, 0.5), half_size=(0.05, 0.15, 0.3))
    if a.takeoff > 0:
        ref = TakeoffReference(ref, a.takeoff, 0.05)
    x = ref.state(0.0)
    mingap, tilt, err, times, resid = 1e9, 0.0, [], [], []
    lt, lp, lr = [], [], []
    for i in range(int(a.T / a.dt)):
        t = i * a.dt
        W = ref.window(t, H, a.dt)
        if not a.no_obstacle:
            op = obs.position_at(t)
            update_solver_bounds(S, x[0:3], [(op, obs.half_size)], a.margin,
                                 drone_vel=x[6:9], dt=a.dt, a_brake=0.5, a_floor=0.5)
            W = shape_reference(W, x[0:3], [(op, obs.half_size)], a.margin)
            mingap = min(mingap, np.max(np.abs(x[0:3] - op) - obs.half_size))
        else:
            S.reset_bounds()
        t0 = time.perf_counter()
        u0, info = S.solve(x, W, max_iter=a.iters, tol=None)
        times.append(time.perf_counter() - t0)
        u0 = np.clip(u0, U_DEV_MIN, U_DEV_MAX)
        x = A @ x + B @ u0
        S.warm_start_shift()
        tilt = max(tilt, np.abs(x[3:5]).max())
        err.append(np.linalg.norm(x[0:2] - ref.state(t)[0:2]))
        lt.append(t + a.dt); lp.append(x[0:3].copy()); lr.append(ref.state(t + a.dt)[0:3])
    if a.save:
        np.savez(a.save, t=np.array(lt), pos=np.array(lp), ref=np.array(lr), obs_center=obs.center,
                 obs_half=obs.half_size, margin=a.margin, dt=a.dt, takeoff=a.takeoff)
    print("%d bước x %d vòng ADMM cố định | TB %.2f ms/bước" % (len(times), a.iters, 1e3 * np.mean(times)))
    print("sai số bám quỹ đạo: TB %.1f mm, lớn nhất %.1f mm | tilt lớn nhất %.3f rad" %
          (1e3 * np.mean(err), 1e3 * np.max(err), tilt))
    if not a.no_obstacle:
        ok = mingap > 0.07
        print("khoảng cách nhỏ nhất từ tâm drone tới hộp: %.1f mm  -> %s" %
              (1e3 * mingap, "KHÔNG VA CHẠM (đạt)" if ok else "VA CHẠM / QUÁ SÁT (không đạt)"))


if __name__ == "__main__":
    main()
