"""Giai đoạn 3-4 - Vòng điều khiển kín: AccelMPCSolver <-> PyBullet (gym-pybullet-drones).

Chạy (Anaconda Prompt, env drones, trong thư mục chứa file này):
    python run_simulation.py                              # cửa sổ 3D, vật cản "ma"
    python run_simulation.py --solid                      # vật cản có va chạm thật
    python run_simulation.py --no_obstacle                # chỉ bay số 8
    python run_simulation.py --dt 0.005 --iters 50
    python run_simulation.py --takeoff 2 --duration 16 --realtime --save ket_qua.npz
    python run_simulation.py --takeoff 2 --duration 16 --no_gui --video demo.mp4 --save ket_qua.npz

Mỗi bước điều khiển (đúng như đề Giai đoạn 3.2):
    1. đọc trạng thái x (12) và vị trí vật cản từ PyBullet
    2. solver.reset_bounds(); đặt biên vật cản (tương đương set_obstacle_bounds)
    3. u0, info = solver.solve(x, x_ref_window, max_iter, tol)
    4. cắt u0 theo giới hạn motor, đổi u -> RPM 4 motor, env.step()
    5. solver.warm_start_shift()

Giai đoạn 4:
    --save  : ghi quỹ đạo thực/tham chiếu ra .npz cho plot_results.py (biểu đồ + RMS)
    --video : quay video .mp4 bằng camera ảo (cần: pip install imageio imageio-ffmpeg)
    --takeoff S : cất cánh thẳng đứng trong S giây rồi mới bay số 8
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
import pybullet as p

from crazyflie_model import (build_linear_model, state20_to_x12, u_dev_to_rpm, U_DEV_MIN, U_DEV_MAX)
from reference import Figure8Reference, TakeoffReference
from obstacles import Obstacle, update_solver_bounds, shape_reference
from solver_ext import make_solver


def _dot(cid, pos, rgba, radius):
    """Chấm màu chỉ để nhìn (không va chạm) - dùng vẽ đường tham chiếu và vệt bay trong video."""
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=cid)
    p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                      basePosition=[float(v) for v in pos], physicsClientId=cid)


def run(a):
    H = 31
    dt = a.dt
    ctrl_freq = int(round(1.0 / dt))
    pyb_freq = 1000
    assert pyb_freq % ctrl_freq == 0, "dt phải là 0.002 hoặc 0.005 (hoặc 0.001)"

    A, B = build_linear_model(dt, "gym")
    solver = make_solver(A, B, H, dare_terminal=not a.no_dare)
    ref = Figure8Reference(center=(0, 0, 0.5), a=0.5, b=0.25, period=a.period, t_ramp=2.0)
    if a.takeoff > 0:
        ref = TakeoffReference(ref, t_takeoff=a.takeoff, z0=0.05)

    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=np.array([ref.state(0.0)[0:3]]),
                     initial_rpys=np.zeros((1, 3)), physics=Physics.PYB,
                     pyb_freq=pyb_freq, ctrl_freq=ctrl_freq,
                     gui=not a.no_gui, record=False, obstacles=False, user_debug_gui=False)
    cid = env.getPyBulletClient()
    assert abs(env.CTRL_TIMESTEP - dt) < 1e-9, env.CTRL_TIMESTEP

    # đường tham chiếu: đường kẻ trong cửa sổ GUI; chấm xanh để thấy trong video
    pts = ref.path_points(200)
    if not a.no_gui:
        for p0, p1 in zip(pts[:-1], pts[1:]):
            p.addUserDebugLine(p0.tolist(), p1.tolist(), [0, 0.6, 1.0], 2, 0, physicsClientId=cid)
    if a.video:
        for q in pts[::3]:
            _dot(cid, q, [0.1, 0.5, 1.0, 1.0], 0.008)

    obstacle = None
    if not a.no_obstacle:
        obstacle = Obstacle(shape="box", center=(a.obstacle_x, 0.0, 0.5), half_size=(0.05, 0.15, 0.3),
                            solid=a.solid)
        obstacle.spawn(cid)

    # ---- quay video (camera ảo, không phụ thuộc cửa sổ) ---------------------------------
    writer = None
    if a.video:
        import imageio.v2 as imageio
        W, Hpx = 960, 540
        stride = max(1, int(round(ctrl_freq / a.fps)))
        writer = imageio.get_writer(a.video, fps=ctrl_freq / stride, macro_block_size=1)
        view = p.computeViewMatrixFromYawPitchRoll([0.0, 0.0, 0.35], a.cam_dist, a.cam_yaw, -35, 0, 2,
                                                   physicsClientId=cid)
        proj = p.computeProjectionMatrixFOV(60, W / Hpx, 0.05, 20.0, physicsClientId=cid)

    action = np.zeros((1, 4))
    n = int(a.duration / dt)
    mingap, tilt, iters_log, t_log, err = 1e9, 0.0, [], [], []
    rp_log, rd_log = [], []
    log_t, log_pos, log_ref = [], [], []
    trail_every = int(0.1 / dt)
    START = time.time()
    obs = None
    for i in range(n):
        t = i * dt
        if obs is None:
            obs, *_ = env.step(np.tile(u_dev_to_rpm(np.zeros(4)), (1, 1)))   # bước đầu: hover
        x = state20_to_x12(obs[0])
        if i == 0:
            print("obs[0].shape =", obs[0].shape, "(phải là (20,))")

        W_ref = ref.window(t, H, dt)
        if obstacle is not None:
            op = obstacle.read_position(cid)
            half = obstacle.half_size
            update_solver_bounds(solver, x[0:3], [(op, half)], a.margin,
                                 drone_vel=x[6:9], dt=dt, a_brake=0.5, a_floor=0.5)
            W_ref = shape_reference(W_ref, x[0:3], [(op, half)], a.margin)
            mingap = min(mingap, float(np.max(np.abs(x[0:3] - op) - half)))
        else:
            solver.reset_bounds()

        t0 = time.perf_counter()
        u0, info = solver.solve(x, W_ref, max_iter=a.iters, tol=a.tol)
        t_log.append(time.perf_counter() - t0)
        iters_log.append(info["iters"])
        rp_log.append(info["r_prim"]); rd_log.append(info["r_dual"])
        u0 = np.clip(u0, U_DEV_MIN, U_DEV_MAX)

        action[0, :] = u_dev_to_rpm(u0)
        obs, reward, terminated, truncated, _ = env.step(action)
        solver.warm_start_shift()

        xn = state20_to_x12(obs[0])
        tilt = max(tilt, float(np.abs(xn[3:5]).max()))
        xr = ref.state(t + dt)
        err.append(float(np.linalg.norm(xn[0:2] - xr[0:2])))
        log_t.append(t + dt); log_pos.append(xn[0:3].copy()); log_ref.append(xr[0:3].copy())

        if (a.video or not a.no_gui) and i % trail_every == 0:
            _dot(cid, xn[0:3], [1.0, 0.2, 0.2, 1.0], 0.006)          # vệt bay thực tế (đỏ)
        if writer is not None and i % stride == 0:
            img = p.getCameraImage(W, Hpx, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=cid)
            writer.append_data(np.reshape(img[2], (Hpx, W, 4))[:, :, :3].astype(np.uint8))
        if i % int(0.5 / dt) == 0:
            print("t=%5.1fs drone=(%.2f,%.2f,%.2f) tilt=%.1f° ADMM=%d vòng %.2f ms" % (
                t, *xn[0:3], np.degrees(np.abs(xn[3:5]).max()), info["iters"], 1e3 * t_log[-1]))
        if not np.isfinite(xn).all() or abs(xn[3]) > 1.2 or abs(xn[4]) > 1.2:
            print("MẤT ỔN ĐỊNH: góc nghiêng quá lớn, dừng.")
            break
        if not a.no_gui and a.realtime:
            time.sleep(max(0.0, START + (i + 1) * dt - time.time()))
    if writer is not None:
        writer.close()
        print("Đã ghi video:", os.path.abspath(a.video))
    env.close()

    if a.save:
        np.savez(a.save, t=np.array(log_t), pos=np.array(log_pos), ref=np.array(log_ref),
                 obs_center=(obstacle.center if obstacle is not None else np.full(3, np.nan)),
                 obs_half=(obstacle.half_size if obstacle is not None else np.full(3, np.nan)),
                 margin=a.margin, dt=dt, takeoff=a.takeoff)
        print("Đã lưu dữ liệu:", os.path.abspath(a.save))

    print("\n===== KẾT QUẢ =====")
    print("ADMM: %d vòng/bước (cố định, tol=%s) | TB %.2f ms/bước (ngân sách %.1f ms)" % (
        int(np.mean(iters_log)), a.tol, 1e3 * np.mean(t_log), 1e3 * dt))
    if a.tol is not None:
        conv = np.mean(np.array(iters_log) < a.iters) * 100
        print("hội tụ tới tol=%g trước khi hết ngân sách: %.1f%% số bước" % (a.tol, conv))
    rp, rd = np.array(rp_log), np.array(rd_log)
    if np.isfinite(rp).all():
        print("sai số cuối mỗi bước: r_prim TB %.2e / max %.2e | r_dual TB %.2e / max %.2e" % (
            rp.mean(), rp.max(), rd.mean(), rd.max()))
    else:
        print("(r_prim/r_dual chỉ có khi đặt --tol, ví dụ --tol 1e-5)")
    print("sai số bám số 8: TB %.1f mm, lớn nhất %.1f mm | tilt lớn nhất %.1f°" % (
        1e3 * np.mean(err), 1e3 * np.max(err), np.degrees(tilt)))
    if obstacle is not None:
        print("khoảng cách nhỏ nhất tâm drone-hộp: %.1f mm -> %s" % (
            1e3 * mingap, "KHÔNG va chạm" if mingap > 0.07 else "VA CHẠM / quá sát"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Giai đoạn 3-4: MPC vòng kín trong PyBullet")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--dt", type=float, default=0.002, help="0.002 hoặc 0.005 s")
    ap.add_argument("--iters", type=int, default=50, help="số vòng ADMM tối đa mỗi bước")
    ap.add_argument("--tol", type=float, default=None, help="ngưỡng dừng sớm (None = chạy đủ iters)")
    ap.add_argument("--period", type=float, default=10.0)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--obstacle_x", type=float, default=0.5)
    ap.add_argument("--takeoff", type=float, default=0.0, help="thời gian cất cánh [s], 0 = bắt đầu ở độ cao bay")
    ap.add_argument("--solid", action="store_true")
    ap.add_argument("--no_obstacle", action="store_true")
    ap.add_argument("--no_gui", action="store_true")
    ap.add_argument("--no_dare", action="store_true", help="dùng Qf=Q gốc của repo (để so sánh)")
    ap.add_argument("--realtime", action="store_true", help="làm chậm cho bằng thời gian thực")
    ap.add_argument("--save", default=None, help="ghi dữ liệu quỹ đạo ra file .npz")
    ap.add_argument("--video", default=None, help="ghi video .mp4 (cần imageio + imageio-ffmpeg)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--cam_yaw", type=float, default=45.0, help="góc xoay camera video [độ]")
    ap.add_argument("--cam_dist", type=float, default=2.0, help="khoảng cách camera video [m]")
    run(ap.parse_args())
