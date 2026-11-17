"""Giai đoạn 2 - Demo 3D: Crazyflie + quỹ đạo số 8 + vật cản trong PyBullet.

Chạy (trong PyCharm: chuột phải -> Run, hoặc Terminal có (drones)):
    python demo_phase2.py
    python demo_phase2.py --moving          # vật cản dao động ngang đường bay
    python demo_phase2.py --solid           # vật cản có va chạm thật (drone sẽ đâm vào)

Những gì bạn sẽ thấy:
  - Đường xanh dương: quỹ đạo số 8 tham chiếu (2.2)
  - Khối đỏ chắn ở mũi bên phải của số 8 (2.3)
  - Drone Crazyflie bay theo quỹ đạo bằng bộ điều khiển PID có sẵn của gym-pybullet-drones
    (CHỈ là "người lái tạm" để bạn nhìn thấy cảnh; MPC của nhóm sẽ thay thế ở Giai đoạn 3).
    PID không biết vật cản nên mặc định vật cản là "ma" (drone bay xuyên qua).
  - Terminal in mỗi 0.5 s: vị trí drone, vị trí vật cản đọc từ PyBullet, và biên
    (trục, phía, vị trí) sẽ được đưa vào solver qua set_obstacle_bounds().
"""
import argparse
import os
import sys
import time

import numpy as np
import pybullet as p

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.utils import sync

from crazyflie_model import state20_to_x12
from reference import Figure8Reference
from obstacles import Obstacle, separating_bound

AXIS_NAME = {0: "x", 1: "y", 2: "z"}


def run(duration=25, solid=False, moving=False, margin=0.10, obstacle_x=0.5):
    ref = Figure8Reference(center=(0.0, 0.0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0)

    env = CtrlAviary(drone_model=DroneModel.CF2X,
                     num_drones=1,
                     initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
                     initial_rpys=np.zeros((1, 3)),
                     physics=Physics.PYB,
                     neighbourhood_radius=10,
                     pyb_freq=240,
                     ctrl_freq=48,
                     gui=True,
                     record=False,
                     obstacles=False,            # tắt vật cản mặc định, dùng vật cản của mình
                     user_debug_gui=False)
    cid = env.getPyBulletClient()

    # ---- 2.2: vẽ đường bay tham chiếu --------------------------------------------------
    pts = ref.path_points(200)
    for a, b in zip(pts[:-1], pts[1:]):
        p.addUserDebugLine(a.tolist(), b.tolist(), [0.0, 0.6, 1.0], 2, 0, physicsClientId=cid)

    # ---- 2.3: tạo vật cản chắn ngang đường bay -----------------------------------------
    motion = dict(axis=1, amplitude=0.3, period=6.0) if moving else None
    obstacle = Obstacle(shape="box", center=(obstacle_x, 0.0, 0.5),
                        half_size=(0.05, 0.15, 0.3), motion=motion, solid=solid)
    obstacle.spawn(cid)

    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    action = np.zeros((1, 4))
    START = time.time()
    print_every = int(0.5 * env.CTRL_FREQ)

    for i in range(int(duration * env.CTRL_FREQ)):
        t = i / env.CTRL_FREQ
        obstacle.update(t, cid)

        obs, reward, terminated, truncated, info = env.step(action)
        state20 = obs[0]

        xref = ref.state(t)
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=state20,
            target_pos=xref[0:3],
            target_rpy=np.zeros(3))

        if i == 0:
            print("obs[0].shape =", obs[0].shape, "(phải là (20,))")
        if i % print_every == 0:
            x12 = state20_to_x12(state20)                     # trạng thái x thuộc R^12
            obs_pos = obstacle.read_position(cid)             # vị trí vật cản thời gian thực
            b = separating_bound(obs_pos, obstacle.half_size, x12[0:3], margin)
            if b is None:
                bound_txt = "drone ĐANG TRONG vùng đệm vật cản (solver sẽ vô nghiệm)"
            else:
                axis, side, position = b
                bound_txt = "set_obstacle_bounds(axis=%d [%s], side='%s', position=%.3f)" % (
                    axis, AXIS_NAME[axis], side, position)
            print("t=%5.1fs | drone=(%.2f, %.2f, %.2f) | vật cản=(%.2f, %.2f, %.2f) | %s" % (
                t, *x12[0:3], *obs_pos, bound_txt))

        sync(i, START, env.CTRL_TIMESTEP)

    env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Demo Giai đoạn 2: số 8 + vật cản")
    ap.add_argument("--duration", type=int, default=25, help="thời gian mô phỏng [s]")
    ap.add_argument("--solid", action="store_true", help="vật cản có va chạm vật lý")
    ap.add_argument("--moving", action="store_true", help="vật cản dao động theo trục y")
    ap.add_argument("--margin", type=float, default=0.10, help="vùng đệm an toàn quanh vật cản [m]")
    ap.add_argument("--obstacle_x", type=float, default=0.5, help="vị trí x của vật cản [m]")
    a = ap.parse_args()
    run(duration=a.duration, solid=a.solid, moving=a.moving, margin=a.margin,
        obstacle_x=a.obstacle_x)
