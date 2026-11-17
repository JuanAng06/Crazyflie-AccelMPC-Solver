"""Giai đoạn 3 - kiểm tra THỨ TỰ và DẤU motor của simulator so với mô hình B (chạy 1 lần).
    python motor_check.py
Với mỗi motor i: tăng RPM motor đó, đọc gia tốc góc trong PyBullet, so dấu với cột B[9:12, i].
Nếu in 'SAI DẤU' hãy báo lại (kèm kết quả) để đổi thứ tự/dấu cột B trong crazyflie_model.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from crazyflie_model import build_linear_model, u_dev_to_rpm, state20_to_x12

_, B = build_linear_model(0.002, "gym")
ok = True
for m in range(4):
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1, initial_xyzs=np.array([[0, 0, 1.0]]),
                     physics=Physics.PYB, pyb_freq=1000, ctrl_freq=500, gui=False, obstacles=False,
                     user_debug_gui=False)
    u = np.zeros(4); u[m] = 0.03
    rpm = u_dev_to_rpm(u)[None, :]
    for _ in range(5):
        obs, *_ = env.step(rpm)
    w = state20_to_x12(obs[0])[9:12]
    exp = np.sign(B[9:12, m])
    got = np.sign(np.where(np.abs(w) < 1e-6, 0, w))
    good = np.array_equal(exp, got)
    ok &= good
    print("motor %d: mô hình %s | simulator %s | %s" % (m, exp.astype(int), got.astype(int),
                                                       "OK" if good else "SAI DẤU"))
    env.close()
print("TẤT CẢ ĐÚNG" if ok else "CÓ MOTOR SAI DẤU - báo lại")
