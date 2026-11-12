"""Giai đoạn 2.1 - Mô hình động lực học Crazyflie 2.1 (35 g).

Các thông số và ma trận ở đây KHỚP với hàm build_crazyflie_model() trong
tests/test_admm_math.py của Thành viên A, để solver và mô phỏng dùng chung một mô hình.

Quy ước:
  - Trạng thái x (12): [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    (vị trí tuyệt đối trong hệ thế giới; góc và vận tốc góc là độ lệch quanh hover)
  - Đầu vào u (4): ĐỘ LỆCH lực đẩy từng motor so với lực đẩy hover [N]
    (u = 0 nghĩa là mỗi motor đang đẩy đúng m*g/4)
  - Mô hình tuyến tính hóa quanh hover, chỉ đúng khi góc nghiêng nhỏ.
"""
import numpy as np

NX, NU = 12, 4

# ---- Thông số vật lý Crazyflie 2.1 -------------------------------------------------
MASS = 0.027                    # khối lượng [kg]               (khớp cf2x.urdf)
G = 9.81                        # gia tốc trọng trường [m/s^2]
IXX = 1.4e-5                    # quán tính quanh trục x [kg m^2] (khớp cf2x.urdf)
IYY = 1.4e-5                    # quán tính quanh trục y [kg m^2] (khớp cf2x.urdf)
IZZ = 2.17e-5                   # quán tính quanh trục z [kg m^2] (khớp cf2x.urdf)

# Hai bộ giá trị cho cánh tay đòn hiệu dụng L và hệ số mô-men xoắn yaw / lực đẩy:
#   "team": đúng như tests/test_admm_math.py của Thành viên A (số liệu Crazyflie thực)
#   "gym" : khớp file cf2x.urdf của gym-pybullet-drones (arm = 0.0397, km/kf = 7.94e-12/3.16e-10)
PRESETS = {
    "team": dict(arm_l=0.046 / np.sqrt(2), yaw_coef=0.005964552),
    "gym":  dict(arm_l=0.0397 / np.sqrt(2), yaw_coef=7.94e-12 / 3.16e-10),
}
ARM_L = PRESETS["team"]["arm_l"]                  # giữ tên cũ để tương thích
YAW_TORQUE_COEF = PRESETS["team"]["yaw_coef"]

# ---- Thông số của simulator (cf2x.urdf) dùng khi đổi lực đẩy <-> RPM ----------------
KF_SIM = 3.16e-10               # lực đẩy = KF * RPM^2 [N]
KM_SIM = 7.94e-12
THRUST2WEIGHT = 2.25            # lực đẩy tối đa / trọng lượng

HOVER_THRUST_PER_MOTOR = MASS * G / 4.0                              # ~0.0662 N
MAX_THRUST_PER_MOTOR = THRUST2WEIGHT * MASS * G / 4.0                # ~0.1490 N
# Giới hạn độ lệch u (solver KHÔNG có ràng buộc đầu vào nên phải tự cắt ngoài solver):
U_DEV_MIN = -HOVER_THRUST_PER_MOTOR                                  # ~ -0.0662 N
U_DEV_MAX = MAX_THRUST_PER_MOTOR - HOVER_THRUST_PER_MOTOR            # ~ +0.0828 N


def thrust_to_rpm(thrust_per_motor):
    """Lực đẩy mỗi motor [N] (mảng 4) -> RPM của simulator, đã cắt trong [0, RPM tối đa]."""
    f = np.clip(np.asarray(thrust_per_motor, float), 0.0, MAX_THRUST_PER_MOTOR)
    return np.sqrt(f / KF_SIM)


def u_dev_to_rpm(u_dev):
    """Độ lệch lực đẩy u (đầu ra solver, 4 motor) -> RPM: hover + u, cắt trong giới hạn."""
    return thrust_to_rpm(HOVER_THRUST_PER_MOTOR + np.asarray(u_dev, float))


def build_linear_model(dt: float = 0.002, preset: str = "team"):
    """Trả về (A, B) rời rạc hóa Euler: A = I + dt*Ac, B = dt*Bc.

    dt = 0.002 s là bước thời gian mà solver của nhóm đang dùng (config/mpc_config.yaml).
    preset: "team" (mặc định, khớp test của Thành viên A) hoặc "gym" (khớp cf2x.urdf).
    """
    arm_l, yaw_coef = PRESETS[preset]["arm_l"], PRESETS[preset]["yaw_coef"]
    Ac = np.zeros((NX, NX))
    Ac[0:3, 6:9] = np.eye(3)          # vị trí' = vận tốc
    Ac[3:6, 9:12] = np.eye(3)         # góc' = vận tốc góc
    Ac[6, 4] = G                      # vx' =  g * pitch
    Ac[7, 3] = -G                     # vy' = -g * roll

    Bc = np.zeros((NX, NU))
    Bc[8, :] = 1.0 / MASS                                           # vz'
    Bc[9, :] = arm_l * np.array([-1, -1, 1, 1]) / IXX               # wx'
    Bc[10, :] = arm_l * np.array([-1, 1, 1, -1]) / IYY              # wy'
    Bc[11, :] = yaw_coef * np.array([-1, 1, -1, 1]) / IZZ    # wz'

    return np.eye(NX) + dt * Ac, dt * Bc


def initial_state(pos=(0.0, 0.0, 0.5), rpy=(0.0, 0.0, 0.0),
                  vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)):
    """Dựng x0 thuộc R^12 theo thứ tự [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]."""
    return np.concatenate([np.asarray(pos, float), np.asarray(rpy, float),
                           np.asarray(vel, float), np.asarray(ang_vel, float)])


def state20_to_x12(state20):
    """Đổi vector trạng thái 20 chiều của gym-pybullet-drones (obs[j]) sang x thuộc R^12.

    Bố cục 20 chiều của gym-pybullet-drones:
        [0:3] vị trí, [3:7] quaternion, [7:10] roll-pitch-yaw,
        [10:13] vận tốc, [13:16] vận tốc góc, [16:20] RPM lần trước.
    HÃY IN obs[0].shape (phải là (20,)) để xác nhận trước khi dùng.
    """
    s = np.asarray(state20, float)
    return np.concatenate([s[0:3], s[7:10], s[10:13], s[13:16]])


if __name__ == "__main__":
    A, B = build_linear_model()
    print("A:", A.shape, " B:", B.shape)
    print("Lực đẩy hover mỗi motor: %.4f N (RPM hover = %.0f)"
          % (HOVER_THRUST_PER_MOTOR, u_dev_to_rpm(np.zeros(4))[0]))
    print("Giới hạn độ lệch u: [%.4f, %+.4f] N" % (U_DEV_MIN, U_DEV_MAX))
    _, Bg = build_linear_model(preset="gym")
    print("Tỉ số tác động B(gym)/B(team): roll/pitch %.3f, yaw %.2f"
          % (Bg[9, 0] / B[9, 0], Bg[11, 0] / B[11, 0]))
    print("x0 mẫu:", initial_state())
