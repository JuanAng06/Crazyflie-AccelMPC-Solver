"""Giai đoạn 2.2 - Đường bay tham chiếu x_ref.

Hai loại tham chiếu, cùng giao diện:
    ref.state(t)            -> vector 12 chiều tại thời điểm t
    ref.window(t0, H, dt)   -> ma trận (H+1, 12): tham chiếu cho H+1 bước MPC bắt đầu từ t0
                               (đưa thẳng vào AccelMPCSolver.solve(x0, x_ref=...))

Thứ tự trạng thái: [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz].
Góc và vận tốc góc tham chiếu để bằng 0 (bay êm quanh hover).
"""
import numpy as np


class PointReference:
    """Quỹ đạo điểm: luôn tham chiếu tới một điểm đứng yên."""

    def __init__(self, point=(0.0, 0.0, 0.5)):
        self.x = np.zeros(12)
        self.x[0:3] = point

    def state(self, t):
        return self.x.copy()

    def window(self, t0, H, dt):
        return np.tile(self.x, (H + 1, 1))


class Figure8Reference:
    """Quỹ đạo hình số 8 trong mặt phẳng ngang, độ cao không đổi.

        x(t) = cx + a * sin(theta)
        y(t) = cy + b * sin(2 * theta)
        z(t) = cz
    với theta = omega * tau(t). tau(t) làm đoạn khởi động mượt trong t_ramp giây đầu
    (vận tốc tham chiếu bắt đầu từ 0 thay vì nhảy vọt), sau đó tau(t) = t - t_ramp/2.
    """

    def __init__(self, center=(0.0, 0.0, 0.5), a=0.5, b=0.25, period=10.0, t_ramp=2.0):
        self.c = np.asarray(center, float)
        self.a, self.b = float(a), float(b)
        self.omega = 2.0 * np.pi / float(period)
        self.t_ramp = float(t_ramp)

    def _tau(self, t):
        """Trả về (tau, dtau/dt): thời gian "pha" và đạo hàm của nó."""
        t = max(float(t), 0.0)
        if self.t_ramp <= 0.0:
            return t, 1.0
        if t < self.t_ramp:
            return t * t / (2.0 * self.t_ramp), t / self.t_ramp
        return t - 0.5 * self.t_ramp, 1.0

    def state(self, t):
        tau, dtau = self._tau(t)
        th, dth = self.omega * tau, self.omega * dtau
        x = np.zeros(12)
        x[0] = self.c[0] + self.a * np.sin(th)
        x[1] = self.c[1] + self.b * np.sin(2.0 * th)
        x[2] = self.c[2]
        x[6] = self.a * np.cos(th) * dth
        x[7] = 2.0 * self.b * np.cos(2.0 * th) * dth
        return x

    def window(self, t0, H, dt):
        return np.array([self.state(t0 + k * dt) for k in range(H + 1)])

    def path_points(self, n=200, t_end=None):
        """Các điểm 3D dọc quỹ đạo (dùng để vẽ đường tham chiếu trong PyBullet)."""
        if t_end is None:
            t_end = self.t_ramp / 2.0 + 2.0 * np.pi / self.omega
        return np.array([self.state(t)[0:3] for t in np.linspace(0.0, t_end, n)])


class TakeoffReference:
    """Bọc một quỹ đạo: drone cất cánh thẳng đứng từ z0 lên độ cao của quỹ đạo trong t_takeoff giây
    (giữ nguyên x,y ban đầu, profile smoothstep nên vận tốc bắt đầu/kết thúc bằng 0), sau đó bắt đầu
    bám quỹ đạo gốc (đồng hồ của quỹ đạo gốc chạy từ t = t_takeoff)."""

    def __init__(self, ref, t_takeoff=2.0, z0=0.05):
        self.ref, self.T, self.z0 = ref, float(t_takeoff), float(z0)

    def state(self, t):
        x = self.ref.state(max(float(t) - self.T, 0.0))
        z1 = x[2]
        if t < self.T:
            u = max(float(t), 0.0) / self.T
            x[2] = self.z0 + (z1 - self.z0) * (3 * u * u - 2 * u ** 3)
            x[8] = (z1 - self.z0) * (6 * u - 6 * u * u) / self.T
        return x

    def window(self, t0, H, dt):
        return np.array([self.state(t0 + k * dt) for k in range(H + 1)])

    def path_points(self, n=200, t_end=None):
        return self.ref.path_points(n, t_end)
