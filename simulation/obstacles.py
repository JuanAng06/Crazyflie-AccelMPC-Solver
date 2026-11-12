"""Giai đoạn 2.3 - Vật cản (tĩnh / động) và đưa vị trí vật cản vào bộ giải.

Hai phần:
  1) Class Obstacle: tạo khối hộp hoặc hình cầu trong PyBullet, có thể đứng yên hoặc dao động,
     và đọc lại vị trí THỜI GIAN THỰC từ PyBullet.
  2) separating_bound() / update_solver_bounds(): đổi vị trí vật cản + vị trí drone thành
     các biên mà AccelMPCSolver.set_obstacle_bounds() hiểu được.

Giới hạn của solver cần biết:
  - set_obstacle_bounds(axis, side, position) chỉ đặt MỘT biên trên/dưới trên MỘT trục vị trí
    (0=x, 1=y, 2=z) cho mọi bước k = 1..H. Vậy mỗi vật cản được mô hình hóa bằng một "mặt phẳng
    tách": drone phải ở lại phía hiện tại của nó. Đây là xấp xỉ bảo thủ, hợp lệ cho khối hộp.
  - Vật cản động được coi là đứng yên trong cả chân trời dự báo (chân trời chỉ ~62 ms).
"""
import numpy as np


# =========================================================================================
# Phần 1: vật cản trong PyBullet
# =========================================================================================
class Obstacle:
    """Vật cản hình hộp ('box') hoặc cầu ('sphere').

    center    : tâm ban đầu (x, y, z) [m]
    half_size : với 'box' là nửa kích thước 3 chiều (hx, hy, hz); với 'sphere' là bán kính (số)
    motion    : None (đứng yên) hoặc dict(axis=1, amplitude=0.3, period=6.0)
                -> tâm dao động hình sin quanh center theo trục `axis`
    solid     : True = có va chạm vật lý; False = chỉ để nhìn (drone bay xuyên qua)
    """

    def __init__(self, shape="box", center=(0.5, 0.0, 0.5), half_size=(0.05, 0.15, 0.3),
                 motion=None, solid=False, rgba=(1.0, 0.2, 0.2, 0.8)):
        self.shape = shape
        self.center = np.asarray(center, float)
        if shape == "sphere":
            r = float(np.max(half_size)) if np.ndim(half_size) else float(half_size)
            self.half_size = np.array([r, r, r])
            self.radius = r
        else:
            self.half_size = np.asarray(half_size, float)
        self.motion = motion
        self.solid = solid
        self.rgba = list(rgba)
        self.body = None

    def position_at(self, t):
        """Vị trí tâm tại thời điểm t theo kế hoạch chuyển động."""
        pos = self.center.copy()
        if self.motion:
            m = self.motion
            pos[m["axis"]] += m["amplitude"] * np.sin(2.0 * np.pi * t / m["period"])
        return pos

    # ---- các hàm dưới đây cần pybullet ----------------------------------------------
    def spawn(self, client_id):
        import pybullet as p
        if self.shape == "sphere":
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius,
                                      rgbaColor=self.rgba, physicsClientId=client_id)
            col = (p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius,
                                          physicsClientId=client_id) if self.solid else -1)
        else:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=self.half_size.tolist(),
                                      rgbaColor=self.rgba, physicsClientId=client_id)
            col = (p.createCollisionShape(p.GEOM_BOX, halfExtents=self.half_size.tolist(),
                                          physicsClientId=client_id) if self.solid else -1)
        self.body = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                                      baseVisualShapeIndex=vis,
                                      basePosition=self.position_at(0.0).tolist(),
                                      physicsClientId=client_id)
        return self.body

    def update(self, t, client_id):
        """Di chuyển vật cản (nếu là vật cản động)."""
        if self.motion and self.body is not None:
            import pybullet as p
            p.resetBasePositionAndOrientation(self.body, self.position_at(t).tolist(),
                                              [0, 0, 0, 1], physicsClientId=client_id)

    def read_position(self, client_id):
        """Đọc vị trí vật cản thời gian thực TỪ PYBULLET (đây là số liệu đưa vào solver)."""
        import pybullet as p
        pos, _ = p.getBasePositionAndOrientation(self.body, physicsClientId=client_id)
        return np.asarray(pos, float)


# =========================================================================================
# Phần 2: từ vị trí vật cản sang biên cho solver (chỉ dùng numpy)
# =========================================================================================
def separating_bound(obs_pos, obs_half, drone_pos, margin=0.10, allow_inside=False):
    """Tìm mặt phẳng tách drone khỏi hộp vật cản đã phình ra thêm `margin`.

    Trả về (axis, side, position) cho solver.set_obstacle_bounds.
    Nếu drone đã nằm trong vùng đệm (không còn mặt phẳng tách thật sự):
      - allow_inside=False: trả None (dùng để phát hiện/ghi log)
      - allow_inside=True : vẫn trả mặt gần được tách nhất (để KHÔNG BAO GIỜ bỏ ràng buộc)

    Cách chọn: trên mỗi trục tính khoảng cách tách sep_i = |p_i - c_i| - (h_i + margin).
    Chọn trục có sep lớn nhất. Drone ở phía nào của vật cản trên trục đó thì đặt biên ở mặt
    hộp phía đó.
    """
    obs_pos = np.asarray(obs_pos, float)
    obs_half = np.asarray(obs_half, float)
    drone_pos = np.asarray(drone_pos, float)

    sep = np.abs(drone_pos - obs_pos) - (obs_half + margin)
    axis = int(np.argmax(sep))
    if sep[axis] <= 0.0 and not allow_inside:
        return None
    if drone_pos[axis] < obs_pos[axis]:
        return axis, "upper", float(obs_pos[axis] - obs_half[axis] - margin)
    return axis, "lower", float(obs_pos[axis] + obs_half[axis] + margin)


def _tighten(solver, state_idx, side, k, value):
    """Siết biên của biến trạng thái `state_idx` tại bước k (0..11 = x,y,z,roll,...,wz).

    Solver chỉ có set_obstacle_bounds (một giá trị cho mọi k), nên ở đây ghi trực tiếp vào
    vector biên solver.l / solver.u (đây là thuộc tính công khai của AccelMPCSolver).
    """
    idx = solver.nd + k * solver.nxu + state_idx
    if side == "upper":
        solver.u[idx] = min(solver.u[idx], value)
    else:
        solver.l[idx] = max(solver.l[idx], value)


def update_solver_bounds(solver, drone_pos, obstacles, margin=0.10,
                         drone_vel=None, dt=None, forced_steps=3, eps=1e-4,
                         a_brake=1.0, a_floor=2.0):
    """Cài lại toàn bộ biên vật cản cho solver ở mỗi bước điều khiển (gọi TRƯỚC solver.solve).

    obstacles: danh sách (vị trí tâm hiện tại, nửa kích thước)

    Hai lớp biên cho mỗi vật cản (khi truyền drone_vel và dt):
      1) Biên VỊ TRÍ (mặt phẳng tách). Riêng k = 1..3 được nới vừa đủ (vị trí ở các bước này bị
         động lực học ép sẵn: x0 + k*dt*v0), từ k = 4 trở đi giữ NGUYÊN biên cứng.
      2) Biên VẬN TỐC (đường bao phanh): vận tốc tiến về phía vật cản không được vượt
         v_allow = sqrt(2 * a_brake * d), với d là khoảng cách còn lại tới biên vị trí. Nhờ vậy
         drone bắt đầu phanh SỚM thay vì đâm vào biên rồi mới phanh (chân trời chỉ ~62 ms).
         Ở k = 1..3 vận tốc bị ép sẵn nên biên bằng đúng vận tốc hiện tại; từ k = 4 trở đi cho
         phép giảm tối đa a_floor m/s^2 để bài toán còn khả thi.

    Trả về (danh sách biên vị trí đã cài, danh sách chỉ số vật cản mà drone đang trong vùng đệm).
    Ràng buộc KHÔNG bao giờ bị bỏ, kể cả khi drone đã trong vùng đệm.
    """
    drone_pos = np.asarray(drone_pos, float)
    use_dyn = drone_vel is not None and dt is not None
    merged = {}      # (axis, side) -> position ; nhiều vật cản cùng trục thì lấy biên chặt nhất
    brakes = {}      # (axis, side) -> d nhỏ nhất (khoảng cách còn lại)
    inside = []
    for i, (pos, half) in enumerate(obstacles):
        if separating_bound(pos, half, drone_pos, margin) is None:
            inside.append(i)
        axis, side, position = separating_bound(pos, half, drone_pos, margin, allow_inside=True)

        d = (position - drone_pos[axis]) if side == "upper" else (drone_pos[axis] - position)

        key = (axis, side)
        if key not in merged:
            merged[key] = position
            brakes[key] = d
        elif side == "upper":
            merged[key] = min(merged[key], position)
            brakes[key] = min(brakes[key], d)
        else:
            merged[key] = max(merged[key], position)
            brakes[key] = min(brakes[key], d)

    solver.reset_bounds()
    applied = []
    for (axis, side), position in merged.items():
        applied.append((axis, side, position))
        for k in range(1, solver.H + 1):
            val = position
            if use_dyn and k <= forced_steps and axis < 3:
                # vị trí ở k = 1..3 bị động lực học ép sẵn: chỉ nới biên ở các bước này
                f = drone_pos[axis] + k * dt * drone_vel[axis]
                val = max(position, f + eps) if side == "upper" else min(position, f - eps)
            _tighten(solver, axis, side, k, val)

        if use_dyn and axis < 3:
            v_allow = np.sqrt(2.0 * a_brake * max(brakes[(axis, side)], 0.0))
            vc0 = drone_vel[axis] if side == "upper" else -drone_vel[axis]     # vận tốc tiến về vật cản
            for k in range(1, solver.H + 1):
                if k <= forced_steps:
                    cap = vc0 + eps
                else:
                    cap = max(v_allow, vc0 - a_floor * (k - forced_steps) * dt)
                if side == "upper":
                    _tighten(solver, 6 + axis, "upper", k, cap)
                else:
                    _tighten(solver, 6 + axis, "lower", k, -cap)
    return applied, inside


def shape_reference(W, drone_pos, obstacles, margin=0.10):
    """Uốn lại cửa sổ tham chiếu W (H+1, 12) để KHÔNG đi xuyên vùng đệm vật cản.

    Dùng đúng mặt phẳng tách như update_solver_bounds: tham chiếu bị kẹp về phía hiện tại của
    drone, vận tốc tham chiếu hướng vào vật cản bị đặt về 0. Nhờ vậy MPC bám một đường bay khả thi
    (thay vì bị ràng buộc cứng kéo ngược lại một đường bay đi xuyên vật cản).
    """
    W = np.array(W, float)
    for pos, half in obstacles:
        axis, side, position = separating_bound(pos, half, drone_pos, margin, allow_inside=True)
        if side == "upper":
            W[:, axis] = np.minimum(W[:, axis], position)
            W[:, 6 + axis] = np.minimum(W[:, 6 + axis], 0.0) if False else np.where(
                W[:, axis] >= position, np.minimum(W[:, 6 + axis], 0.0), W[:, 6 + axis])
        else:
            W[:, axis] = np.maximum(W[:, axis], position)
            W[:, 6 + axis] = np.where(W[:, axis] <= position, np.maximum(W[:, 6 + axis], 0.0), W[:, 6 + axis])
    return W
