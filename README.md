  # Crazyflie AccelMPC Solver (Python Implementation)

Dự án tái hiện thuật toán bộ giải **ADMM-MPC** (Model Predictive Control) cho drone **Crazyflie 35g** (12 trạng thái ```x```, 4 đầu vào điều khiển ```u```) dựa trên nghiên cứu của bài báo *AccelMPC* (Grillo & Plancher, A²R Lab).

Dự án được thực hiện hoàn toàn bằng **Python/NumPy/SciPy** trên CPU với kiểu số thực `Float64`.

---

## 🛠️ Cấu trúc dự án

```text
Crazyflie-AccelMPC-Solver/
├── config/
│   └── mpc_config.yaml        # Tham số Q, R, penalty rho (4096.0), horizon H (31)
├── src/
│   ├── __init__.py            # Khai báo package Python
│   └── admm_mpc_solver.py     # Class AccelMPCSolver chính (nhúng Full Scaling & Cholesky Băng)
├── tests/
│   └── test_admm_math.py      # Unit test kiểm chứng toán học & so sánh với OSQP
├── .gitignore                 # Lọc file rác venv và cache Python
└── README.md                  # Tài liệu hướng dẫn nhanh
```

# 🚀 Hướng dẫn khởi chạy nhanh
## 1. Khởi tạo môi trường ảo & Cài đặt thư viện

```Bash
# Tạo và kích hoạt môi trường ảo (venv)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
# Cài đặt các thư viện bắt buộc
pip install numpy scipy osqp pytest pyyaml
```

## 2. Chạy Unit Test kiểm tra thuật toán
```Bash
pytest tests/test_admm_math.py
```

_(Nếu tất cả các test case báo màu xanh PASS, bộ giải toán học đã sẵn sàng để nối vào mô phỏng 3D)._
# 🔌 Hướng dẫn giao diện Bridge API (Dành cho Thành viên B)
Thành viên B khi viết môi trường mô phỏng 3D (PyBullet / Webots / Crazyflie Sim) có thể gọi trực tiếp solver như sau:
```Python
import numpy as np
from src.admm_mpc_solver import AccelMPCSolver

# 1. Khởi tạo Solver (Đã tích hợp Offline Banded Cholesky & Full Scaling)
solver = AccelMPCSolver(A, B, Q, R, H=31, rho=4096.0, scale_x=scale_x, scale_u=scale_u)

# 2. Vòng lặp điều khiển vòng kín (Closed-loop Control Loop)
for t in range(sim_steps):
    # a. Nếu phát hiện vật cản, ghi đè vị trí biên an toàn (ví dụ: né trục x tại position = 0.2m)
    solver.reset_bounds()
    solver.set_obstacle_bounds(axis=0, side='upper', position=0.2)
    
    # b. Giải MPC nhận tín hiệu điều khiển u0 (4 motor thrusts)
    # x0: trạng thái hiện tại (12x1), x_ref: tham chiếu bám đường bay
    u0, info = solver.solve(x0, x_ref, max_iter=20000, tol=1e-5)
    
    # c. Áp dụng u0 vào mô phỏng 3D
    drone.step(u0)
    
    # d. Dịch chuyển Warm-Start cho bước thời gian k+1
    solver.warm_start_shift()
```

## 📌 Đặc điểm kỹ thuật lõi
- Interleaved Decision Vector ($w$): Vector đan xen ```$[x_0, u_0, x_1, u_1, \ldots, x_H]$``` giúp đưa ma trận KKT về dạng băng với độ rộng ```h >= 27```.
- Offline Pre-factorization: Phân rã Cholesky dạng băng (```LL^T```) chỉ tính đúng 1 lần duy nhất ở ```__init__```.
- Runtime Bound Overrides: Né vật cản thời gian thực qua ```set_obstacle_bounds()``` mà không làm refactor hay sập ma trận Cholesky.
- Full Scaling: Chuẩn hóa toàn bộ hệ tọa độ biến giúp ADMM hội tụ cực nhanh về nghiệm vật lý an toàn ```(maxx/bien = 1.000)```.
