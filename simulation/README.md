# Mô phỏng 3D vòng kín (Thành viên B)

Điều khiển Crazyflie CF2X trong PyBullet (gym-pybullet-drones) bằng `AccelMPCSolver`: bay hình số 8,
né vật cản, đo sai số bám.

## Cài đặt (Windows, Anaconda Prompt)
```
conda create -n drones python=3.12
conda activate drones
conda install -c conda-forge pybullet
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones && pip install -e . --no-deps && cd ..
pip install -r simulation/requirements-sim.txt
```

## Chạy
| Mục đích | Lệnh |
|---|---|
| Kiểm tra không cần PyBullet | `python check_phase2.py`, `python simulate_linear.py` |
| Kiểm tra dấu 4 motor | `python motor_check.py` |
| Bay số 8 + vật cản | `python run_simulation.py --takeoff 2 --duration 16 --realtime --save ket_qua.npz` |
| Quay video | `python run_simulation.py --takeoff 2 --duration 16 --no_gui --video demo.mp4 --save ket_qua.npz` |
| Biểu đồ + RMS | `python plot_results.py ket_qua.npz --out hinh --no_show` |

## Ghi chú kỹ thuật
- `solver_ext.py`: thay chi phí cuối Qf = Q bằng nghiệm DARE (không sửa `src/admm_mpc_solver.py`).
- `obstacles.py`: biên vật cản + `shape_reference()` uốn đường tham chiếu (chân trời chỉ 62 ms nên
  chỉ dùng `set_obstacle_bounds` thì drone vẫn xuyên vật cản).
- ADMM chạy ngân sách cố định 50-100 vòng/bước; chưa đạt `tol=1e-5` (r_dual lớn, xem báo cáo).
