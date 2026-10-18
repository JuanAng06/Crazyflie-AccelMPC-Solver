import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

# Tạo ma trận đối xứng dương dạng băng mẫu (kích thước 5x5, độ rộng băng h=1)
# Lưu dạng đối xứng băng theo chuẩn SciPy (lower=True)
ab = np.array([
    [4.0, 5.0, 6.0, 7.0, 8.0],  # Hàng 0: Đường chéo chính
    [1.0, 1.0, 1.0, 1.0, 0.0]   # Hàng 1: Đường chéo phụ thứ 1
])

# 1. Offline Pre-factorization (Cholesky dạng băng)
c_and_lower = cholesky_banded(ab, lower=True)

# 2. Online Solve cho vế phải b = [1, 2, 3, 4, 5]
b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x = cho_solve_banded((c_and_lower, True), b)

print("Môi trường PyCharm đã sẵn sàng!")
print("Nghiệm x = ", x)