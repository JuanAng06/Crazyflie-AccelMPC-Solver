import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cấu hình font và kiểu biểu đồ chuẩn khoa học
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
})

RESULTS_DIR = "results"
OUT_DIR = "hinh"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_warm_vs_cold():
    path = os.path.join(RESULTS_DIR, "benchmark_warm_cold.csv")
    if not os.path.exists(path):
        print(f"[SKIP] Không tìm thấy {path}")
        return
    df = pd.read_csv(path)

    # Lọc các cột nhận diện chế độ warm/cold và ngân sách K
    mode_col = next((c for c in ["mode", "init_mode", "warm_start", "kind"] if c in df.columns), None)
    k_col = next((c for c in ["K", "max_iter", "iters", "param"] if c in df.columns), None)
    med_col = next((c for c in ["median_ms", "solve_median_ms", "time_median_ms"] if c in df.columns), None)
    p99_col = next((c for c in ["p99_ms", "solve_p99_ms", "time_p99_ms"] if c in df.columns), None)
    gap_col = next((c for c in ["mingap_mm", "min_gap_mm"] if c in df.columns), None)
    rms_col = next((c for c in ["rms_far_mm", "rms_mm"] if c in df.columns), None)

    if not (mode_col and k_col and med_col):
        print(f"[WARN] Cấu trúc cột benchmark_warm_cold.csv chưa khớp: {list(df.columns)}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Subplot 1: Thời gian giải (Median & P99)
    ax = axes[0]
    for mode, grp in df.groupby(mode_col):
        grp = grp.sort_values(k_col)
        label_str = "Warm-start" if "warm" in str(mode).lower() or mode is True else "Cold-start"
        ax.plot(grp[k_col], grp[med_col], marker="o", linewidth=2, label=f"{label_str} (Median)")
        if p99_col:
            ax.plot(grp[k_col], grp[p99_col], marker="^", linestyle="--", alpha=0.75, label=f"{label_str} (P99)")
    ax.axhline(2.0, color="red", linestyle=":", linewidth=1.5, label="Ngân sách 500 Hz (2.0 ms)")
    ax.set_title("Thời gian giải trên CPU theo số vòng lặp K")
    ax.set_xlabel("Ngân sách vòng lặp cố định K")
    ax.set_ylabel("Thời gian giải (ms/bước)")
    ax.legend()

    # Subplot 2: Khoảng cách an toàn tối thiểu hoặc RMS
    ax = axes[1]
    target_col = gap_col if gap_col else rms_col
    if target_col:
        for mode, grp in df.groupby(mode_col):
            grp = grp.sort_values(k_col)
            label_str = "Warm-start" if "warm" in str(mode).lower() or mode is True else "Cold-start"
            ax.plot(grp[k_col], grp[target_col], marker="s", linewidth=2, label=label_str)
        if gap_col:
            ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, label="Ngưỡng va chạm (0 mm)")
            ax.set_ylabel("Khoảng cách an toàn tối thiểu (mm)")
            ax.set_title("Độ an toàn né vật cản (Warm vs. Cold)")
        else:
            ax.set_ylabel("Sai số quỹ đạo RMS (mm)")
            ax.set_title("Độ chính xác bám quỹ đạo (Warm vs. Cold)")
        ax.set_xlabel("Ngân sách vòng lặp cố định K")
        ax.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "bench_warm_vs_cold.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Đã xuất {out_path}")


def plot_admm_vs_osqp():
    path = os.path.join(RESULTS_DIR, "benchmark_osqp.csv")
    if not os.path.exists(path):
        print(f"[SKIP] Không tìm thấy {path}")
        return
    df = pd.read_csv(path)

    labels = []
    for _, r in df.iterrows():
        if r["controller"] == "admm":
            labels.append(f"ADMM\n(K={int(r['param'])})")
        else:
            labels.append(f"OSQP\neps={r['param']:.0e}")  # Bỏ dấu ngoặc đơn cho gọn ngang

    x = np.arange(len(labels))
    width = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Subplot 1: Mean, Median, P99 (Thể hiện tính tất định của ADMM so với P99 vọt của OSQP)
    ax = axes[0]
    ax.bar(x - width, df["mean_ms"], width, label="Mean", color="#4C72B0")
    ax.bar(x, df["median_ms"], width, label="Median", color="#55A868")
    ax.bar(x + width, df["p99_ms"], width, label="P99 (Đuôi trễ)", color="#C44E52")
    ax.axhline(2.0, color="black", linestyle="--", linewidth=1.2, label="Chu kỳ 500 Hz (2.0 ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Thời gian giải (ms/bước)")
    ax.set_title("Phân bố độ trễ: ADMM-MPC vs. OSQP (5,000 bước)")
    ax.legend()

    # Subplot 2: Đánh đổi chất lượng quỹ đạo (RMS xa vật cản) & Khoảng cách an toàn (mingap)
    ax = axes[1]
    ax2 = ax.twinx()
    ax2.set_ylim(-60, 150)  # Mở rộng trần trục phải để legend không đè lên cột
    ax2.grid(False)

    p1 = ax.bar(x - 0.18, df["rms_far_mm"], 0.35, color="#8172B3", label="RMS xa vật cản (mm)")
    p2 = ax2.bar(x + 0.18, df["mingap_mm"], 0.35, color="#CCB974", label="Khoảng cách an toàn mingap (mm)")
    ax2.axhline(0.0, color="red", linestyle=":", linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sai số RMS xa vật cản (mm)", color="#4B3B75")
    ax2.set_ylabel("Khoảng cách an toàn tối thiểu (mm)", color="#7A6A28")
    ax.set_title("Chất lượng bám quỹ đạo & An toàn né vật cản")

    lines = [p1, p2]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "bench_admm_vs_osqp.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Đã xuất {out_path}")


def plot_scaling_H():
    path = os.path.join(RESULTS_DIR, "benchmark_scaling.csv")
    if not os.path.exists(path):
        print(f"[SKIP] Không tìm thấy {path}")
        return
    df = pd.read_csv(path).sort_values("H")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Subplot 1: Thời gian giải online (log-log)
    ax = axes[0]
    ax.loglog(df["Nvar"], df["median_lo_ms"], marker="o", linewidth=2, label="K = 10 (Median)")
    ax.loglog(df["Nvar"], df["median_hi_ms"], marker="s", linewidth=2, label="K = 50 (Median)")
    for _, r in df.iterrows():
        ax.annotate(f"H={int(r['H'])}", (r["Nvar"], r["median_hi_ms"]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xlabel("Số biến quyết định N_var (Log)")
    ax.set_ylabel("Thời gian giải online (ms, Log)")
    ax.set_title("Khả năng mở rộng thời gian giải O(N_var)")
    ax.legend()

    # Subplot 2: Phân rã Cholesky Offline
    ax = axes[1]
    ax.plot(df["H"], df["offline_chol_ms"], marker="^", color="#D95F02", linewidth=2)
    ax.set_xticks(df["H"])
    ax.set_xlabel("Chân trời dự báo H")
    ax.set_ylabel("Thời gian Cholesky offline (ms)")
    ax.set_title("Chi phí tiền phân rã Cholesky (1 lần)")

    # Subplot 3: Bộ nhớ nhân tử L (MB) & Độ rộng băng h=20
    ax = axes[2]
    ax.bar(df["H"].astype(str), df["L_MB"], color="#1B9E77", width=0.5)
    for i, v in enumerate(df["L_MB"]):
        ax.text(i, v + 0.008, f"{v:.3f} MB", ha="center", fontsize=8.5)
    ax.set_ylim(0, max(df["L_MB"]) * 1.18)
    ax.set_xlabel("Chân trời dự báo H (Độ rộng băng h = 20)")
    ax.set_ylabel("Dung lượng nhân tử L (MB)")
    ax.set_title("Bộ nhớ lưu trữ nhân tử Cholesky dạng băng")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "bench_scaling_H.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Đã xuất {out_path}")


def plot_paper_comparison():
    # Đối chiếu chi phí tính toán chuẩn hóa: us / (vòng lặp * bước dự báo)
    platforms = [
        "CPU Python (Nhóm)\n[Float64, H=31]",
        "AccelMPC FPGA Staged\n[Artix-7, 32-bit Fixed]",
        "AccelMPC FPGA Full-Sparse\n[Artix-7, 32-bit Fixed]",
        "TinyMPC MCU\n[STM32 Embedded]"
    ]
    costs = [2.87, 3.33, 3.69, 52.30]
    colors = ["#2B5C8F", "#2CA02C", "#66C2A5", "#D62728"]

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    bars = ax.bar(platforms, costs, color=colors, width=0.48)
    ax.set_yscale("log")
    ax.set_ylim(1, 120)

    for bar, val in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15,
                f"{val:.2f} µs", ha="center", va="bottom", fontweight="bold", fontsize=9.5)

    ax.set_ylabel("Chi phí chuẩn hóa: µs / (vòng · bước) [Thang Log]")
    ax.set_title("Đối chiếu hiệu năng tính toán chuẩn hóa với bài báo AccelMPC")
    ax.text(
        0.5, -0.22,
        "Lưu ý nền tảng: CPU máy tính để bàn (Float64) có xung nhịp/băng thông vượt trội so với FPGA công suất thấp\n"
        "và vi điều khiển STM32 gắn trên drone Crazyflie 35g. Mức tăng tốc 15.6x của bài báo là FPGA so với MCU ở K=10.",
        transform=ax.transAxes, ha="center", va="top", fontsize=8.5, style="italic", color="#333333"
    )

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "bench_paper_comparison.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Đã xuất {out_path}")


if __name__ == "__main__":
    plot_warm_vs_cold()
    plot_admm_vs_osqp()
    plot_scaling_H()
    plot_paper_comparison()
    print("Hoàn tất xuất bộ biểu đồ benchmark vào thư mục hinh/!")