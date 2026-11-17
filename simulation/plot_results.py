"""Giai đoạn 4.1 - Biểu đồ quỹ đạo 2D/3D, sai số bám và RMS từ file .npz của run_simulation.py.

Chạy:   python plot_results.py ket_qua.npz
        python plot_results.py ket_qua.npz --out hinh --no_show
Tạo trong thư mục --out (mặc định "hinh"): quy_dao_2D.png, quy_dao_3D.png, sai_so_theo_thoi_gian.png,
ket_qua_rms.txt. Đoạn bay gần vật cản (khoảng cách tới hộp < --near, mặc định 0.20 m) được tô cam.
RMS = sqrt(trung bình(|vị trí thực - vị trí tham chiếu|^2)); tham chiếu là đường số 8 GỐC (chưa uốn né).
"""
import argparse
import os

import numpy as np


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def box_gap(pos, c, h):
    """Khoảng cách từ tâm drone tới mặt hộp: max_i(|p_i - c_i| - h_i)  (< 0 nghĩa là ở trong hộp)."""
    return np.max(np.abs(pos - c) - h, axis=1)


def rms(e):
    return float(np.sqrt(np.mean(np.sum(np.atleast_2d(e) ** 2, axis=-1)))) if len(e) else float("nan")


def analyse(d, near=0.20):
    t, pos, ref = d["t"], d["pos"], d["ref"]
    err = pos - ref
    has_obs = np.isfinite(d["obs_center"]).all()
    near_mask = (box_gap(pos, d["obs_center"], d["obs_half"]) < near) if has_obs else np.zeros(len(t), bool)
    fly = t >= float(d["takeoff"])                      # bỏ giai đoạn cất cánh khi tính RMS "bám số 8"
    res = {
        "RMS 3D, toàn bộ chuyến bay": rms(err),
        "RMS 3D, sau cất cánh": rms(err[fly]),
        "RMS ngang (x,y), sau cất cánh": rms(err[fly][:, :2]),
        "RMS ngang, đoạn bay xa vật cản": rms(err[fly & ~near_mask][:, :2]),
        "RMS ngang, đoạn né vật cản": rms(err[near_mask][:, :2]) if near_mask.any() else float("nan"),
        "Sai số ngang lớn nhất [mm]": 1e3 * float(np.max(np.linalg.norm(err[:, :2], axis=1))),
    }
    if has_obs:
        res["Khoảng cách nhỏ nhất tới hộp [mm]"] = 1e3 * float(box_gap(pos, d["obs_center"], d["obs_half"]).min())
    return res, near_mask, err


def _intervals(t, mask):
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = t[i]
        if not m and start is not None:
            out.append((start, t[i])); start = None
    if start is not None:
        out.append((start, t[-1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="hinh")
    ap.add_argument("--near", type=float, default=0.20)
    ap.add_argument("--no_show", action="store_true")
    a = ap.parse_args()

    import matplotlib
    if a.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = load(a.npz)
    os.makedirs(a.out, exist_ok=True)
    t, pos, ref = d["t"], d["pos"], d["ref"]
    res, near, err = analyse(d, a.near)
    has_obs = np.isfinite(d["obs_center"]).all()
    c, h, m = d["obs_center"], d["obs_half"], float(d["margin"])

    # ---- 2D (nhìn từ trên xuống) ----
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ref[:, 0], ref[:, 1], "--", color="tab:blue", lw=1.6, label="Tham chiếu (số 8)")
    ax.plot(pos[:, 0], pos[:, 1], color="tab:red", lw=1.6, label="Drone thực tế")
    if near.any():
        ax.plot(pos[near, 0], pos[near, 1], ".", color="orange", ms=4, label="Đoạn né vật cản")
    if has_obs:
        ax.add_patch(Rectangle((c[0] - h[0], c[1] - h[1]), 2 * h[0], 2 * h[1], color="gray", alpha=0.8, label="Vật cản"))
        ax.add_patch(Rectangle((c[0] - h[0] - m, c[1] - h[1] - m), 2 * (h[0] + m), 2 * (h[1] + m), fill=False,
                               ls=":", ec="gray", label="Vùng đệm an toàn"))
    ax.plot(*pos[0, :2], "go", label="Điểm xuất phát")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Quỹ đạo bay (nhìn từ trên xuống)"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "quy_dao_2D.png"), dpi=200)

    # ---- 3D ----
    fig = plt.figure(figsize=(8, 6)); ax = fig.add_subplot(111, projection="3d")
    ax.plot(*ref.T, "--", color="tab:blue", lw=1.4, label="Tham chiếu")
    ax.plot(*pos.T, color="tab:red", lw=1.4, label="Drone thực tế")
    if near.any():
        ax.plot(*pos[near].T, ".", color="orange", ms=3, label="Đoạn né vật cản")
    if has_obs:
        x0, x1, y0, y1, z0, z1 = c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], c[2] - h[2], c[2] + h[2]
        X = [x0, x1]; Y = [y0, y1]; Z = [z0, z1]
        for yy in Y:
            for zz in Z:
                ax.plot(X, [yy, yy], [zz, zz], color="gray")
        for xx in X:
            for zz in Z:
                ax.plot([xx, xx], Y, [zz, zz], color="gray")
            for yy in Y:
                ax.plot([xx, xx], [yy, yy], Z, color="gray")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Quỹ đạo bay 3D"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "quy_dao_3D.png"), dpi=200)

    # ---- sai số theo thời gian ----
    fig, ax = plt.subplots(figsize=(9, 4))
    e_xy = 1e3 * np.linalg.norm(err[:, :2], axis=1)
    ax.plot(t, e_xy, color="tab:red", label="Sai số ngang |e_xy| [mm]")
    ax.plot(t, 1e3 * np.abs(err[:, 2]), color="tab:green", lw=1, label="Sai số độ cao |e_z| [mm]")
    for a0, a1 in _intervals(t, near):
        ax.axvspan(a0, a1, color="orange", alpha=0.25)
    if float(d["takeoff"]) > 0:
        ax.axvspan(0, float(d["takeoff"]), color="lightblue", alpha=0.3, label="Cất cánh")
    ax.axhline(res["RMS ngang (x,y), sau cất cánh"] * 1e3, color="k", ls=":",
               label="RMS ngang = %.1f mm" % (1e3 * res["RMS ngang (x,y), sau cất cánh"]))
    ax.set_xlabel("thời gian [s]"); ax.set_ylabel("sai số [mm]"); ax.grid(alpha=.3)
    ax.set_title("Sai số bám quỹ đạo (vùng cam: đang né vật cản)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "sai_so_theo_thoi_gian.png"), dpi=200)

    lines = ["Kết quả sai số bám quỹ đạo (RMS)", "=" * 40]
    for k, v in res.items():
        lines.append("%-38s %10.2f" % (k, v * (1e3 if k.startswith("RMS") else 1)) +
                     (" mm" if k.startswith("RMS") else ""))
    lines.append("")
    lines.append("Ghi chú: sai số lớn ở đoạn né là CHỦ ĐÍCH (đường số 8 gốc xuyên qua vật cản,")
    lines.append("drone phải lệch khỏi đường tham chiếu để không va chạm).")
    txt = "\n".join(lines)
    open(os.path.join(a.out, "ket_qua_rms.txt"), "w", encoding="utf-8").write(txt)
    print(txt)
    print("\nĐã lưu 3 hình + ket_qua_rms.txt vào:", os.path.abspath(a.out))
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
