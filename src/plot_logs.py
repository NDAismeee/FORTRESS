import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_efficiency(csv_path, out_dir="figs"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # chỉ dùng eval rows (có HR@20)
    df = df[df["phase"] == "eval"].copy()
    df = df.dropna(subset=["HR@20"])

    # ===============================
    # Plot 1: HR@20 vs Communication
    # ===============================
    x_comm = df["cum_bytes_total"] / 1e6  # MB
    y_hr = df["HR@20"]

    plt.figure(figsize=(6, 4))
    plt.plot(x_comm, y_hr, marker="o")
    plt.xlabel("Cumulative Communication (MB)")
    plt.ylabel("HR@20")
    plt.title("Recommendation Accuracy vs Communication Cost")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hr_vs_communication.png"), dpi=300)
    plt.close()

    # ===============================
    # Plot 2: HR@20 vs Training Time
    # ===============================
    x_time = df["cum_time_sec"] / 60.0  # minutes

    plt.figure(figsize=(6, 4))
    plt.plot(x_time, y_hr, marker="o")
    plt.xlabel("Cumulative Training Time (minutes)")
    plt.ylabel("HR@20")
    plt.title("Recommendation Accuracy vs Training Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hr_vs_time.png"), dpi=300)
    plt.close()

    print(f"[OK] Efficiency plots saved to `{out_dir}/`")


if __name__ == "__main__":
    plot_efficiency("logs/train_log_latest.csv")  # đổi tên file CSV cho đúng
