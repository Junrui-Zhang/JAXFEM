"""
plot_benchmark.py — 读取 benchmark_data/ 生成耗时对比图
=========================================================
生成三张 PDF:
  - benchmark_time_cost.pdf    总耗时对比
  - benchmark_per_iter.pdf     单次迭代耗时 (可扩展性证据)
  - benchmark_iterations.pdf   迭代次数

用法:
    /home/zjr/anaconda3/envs/jaxfem/bin/python3 scripts/plot_benchmark.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 配色 & 样式 (ggsci::npg)
# ============================================================

npg_colors = [
    "#E64B35",  # 红橙
    "#4DBBD5",  # 蓝青
    "#00A087",  # 绿青
    "#3C5488",  # 紫蓝
    "#F39B7F",  # 浅橙
]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
})


# ============================================================
# 数据加载
# ============================================================

def load_data(data_dir="benchmark_data"):
    """从 benchmark_data/ 加载基准测试结果(脚本位于 scripts/)。"""
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        data_dir)
    return {
        "n_elem_list": np.load(os.path.join(base, "n_elem_list.npy")),
        "times": {
            "direct":    np.load(os.path.join(base, "times_direct.npy")),
            "numpy_ebe": np.load(os.path.join(base, "times_numpy_ebe.npy")),
            "jax_ebe":   np.load(os.path.join(base, "times_jax_ebe.npy")),
        },
        "n_iters": {
            "numpy_ebe": np.load(os.path.join(base, "iters_numpy_ebe.npy")),
            "jax_ebe":   np.load(os.path.join(base, "iters_jax_ebe.npy")),
        },
    }


# ============================================================
# 绘图
# ============================================================

def plot_time_cost(n_elem_list, times, save_dir=None):
    """图1: 三种求解器总耗时对比。"""
    fig, ax = plt.subplots(figsize=(3, 2.2), dpi=300)

    ax.plot(n_elem_list, times["direct"],
            marker='o', linestyle='-', linewidth=1.4, alpha=0.9,
            markersize=5, markeredgecolor='white', markeredgewidth=0.5,
            color=npg_colors[0], label="NumPy Direct", zorder=2)

    ax.plot(n_elem_list, times["numpy_ebe"],
            marker='s', linestyle='-', linewidth=1.4, alpha=0.9,
            markersize=5, markeredgecolor='white', markeredgewidth=0.5,
            color=npg_colors[1], label="NumPy EBE-PCG", zorder=2)

    ax.plot(n_elem_list, times["jax_ebe"],
            marker='^', linestyle='-', linewidth=1.4, alpha=0.9,
            markersize=5, markeredgecolor='white', markeredgewidth=0.5,
            color=npg_colors[2], label="JAX EBE-PCG", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of elements $n_{\\mathrm{elem}}$")
    ax.set_ylabel("Time cost (ms)")
    ax.set_xticks(n_elem_list)
    ax.set_xticklabels([str(n) for n in n_elem_list])
    ax.legend(frameon=True, ncol=1, loc="upper left",
              bbox_to_anchor=(0.02, 0.99))

    plt.tight_layout()
    path = os.path.join(save_dir, "benchmark_time_cost.pdf") if save_dir else "benchmark_time_cost.pdf"
    path_png = os.path.join(save_dir, "benchmark_time_cost.png") if save_dir else "benchmark_time_cost.png"
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.savefig(path_png, format="png", bbox_inches="tight", pad_inches=0.01, dpi=600)
    print(f"  {path}")
    plt.close()


def plot_per_iter(n_elem_list, times, n_iters, save_dir=None):
    """图2: 单次迭代耗时 — JAX 可扩展性核心证据。"""
    per_iter_np = times["numpy_ebe"] / np.maximum(n_iters["numpy_ebe"], 1)
    per_iter_jx = times["jax_ebe"] / np.maximum(n_iters["jax_ebe"], 1)

    fig, ax = plt.subplots(figsize=(3, 2.2), dpi=300)

    ax.plot(n_elem_list, per_iter_np,
            marker='s', linestyle='-', linewidth=1.4, alpha=0.9,
            markersize=5, markeredgecolor='white', markeredgewidth=0.5,
            color=npg_colors[1], label="NumPy EBE-PCG", zorder=2)

    ax.plot(n_elem_list, per_iter_jx,
            marker='^', linestyle='-', linewidth=1.4, alpha=0.9,
            markersize=5, markeredgecolor='white', markeredgewidth=0.5,
            color=npg_colors[2], label="JAX EBE-PCG (GPU)", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Number of elements $n_{\\mathrm{elem}}$")
    ax.set_ylabel("Time per iteration (ms)")
    ax.set_xticks(n_elem_list)
    ax.set_xticklabels([str(n) for n in n_elem_list])
    ax.legend(frameon=True, ncol=1, loc="upper right",
              bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    path = os.path.join(save_dir, "benchmark_per_iter.pdf") if save_dir else "benchmark_per_iter.pdf"
    path_png = os.path.join(save_dir, "benchmark_per_iter.png") if save_dir else "benchmark_per_iter.png"
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.savefig(path_png, format="png", bbox_inches="tight", pad_inches=0.01, dpi=600)
    print(f"  {path}")
    plt.close()


def plot_iterations(n_elem_list, n_iters, save_dir=None):
    """图3: CG 迭代次数。"""
    fig, ax = plt.subplots(figsize=(3, 2.2), dpi=300)

    ax.plot(n_elem_list, n_iters["numpy_ebe"],
            marker='s', linestyle='--', linewidth=1.4, alpha=0.9,
            markersize=5, markeredgecolor='white', markeredgewidth=0.5,
            color=npg_colors[1], label="NumPy EBE-PCG", zorder=2)

    ax.plot(n_elem_list, n_iters["jax_ebe"],
            marker='^', linestyle='--', linewidth=1.4, alpha=0.9,
            markersize=5, markeredgecolor='white', markeredgewidth=0.5,
            color=npg_colors[2], label="JAX EBE-PCG", zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of elements $n_{\\mathrm{elem}}$")
    ax.set_ylabel("Iterations")
    ax.set_xticks(n_elem_list)
    ax.set_xticklabels([str(n) for n in n_elem_list])
    ax.legend(frameon=True, ncol=1, loc="upper left",
              bbox_to_anchor=(0.02, 0.99))

    plt.tight_layout()
    path = os.path.join(save_dir, "benchmark_iterations.pdf") if save_dir else "benchmark_iterations.pdf"
    path_png = os.path.join(save_dir, "benchmark_iterations.png") if save_dir else "benchmark_iterations.png"
    plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.savefig(path_png, format="png", bbox_inches="tight", pad_inches=0.01, dpi=600)
    print(f"  {path}")
    plt.close()


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Benchmark 图表生成")
    print("=" * 55)

    data = load_data()
    n_elem_list = data["n_elem_list"]
    times = data["times"]
    n_iters = data["n_iters"]

    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

    plot_time_cost(n_elem_list, times, save_dir)
    plot_per_iter(n_elem_list, times, n_iters, save_dir)
    plot_iterations(n_elem_list, n_iters, save_dir)

    print("=" * 55)
    print("  ✅ 三张图表已生成")
    print("=" * 55)
