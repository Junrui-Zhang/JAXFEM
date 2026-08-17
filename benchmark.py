"""
benchmark.py — 三种求解器耗时基准测试
=======================================
循环不同单元数，测试:
  - NumPy 直接求解 (K^-1 F)
  - NumPy EBE-PCG (串行)
  - JAX EBE-PCG   (vmap GPU 并行)

结果保存至 benchmark_data/，供 plot_benchmark.py 读取绘图。

用法:
    /home/zjr/anaconda3/envs/jaxfem/bin/python3 benchmark.py
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ansys.ansys_parser import build_native_model
from post import direct_solve, solve


def run_benchmark(n_elem_list, L=100.0):
    """
    对每个单元数运行三种求解器，记录耗时。

    返回:
        times: dict, keys = ['direct', 'numpy_ebe', 'jax_ebe']
               每个是 (n_configs,) ndarray (ms)
        n_iters: dict, 记录迭代次数
    """
    n_configs = len(n_elem_list)
    times = {
        "direct":    np.zeros(n_configs),
        "numpy_ebe": np.zeros(n_configs),
        "jax_ebe":   np.zeros(n_configs),
    }
    n_iters = {
        "numpy_ebe": np.zeros(n_configs, dtype=int),
        "jax_ebe":   np.zeros(n_configs, dtype=int),
    }

    for i, n_elem in enumerate(n_elem_list):
        print(f"\n{'='*55}")
        print(f"  n_elem = {n_elem}")
        print(f"{'='*55}")

        model = build_native_model(L=L, n_elem=n_elem)
        n_dofs = model["n_dofs"]
        print(f"  DOFs = {n_dofs}")

        # ---- (1) NumPy 直接求解 ----
        print("  [1/3] NumPy 直接求解...")
        t0 = time.perf_counter()
        u_dir = direct_solve(model)
        t_dir = (time.perf_counter() - t0) * 1000  # ms
        times["direct"][i] = t_dir
        print(f"        耗时: {t_dir:.2f} ms")

        # ---- (2) NumPy EBE-PCG ----
        print("  [2/3] NumPy EBE-PCG...")
        u_np, n_np, info_np = solve(model, solver="numpy", verbose=False)
        times["numpy_ebe"][i] = info_np["time_ms"]
        n_iters["numpy_ebe"][i] = n_np
        err_np = np.linalg.norm(u_np - u_dir) / np.linalg.norm(u_dir) * 100
        print(f"        耗时: {info_np['time_ms']:.1f} ms, "
              f"iter={n_np}, err={err_np:.4f}%")

        # ---- (3) JAX EBE-PCG ----
        print("  [3/3] JAX EBE-PCG...")
        u_jx, n_jx, info_jx = solve(model, solver="jax", verbose=False)
        times["jax_ebe"][i] = info_jx["time_ms"]
        n_iters["jax_ebe"][i] = n_jx
        err_jx = np.linalg.norm(u_jx - u_dir) / np.linalg.norm(u_dir) * 100
        print(f"        耗时: {info_jx['time_ms']:.1f} ms, "
              f"iter={n_jx}, err={err_jx:.4f}%")

    return times, n_iters


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  BEAM4 简支梁 — 三求解器 Benchmark")
    print("=" * 55)

    n_elem_list = np.array([50, 100, 200, 500, 1000, 2000])

    times, n_iters = run_benchmark(n_elem_list, L=100.0)

    # ---- 打印表格 ----
    print("\n" + "=" * 80)
    print("  结果汇总")
    print("=" * 80)
    print(f"  {'n_elem':>7} {'DOFs':>6} "
          f"{'Direct(ms)':>11} {'NP-EBE(ms)':>11} {'JAX-EBE(ms)':>12} "
          f"{'NP-iter':>8} {'JAX-iter':>9}")
    print(f"  {'-'*70}")
    for i, n in enumerate(n_elem_list):
        n_dofs = (n + 1) * 6
        print(f"  {n:>7} {n_dofs:>6} "
              f"{times['direct'][i]:>11.2f} {times['numpy_ebe'][i]:>11.1f} "
              f"{times['jax_ebe'][i]:>12.1f} "
              f"{n_iters['numpy_ebe'][i]:>8} {n_iters['jax_ebe'][i]:>9}")

    # ---- 保存数据 ----
    data_dir = os.path.join(os.path.dirname(__file__), "benchmark_data")
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, "n_elem_list.npy"), n_elem_list)
    np.save(os.path.join(data_dir, "times_direct.npy"), times["direct"])
    np.save(os.path.join(data_dir, "times_numpy_ebe.npy"), times["numpy_ebe"])
    np.save(os.path.join(data_dir, "times_jax_ebe.npy"), times["jax_ebe"])
    np.save(os.path.join(data_dir, "iters_numpy_ebe.npy"), n_iters["numpy_ebe"])
    np.save(os.path.join(data_dir, "iters_jax_ebe.npy"), n_iters["jax_ebe"])
    print(f"\n  数据已保存至: {data_dir}/")
    print(f"  运行 plot_benchmark.py 生成图表")
    print("=" * 55)
