"""
benchmark_data.py — 基准数据读取 + NumPy 耗时估算 + 静态基准曲线图
====================================================================
数据来源:benchmark.py 生成于 benchmark_data/*.npy
  (n_elem_list, times_direct, times_numpy_ebe, times_jax_ebe)

estimate_numpy_ms:对 log10(t) ~ log10(n) 线性拟合外推
  (实测斜率 ≈ 2.97:PCG 迭代 ~n^1.7 × 每步代价 ~n)。
"""

import os

import numpy as np

from webapp import config

# 目录结构:JAXFEM/product/webapp/services/benchmark_data.py
# benchmark_data/*.npy 位于 JAXFEM 根(核心科研数据,不在 product/ 内)
_CORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_DATA_DIR = os.path.join(_CORE_ROOT, 'benchmark_data')


def load_benchmark() -> dict:
    """读取 benchmark_data/*.npy;缺失文件时返回 None。"""
    names = ['n_elem_list.npy', 'times_direct.npy',
             'times_numpy_ebe.npy', 'times_jax_ebe.npy']
    if not all(os.path.exists(os.path.join(_DATA_DIR, n)) for n in names):
        return None
    return {
        'n_elem': np.load(os.path.join(_DATA_DIR, names[0])),
        'direct': np.load(os.path.join(_DATA_DIR, names[1])),
        'numpy': np.load(os.path.join(_DATA_DIR, names[2])),
        'jax': np.load(os.path.join(_DATA_DIR, names[3])),
    }


_FIT = {}


def estimate_numpy_ms(n_elem: int) -> float:
    """NumPy EBE-PCG 耗时估算(ms),基于基准数据 log-log 拟合。"""
    if not _FIT:
        data = load_benchmark()
        if data is None:
            _FIT['k'], _FIT['b'] = 2.97, -3.656   # 回退:来自实测拟合
        else:
            x = np.log10(data['n_elem'])
            y = np.log10(np.maximum(data['numpy'], 1e-3))
            _FIT['k'], _FIT['b'] = np.polyfit(x, y, 1)
    return float(10 ** (_FIT['k'] * np.log10(max(n_elem, 1)) + _FIT['b']))


def build_benchmark_figure():
    """静态基准曲线图(3 条 log-log 折线),启动时生成一次。"""
    import plotly.graph_objects as go
    from webapp.services.charts import apply_chrome, line_trace

    fig = go.Figure()
    data = load_benchmark()

    if data is None:
        fig.add_annotation(
            text='暂无基准数据(benchmark_data/*.npy),请先运行 benchmark.py',
            showarrow=False, font=dict(color=config.CHART_INK2, size=13))
        return apply_chrome(fig, '单元数 n_elem', '耗时 (ms)',
                            x_log=True, y_log=True)

    fig.add_trace(line_trace(data['n_elem'], data['direct'], 'NumPy 直接解',
                             config.COL_FEM))
    fig.add_trace(line_trace(data['n_elem'], data['numpy'], 'NumPy EBE-PCG',
                             config.COL_THEORY))
    fig.add_trace(line_trace(data['n_elem'], data['jax'], 'JAX EBE-PCG (GPU)',
                             config.COL_JAX))
    return apply_chrome(fig, '单元数 n_elem', '耗时 (ms)',
                        x_log=True, y_log=True, height=340)
