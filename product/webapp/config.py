"""
config.py — 全局配置常量
==========================
参数上限来自实测耗时(见 README_APP.md「已知限制」):
  - NumPy EBE n=500 约 15s,n=2000 需约 36 分钟 → NumPy 后端硬上限 500
  - 直接解 n=2000 的 K_global 约占 1.15GB 内存 → n_elem 总上限 2000
"""

# ---- 参数边界 ----
N_ELEM_MIN = 10
N_ELEM_MAX = 2000
NUMPY_MAX_ELEM = 500            # NumPy 后端硬上限
NUMPY_COMPARE_MAX_ELEM = 400    # 双后端对比实跑上限,超出用基准数据拟合估算
TOL_MIN, TOL_MAX = 1e-12, 1e-4
MAX_ITER_MIN, MAX_ITER_MAX = 10, 200_000
SCALE_MIN, SCALE_MAX, SCALE_STEP = 1.0, 2000.0, 10.0

# ---- UI 默认值(单位:GPa / kN / mm,与 SI 的换算在 solver_service.parse_params) ----
DEFAULT_UI = dict(
    L=10.0, n_elem=100, b_width=200.0, b_height=300.0,
    E=210.0, nu=0.3, rho=7850.0, P=-10.0,
    solver='jax', tol=1e-8, max_iter='',
    field='uy', scale=200.0, theory=True,
)

# ---- 3D 窗口 ----
WINDOW_3D = (1000, 640)

# ---- 图表配色(参考调色板固定槽位,已通过验证器) ----
COL_FEM = '#2a78d6'          # slot 1 蓝 — FEM 数值解 / 直接解
COL_THEORY = '#eb6834'       # slot 2 橙 — 理论解 / NumPy EBE
COL_JAX = '#1baf7a'          # slot 3 青 — JAX EBE
COL_SUCCESS = '#0ca30c'
COL_WARNING = '#fab219'
COL_ERROR = '#d03b3b'

# ---- 图表 chrome(浅色) ----
CHART_SURFACE = '#fcfcfb'
CHART_INK = '#0b0b0b'
CHART_INK2 = '#52514e'
CHART_MUTED = '#898781'
CHART_GRID = '#e1e0d9'
CHART_AXIS = '#c3c2b7'

# ---- 缓存 ----
RESULT_CACHE_SIZE = 8
EXPORT_CACHE_SIZE = 12

# ---- 应用 ----
APP_TITLE = 'JAXFEM · 简支梁 EBE-PCG GPU 并行有限元演示平台'
