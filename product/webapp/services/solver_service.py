"""
solver_service.py — 求解服务层
================================
参数校验(UI 单位 → SI)、GPU 检测、run_case 求解管线、结果缓存、曲线数据。

零侵入:复用项目既有模块
  - ansys.ansys_parser.build_native_model()   模型生成
  - post.solve() / direct_solve() / analytic_deflection()   求解与基准
UI 单位约定:GPa / kN / mm(换算为 Pa / N / m 后进入 CaseParams)。
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ansys.ansys_parser import build_native_model
from post import solve, direct_solve, analytic_deflection

from webapp import config


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CaseParams:
    """一次求解的全部参数(SI 单位)。"""
    L: float                    # 跨度 (m)
    n_elem: int                 # 单元数(偶数)
    b_width: float              # 截面宽 (m)
    b_height: float             # 截面高 (m)
    E: float                    # 弹性模量 (Pa)
    nu: float                   # 泊松比
    rho: float                  # 密度 (kg/m³)
    P: float                    # 跨中集中力 (N,负值向下)
    solver: str                 # 'jax' | 'numpy'
    tol: float                  # 收敛容差
    max_iter: Optional[int]     # None = 自适应

    def cache_tuple(self) -> tuple:
        """结果缓存键(只含影响求解结果的字段)。"""
        return (self.L, self.n_elem, self.b_width, self.b_height,
                self.E, self.nu, self.rho, self.P, self.solver,
                self.tol, self.max_iter)


@dataclass
class CaseResult:
    """一次求解的完整结果(供回调渲染)。"""
    params: CaseParams
    model: dict
    u: np.ndarray
    n_iter: int
    info: dict
    u_direct: np.ndarray
    theory: float                       # 理论跨中挠度 (m, 正数)
    err_vs_theory_pct: float
    err_vs_direct_pct: float
    time_direct_ms: float
    time_wall_ms: float                 # 求解壁钟耗时(含 JIT 编译)
    converged: bool
    jax_compare: dict                   # {'time_ms','n_iter','live'}
    numpy_compare: dict                 # {'time_ms','n_iter','live'}


# ============================================================
# 参数校验与解析
# ============================================================

def _num(raw: dict, key: str, label: str, errors: list, default: float) -> float:
    """提取数值型输入,失败时登记中文错误并返回默认值。"""
    v = raw.get(key, default)
    if v is None or v == '' or (isinstance(v, str) and not v.strip()):
        if default is not None:
            return default
        errors.append(f"{label}:不能为空")
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        errors.append(f"{label}:请输入有效数字(当前为 {v!r})")
        return default if default is not None else 0.0


def parse_params(raw: dict) -> CaseParams:
    """
    UI 原始值 → CaseParams(SI),非法参数抛 ValueError(中文信息)。

    接受一个包含全部参数键的 dict(如 dash State 集合):
      L(m), n_elem, b_width(mm), b_height(mm), E(GPa), nu, rho(kg/m³),
      P(kN), solver('jax'|'numpy'), tol, max_iter(空串=自动)
    """
    errors = []

    L = _num(raw, 'L', '跨度 L', errors, config.DEFAULT_UI['L'])
    b_width = _num(raw, 'b_width', '截面宽', errors, config.DEFAULT_UI['b_width'])
    b_height = _num(raw, 'b_height', '截面高', errors, config.DEFAULT_UI['b_height'])
    E_gpa = _num(raw, 'E', '弹性模量 E', errors, config.DEFAULT_UI['E'])
    nu = _num(raw, 'nu', '泊松比 ν', errors, config.DEFAULT_UI['nu'])
    rho = _num(raw, 'rho', '密度 ρ', errors, config.DEFAULT_UI['rho'])
    P_kn = _num(raw, 'P', '集中力 P', errors, config.DEFAULT_UI['P'])
    tol = _num(raw, 'tol', '容差 tol', errors, config.DEFAULT_UI['tol'])

    try:
        n_elem = int(float(raw.get('n_elem', config.DEFAULT_UI['n_elem'])))
    except (TypeError, ValueError):
        n_elem = 0
        errors.append("单元数 n_elem:请输入整数")

    max_iter = None
    max_iter_raw = (raw.get('max_iter') or '')   # dcc.Input 空值为 None
    max_iter_raw = str(max_iter_raw).strip()
    if max_iter_raw and max_iter_raw.lower() != 'none':
        try:
            max_iter = int(float(max_iter_raw))
        except ValueError:
            errors.append(f"最大迭代数:请输入整数或留空(自动),当前为 {max_iter_raw!r}")

    solver = str(raw.get('solver', config.DEFAULT_UI['solver']))

    # ---- 语义校验 ----
    if L <= 0:
        errors.append(f"跨度 L 必须 > 0(当前 {L:g} m)")
    if b_width <= 0:
        errors.append(f"截面宽必须 > 0(当前 {b_width:g} mm)")
    if b_height <= 0:
        errors.append(f"截面高必须 > 0(当前 {b_height:g} mm)")
    if E_gpa <= 0:
        errors.append(f"弹性模量 E 必须 > 0(当前 {E_gpa:g} GPa)")
    if not (0.0 < nu < 0.5):
        errors.append(f"泊松比 ν 必须在 (0, 0.5) 之间(当前 {nu:g})")
    if rho < 0:
        errors.append(f"密度 ρ 不能为负(当前 {rho:g} kg/m³)")
    if P_kn == 0:
        errors.append("集中力 P 不能为 0(零荷载无解)")
    if not (config.TOL_MIN <= tol <= config.TOL_MAX):
        errors.append(f"容差 tol 必须在 [{config.TOL_MIN:g}, {config.TOL_MAX:g}] 之间(当前 {tol:g})")
    if max_iter is not None and not (config.MAX_ITER_MIN <= max_iter <= config.MAX_ITER_MAX):
        errors.append(f"最大迭代数必须在 [{config.MAX_ITER_MIN}, {config.MAX_ITER_MAX}] 之间(当前 {max_iter})")
    if not (config.N_ELEM_MIN <= n_elem <= config.N_ELEM_MAX):
        errors.append(f"单元数必须在 [{config.N_ELEM_MIN}, {config.N_ELEM_MAX}] 之间(当前 {n_elem})")
    if n_elem % 2 != 0:
        errors.append("单元数必须为偶数(荷载位于跨中节点,奇数时理论值无法对照)")
    if solver not in ('jax', 'numpy'):
        errors.append(f"未知求解器 {solver!r}(可选 jax / numpy)")
    if solver == 'numpy' and n_elem > config.NUMPY_MAX_ELEM:
        est = estimate_numpy_ms(n_elem)
        errors.append(
            f"NumPy 串行后端不支持 n_elem > {config.NUMPY_MAX_ELEM} "
            f"(n={n_elem} 预计耗时 {_fmt_ms(est)},请改用 JAX 后端)"
        )

    if errors:
        raise ValueError("；".join(errors))

    return CaseParams(
        L=L, n_elem=n_elem, b_width=b_width / 1e3, b_height=b_height / 1e3,
        E=E_gpa * 1e9, nu=nu, rho=rho, P=P_kn * 1e3,
        solver=solver, tol=tol, max_iter=max_iter,
    )


def parse_scale(raw: dict) -> float:
    """解析变形放大系数(显示参数,非法时静默回退默认值)。"""
    try:
        s = float(raw.get('scale', config.DEFAULT_UI['scale']))
    except (TypeError, ValueError):
        return config.DEFAULT_UI['scale']
    if not (config.SCALE_MIN <= s <= config.SCALE_MAX):
        return config.DEFAULT_UI['scale']
    return s


# ============================================================
# 求解管线
# ============================================================

def _adaptive_max_iter(n_elem: int) -> int:
    """
    EBE-PCG 迭代上限(webapp 版,余量比 post.estimate_iterations 更足)。

    实测 benchmark 迭代数:200→763、500→4376、1000→17845、2000→57641,
    post.py 的 0.12·n^1.72 + 500 在 n≈600 时余量不足(上限 7709 < 实际 8380),
    此处改为 +max(500, 25%·base):收敛前不受影响(提前 break),停滞时不会误触顶。
    """
    base = int(0.12 * n_elem ** 1.72)
    return base + max(500, base // 4)


def run_case(p: CaseParams) -> CaseResult:
    """
    核心管线:建模型 → 直接解(基准) → EBE-PCG 求解 → 双后端对比/误差。

    注意:本函数不做参数校验(由 parse_params 负责),且必须在
    SOLVE_LOCK 保护下调用(VTK 渲染与求解串行化)。
    """
    model = build_native_model(L=p.L, n_elem=p.n_elem, b_width=p.b_width,
                               b_height=p.b_height, E=p.E, nu=p.nu,
                               rho=p.rho, P=p.P)
    theory = analytic_deflection(model)

    # ---- 直接解(参考基准) ----
    t0 = time.perf_counter()
    u_direct = direct_solve(model)
    time_direct_ms = (time.perf_counter() - t0) * 1e3

    # ---- EBE-PCG 求解(壁钟含 JIT 编译) ----
    max_iter = p.max_iter if p.max_iter is not None else _adaptive_max_iter(p.n_elem)
    t0 = time.perf_counter()
    u, n_iter, info = solve(model, solver=p.solver, tol=p.tol,
                            max_iter=max_iter, verbose=False)
    time_wall_ms = (time.perf_counter() - t0) * 1e3

    mid_dof = model["_mid_node"] * model["dof_per_node"] + 1  # UY
    u_mid = float(u[mid_dof])
    err_vs_theory_pct = abs(abs(u_mid) - theory) / theory * 100
    err_vs_direct_pct = float(
        np.linalg.norm(u - u_direct) / np.linalg.norm(u_direct) * 100)
    converged = n_iter < info['max_iter']

    # ---- JAX 对比数据(无论所选后端,实跑代价小) ----
    if p.solver == 'jax':
        jax_compare = {'time_ms': info['time_ms'], 'n_iter': n_iter, 'live': True}
    else:
        _, it_jax, info_jax = solve(model, solver='jax', tol=p.tol,
                                    max_iter=max_iter, verbose=False)
        jax_compare = {'time_ms': info_jax['time_ms'], 'n_iter': it_jax, 'live': True}

    # ---- NumPy 对比数据(大模型用基准数据拟合估算) ----
    if p.n_elem <= config.NUMPY_COMPARE_MAX_ELEM:
        if p.solver == 'numpy':
            numpy_compare = {'time_ms': info['time_ms'], 'n_iter': n_iter, 'live': True}
        else:
            _, it_np, info_np = solve(model, solver='numpy', tol=p.tol,
                                      max_iter=max_iter, verbose=False)
            numpy_compare = {'time_ms': info_np['time_ms'], 'n_iter': it_np, 'live': True}
    else:
        numpy_compare = {'time_ms': estimate_numpy_ms(p.n_elem),
                         'n_iter': None, 'live': False}

    return CaseResult(
        params=p, model=model, u=u, n_iter=n_iter, info=info,
        u_direct=u_direct, theory=theory,
        err_vs_theory_pct=err_vs_theory_pct,
        err_vs_direct_pct=err_vs_direct_pct,
        time_direct_ms=time_direct_ms, time_wall_ms=time_wall_ms,
        converged=converged, jax_compare=jax_compare,
        numpy_compare=numpy_compare,
    )


# ============================================================
# 曲线数据
# ============================================================

def node_curves(model: dict, u) -> dict:
    """节点位移曲线:FEM 解的 UY(mm) 与 ROTZ(rad) 沿梁分布。"""
    d = model['dof_per_node']           # 6
    u = np.asarray(u)
    return {
        'x': model['nodes'][:, 0],
        'uy_mm': u[1::d] * 1e3,
        'rotz': u[5::d],
    }


def theory_curves(model: dict, n_pts: int = 101) -> dict:
    """
    理论曲线(带荷载符号,与 FEM 同号):
      左半  δ(x) = P·x·(3L²−4x²)/(48EI),  θ(x) = P·(L²−4x²)/(16EI)
      右半按对称镜像。
    ROTZ 符号与 BEAM4 单元约定一致(已验证)。
    """
    L = model['_L']
    E = model['materials']['E']
    I = model['sections']['Iz']
    P = model['loads'][0][2]
    x = np.linspace(0.0, L, n_pts)
    xr = L - x
    uy = np.where(x <= L / 2,
                  P * x * (3 * L**2 - 4 * x**2),
                  P * xr * (3 * L**2 - 4 * xr**2)) / (48 * E * I)
    rotz = np.where(x <= L / 2,
                    P * (L**2 - 4 * x**2),
                    -P * (L**2 - 4 * xr**2)) / (16 * E * I)
    return {'x': x, 'uy_mm': uy * 1e3, 'rotz': rotz}


# ============================================================
# GPU 检测
# ============================================================

def detect_gpu() -> dict:
    """启动时检测 GPU(供徽章显示)。"""
    try:
        import jax
        devices = jax.devices('gpu')
        if devices:
            name = str(devices[0]).replace('CudaDevice', 'GPU')
            return {'available': True, 'name': name}
        return {'available': False, 'name': 'CPU'}
    except Exception as exc:  # jax 导入失败等
        return {'available': False, 'name': f'CPU(检测失败: {exc})'}


# ============================================================
# 结果缓存 + 全局锁
# ============================================================

class ResultCache:
    """按参数缓存 CaseResult:切换显示场/放大系数时零求解秒回。"""

    def __init__(self, maxsize: int = config.RESULT_CACHE_SIZE):
        self._data = {}
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def key(self, params: CaseParams) -> str:
        blob = json.dumps(params.cache_tuple(), sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[CaseResult]:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, result: CaseResult) -> None:
        with self._lock:
            if len(self._data) >= self._maxsize:
                self._data.pop(next(iter(self._data)))   # 清最旧
            self._data[key] = result


RESULT_CACHE = ResultCache()
# 串行化「求解 + VTK 渲染」,防连点并发;RLock:回调1持锁期间可再进入导出函数
SOLVE_LOCK = threading.RLock()


# ============================================================
# 工具
# ============================================================

def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1e3:.2f} s"
    return f"{ms:.1f} ms"


def estimate_numpy_ms(n_elem: int) -> float:
    """NumPy EBE 耗时估算(基于 benchmark 数据的 log-log 拟合,延迟导入避免环)。"""
    from webapp.services.benchmark_data import estimate_numpy_ms as _est
    return _est(n_elem)
