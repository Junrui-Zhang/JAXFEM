"""
export3d.py — 无头 3D 云图导出(pyvista → HTML 字符串)
======================================================
复用 post.Post 的网格构建逻辑(_build_solid_mesh / computeMisesStress),
但自建 Plotter(off_screen=True)并直接导出 HTML 字符串,
供 html.Iframe(srcDoc=...) 嵌入,不写任何临时文件、不弹窗口。

要点:
  - 本机有 DISPLAY,不显式 off_screen=True 会在回调线程弹 VTK 窗口
  - 导出在 SOLVE_LOCK 保护下执行(RLock,与求解串行化)
  - 3D 场景内文字一律英文(VTK 默认字体无中文字形)
"""

import threading
from typing import Optional

import numpy as np

from webapp import config

# 显示场定义
FIELD_SCALARS = {
    'uy':    'UY (mm)',
    'usum':  'USUM (mm)',
    'mises': 'von Mises (MPa)',
}
FIELD_TITLES = {
    'uy':    'Vertical Displacement UY',
    'usum':  'Total Displacement USUM',
    'mises': 'von Mises Stress',
}


class ExportCache:
    """按 (结果缓存键, 显示场, 放大系数) 缓存导出的 HTML,切换显示秒回。"""

    def __init__(self, maxsize: int = config.EXPORT_CACHE_SIZE):
        self._data = {}
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[str]:
        with self._lock:
            return self._data.get(key)

    def put(self, key: tuple, html_str: str) -> None:
        with self._lock:
            if len(self._data) >= self._maxsize:
                self._data.pop(next(iter(self._data)))
            self._data[key] = html_str


EXPORT_CACHE = ExportCache()


def export_solid_html(model: dict, u, field: str = 'uy',
                      scale: float = 1.0,
                      window_size: tuple = config.WINDOW_3D) -> str:
    """
    渲染变形 3D 实体梁(六面体扫掠)并返回完整 HTML 字符串。

    参数:
        model:  build_native_model() 输出
        u:      (n_dofs,) 位移向量
        field:  'uy' | 'usum' | 'mises'
        scale:  变形放大系数
    """
    from webapp.services.solver_service import SOLVE_LOCK
    with SOLVE_LOCK:
        return _export_unlocked(model, u, field, scale, window_size)


def _export_unlocked(model: dict, u, field: str, scale: float,
                     window_size: tuple) -> str:
    import pyvista as pv
    from post import Post

    if field not in FIELD_SCALARS:
        raise ValueError(f"未知显示场 {field!r}(可选 {list(FIELD_SCALARS)})")
    if scale <= 0:
        raise ValueError(f"放大系数必须 > 0(当前 {scale:g})")

    post = Post(model)
    d = model['dof_per_node']
    u = np.asarray(u)

    # ---- 变形实体网格(复用 Post 的截面扫掠) ----
    mesh, n_per_node = post._build_solid_mesh(u, scale)

    # ---- 标量场 ----
    if field == 'uy':
        vals = np.repeat(u[1::d] * 1e3, n_per_node)
    elif field == 'usum':
        ux, uy = u[0::d], u[1::d]
        vals = np.repeat(np.sqrt(ux ** 2 + uy ** 2) * 1e3, n_per_node)
    else:  # mises
        sigma = post.computeMisesStress(u)
        vals = np.repeat(sigma / 1e6, n_per_node)

    name = FIELD_SCALARS[field]
    mesh.point_data[name] = vals
    mesh.point_data.active_scalars_name = name

    # ---- 渲染 ----
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    vmin, vmax = float(vals.min()), float(vals.max())
    dargs = dict(
        cmap='coolwarm', show_scalar_bar=True,
        interpolate_before_map=True, clim=[vmin, vmax],
        scalar_bar_args=dict(title=name, color='k', n_labels=7, fmt='%.4f',
                             label_font_size=10, title_font_size=12),
    )
    show_edges = model['n_elem'] <= 300   # 大网格关边线,避免噪点
    pl.add_mesh(mesh, **dargs, show_edges=show_edges)

    # 未变形虚影(对照)
    mesh_undef, _ = post._build_solid_mesh()
    pl.add_mesh(mesh_undef, color='lightgray', opacity=0.2, show_edges=False)

    L_span = model['_L']
    pl.background_color = 'white'
    pl.camera_position = [(L_span / 2, -L_span * 0.6, L_span * 0.3),
                          (L_span / 2, 0, 0), (0, 0, 1)]
    pl.add_text(FIELD_TITLES[field], color='k')
    pl.add_text(f'Max: {vmax:.4f}\nMin: {vmin:.4f}',
                position='upper_right', font_size=8, color='k')
    pl.add_camera_orientation_widget()

    buf = pl.export_html(None)      # filename=None → StringIO
    html_str = buf.getvalue()
    pl.close()
    return html_str


def placeholder_html(message: str = '请先在左侧设置参数并点击「运行求解」') -> str:
    """3D 区域的空状态占位 HTML。"""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;display:flex;align-items:center;justify-content:center;
background:#f9f9f7;font-family:system-ui,'Segoe UI',sans-serif;">
<div style="color:#52514e;text-align:center;">
<div style="font-size:44px;margin-bottom:14px;">&#128208;</div>
<div style="font-size:15px;">{message}</div>
</div></body></html>"""
