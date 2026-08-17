"""
Post.py — 统一后处理器 + 自适应 EBE-PCG 求解器
===============================================
可视化类 Post（支持线单元 + 3D 实体梁扫掠），
统一求解入口 solve()，支持 JAX/NumPy 双后端。

用法:
    # 命令行
    python Post.py                        # JAX EBE-PCG (默认)
    python Post.py --solver numpy         # NumPy EBE-PCG
    python Post.py --solver jax           # JAX EBE-PCG
    python Post.py --n-elem 200 --L 100   # 自定义参数
    python Post.py --no-viz               # 仅求解，不绘图

    # 编程接口
    from Post import Post, solve
    model = build_native_model(L=100, n_elem=200)
    u, n_iter = solve(model, solver='jax')
    post = Post(model)
    post.showBeamSolid(u, direction='y', show_deformed=True, scale=200)
"""

import sys, os, time
import numpy as np

# ---- PyVista (可视化依赖，非必需) ----
try:
    import pyvista as pv
    import vtkmodules.all as vtk
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False


# ============================================================
# 工具函数
# ============================================================

def calculateSum(u):
    """计算各位移节点的矢量和"""
    ndim = u.shape[1]
    if ndim == 1:
        return np.abs(u[:, 0])
    elif ndim == 2:
        return np.sqrt(u[:, 0]**2 + u[:, 1]**2)
    elif ndim == 3:
        return np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2)
    return np.zeros(u.shape[0])


def analytic_deflection(model: dict) -> float:
    """理论跨中挠度 δ = PL³/(48EI)"""
    L = model.get("_L", model.get("_dx", 1.0) * model["n_elem"])
    E = model["materials"]["E"]
    I = model["sections"]["Iz"]
    P = abs(model["loads"][0][2])
    return P * L**3 / (48 * E * I)


# ============================================================
# 自适应迭代数估计
# ============================================================

def estimate_iterations(n_elem: int, solver: str = 'jax') -> int:
    """
    基于 n_elem 估计 EBE-PCG 所需迭代数。

    经验公式（对角 PCG for Euler-Bernoulli beam3）:
        iter ≈ 0.12 × n_elem^1.72

    条件数 κ ~ O(n_elem⁴), CG 迭代 ~ O(√κ) ~ O(n_elem²)
    实际拟合 (10 ≤ n_elem ≤ 500): iter ~ O(n_elem^1.7)

    返回建议的 max_iter（含安全余量）。
    """
    base = max(100, int(0.12 * n_elem**1.72))
    return base + 500  # 安全余量


# ============================================================
# 直接求解（参考基准）
# ============================================================

def direct_solve(model: dict) -> np.ndarray:
    """NumPy 直接求解 K·u = F（BEAM4 参考基准）。"""
    from beam_element import beam4_stiffness, DOF_PER_ELEM, DOF_PER_NODE

    n_elem = model["n_elem"]; n_dofs = model["n_dofs"]
    elements = model["elements"]
    E = model["materials"]["E"]; nu = model["materials"]["nu"]
    G = E / (2.0 * (1.0 + nu))
    A = model["sections"]["area"]; Iz = model["sections"]["Iz"]
    Iy = model["sections"]["Iy"]; J_val = model["sections"]["J"]
    dx = model["_dx"]

    K_single = beam4_stiffness(E, G, A, Iz, Iy, J_val, dx)
    K_e = np.tile(K_single[np.newaxis, :, :], (n_elem, 1, 1))

    dof_map = np.zeros((n_elem, DOF_PER_ELEM), dtype=int)
    for e in range(n_elem):
        n1, n2 = elements[e]
        for d in range(DOF_PER_NODE):
            dof_map[e, d] = n1 * DOF_PER_NODE + d
            dof_map[e, DOF_PER_NODE + d] = n2 * DOF_PER_NODE + d

    K_global = np.zeros((n_dofs, n_dofs))
    for e in range(n_elem):
        for i in range(DOF_PER_ELEM):
            for j in range(DOF_PER_ELEM):
                K_global[dof_map[e,i], dof_map[e,j]] += K_e[e,i,j]

    F = np.zeros(n_dofs)
    for node, dof, val in model["loads"]:
        F[node * DOF_PER_NODE + dof] = val

    K_bc, F_bc = K_global.copy(), F.copy()
    for node, dof in model["boundary"]:
        g = node * DOF_PER_NODE + dof
        K_bc[g, g] = 1e15
        F_bc[g] = 0.0

    return np.linalg.solve(K_bc, F_bc)


# ============================================================
# 统一求解入口
# ============================================================

def solve(model: dict, solver: str = 'jax',
          tol: float = 1e-8, max_iter: int = None,
          verbose: bool = True) -> tuple:
    """
    统一 EBE-PCG 求解入口 (BEAM4, 6 DOF/节点, 12 DOF/元素)。

    参数:
        model:    build_native_model() 输出
        solver:   'jax' | 'numpy'
        tol:      收敛容差
        max_iter: None=自适应

    返回:
        u:        (n_dofs,) np.ndarray
        n_iter:   int
        info:     dict
    """
    from beam_element import (beam4_stiffness, DOF_PER_NODE, DOF_PER_ELEM,
                               build_dof_map as beam_dof_map)

    n_elem = model["n_elem"]
    n_dofs = model["n_dofs"]
    elements = model["elements"]
    E = model["materials"]["E"]; nu = model["materials"]["nu"]
    G = E / (2.0 * (1.0 + nu))
    A = model["sections"]["area"]; Iz = model["sections"]["Iz"]
    Iy = model["sections"]["Iy"]; J_val = model["sections"]["J"]
    dx = model["_dx"]

    # ---- 自适应 max_iter ----
    if max_iter is None:
        max_iter = estimate_iterations(n_elem, solver)
        if verbose:
            print(f"  [solve] 自适应 max_iter = {max_iter} "
                  f"(n_elem={n_elem})")

    # ---- DOF 映射 ----
    dof_map_np = np.zeros((n_elem, DOF_PER_ELEM), dtype=int)
    for e in range(n_elem):
        n1, n2 = elements[e]
        for d in range(DOF_PER_NODE):
            dof_map_np[e, d] = n1 * DOF_PER_NODE + d
            dof_map_np[e, DOF_PER_NODE + d] = n2 * DOF_PER_NODE + d

    # ---- 约束 DOF ----
    constrained_dofs = []
    for node, dof in model["boundary"]:
        constrained_dofs.append(node * DOF_PER_NODE + dof)
    constrained_dofs = sorted(constrained_dofs)

    # ---- 力向量 ----
    F_np = np.zeros(n_dofs)
    for node, dof, val in model["loads"]:
        F_np[node * DOF_PER_NODE + dof] = val

    # ---- 单元刚度 (BEAM4) ----
    K_single = beam4_stiffness(E, G, A, Iz, Iy, J_val, dx)

    # ---- BC 施加 ----
    def apply_bc_np(K_e_arr, dof_map_arr, constrained, large=1e12):
        K_bc = K_e_arr.copy()
        mask = np.zeros((n_elem, DOF_PER_ELEM), dtype=bool)
        for g in constrained:
            mask = mask | (dof_map_arr == g)
        for e in range(n_elem):
            for ld in range(DOF_PER_ELEM):
                if mask[e, ld]:
                    K_bc[e, ld, ld] = large
        return K_bc

    # ---- 选择后端 ----
    if solver == 'jax':
        import jax, jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        from jax_ebe.ebe_pcg import (
            build_dof_map as jax_dof_map,
            build_adjacency_matrices,
            ebe_pcg_solve as jax_ebe_solve,
        )

        elements_jax = jnp.array(elements, dtype=jnp.int32)
        dof_map_jax = jax_dof_map(elements_jax)
        adj_left, adj_right = build_adjacency_matrices(elements_jax)
        K_e_raw = jnp.tile(jnp.array(K_single)[jnp.newaxis, :, :], (n_elem, 1, 1))
        K_e_bc = apply_bc_np(np.array(K_e_raw), dof_map_np, constrained_dofs)
        K_e_bc_jax = jnp.array(K_e_bc)
        F_jax = jnp.array(F_np)

        # Warmup
        uw, _ = jax_ebe_solve(K_e_bc_jax, F_jax, dof_map_jax,
                               adj_left, adj_right, constrained_dofs,
                               tol=tol, max_iter=min(100, max_iter), verbose=False)
        _ = uw.block_until_ready()

        if verbose:
            print(f"  [solve] JAX EBE-PCG 求解中 (max_iter={max_iter})...")
        t0 = time.perf_counter()
        u_raw, n_iter = jax_ebe_solve(K_e_bc_jax, F_jax, dof_map_jax,
                                        adj_left, adj_right, constrained_dofs,
                                        tol=tol, max_iter=max_iter, verbose=verbose)
        t_elapsed = time.perf_counter() - t0
        u = np.array(u_raw)

    elif solver == 'numpy':
        from numpy_ebe.ebe_pcg import (
            build_adjacency,
            ebe_pcg_solve as numpy_ebe_solve,
        )

        K_e_np = np.tile(K_single[np.newaxis, :, :], (n_elem, 1, 1))
        K_e_bc = apply_bc_np(K_e_np, dof_map_np, constrained_dofs)
        adj = build_adjacency(elements)

        if verbose:
            print(f"  [solve] NumPy EBE-PCG 求解中 (max_iter={max_iter})...")
        t0 = time.perf_counter()
        u, n_iter = numpy_ebe_solve(K_e_bc, F_np, dof_map_np, adj,
                                     constrained_dofs,
                                     tol=tol, max_iter=max_iter, verbose=verbose)
        t_elapsed = time.perf_counter() - t0

    else:
        raise ValueError(f"Unknown solver '{solver}'. Choose 'jax' or 'numpy'.")

    info = {
        'time_ms': t_elapsed * 1000,
        'max_iter': max_iter,
        'solver': solver,
        'n_elem': n_elem,
    }
    return u, n_iter, info


# ============================================================
# Post 类（可视化）
# ============================================================

class Post:
    """EBE-PCG 梁求解器后处理器。

    参数:
        model: ansys_parser 输出的模型 dict，包含:
            nodes, elements, n_nodes, n_elem,
            dof_per_node, n_dofs
    """

    def __init__(self, model: dict):
        if not _HAS_PYVISTA:
            raise ImportError(
                "pyvista 未安装。请运行: pip install pyvista"
            )
        self.model = model
        self.n_nodes = model["n_nodes"]
        self.n_elem = model["n_elem"]
        self.dof_per_node = model["dof_per_node"]   # 3
        self.n_dofs = model["n_dofs"]
        self.nodes = model["nodes"]                  # (n_nodes, 3)
        self.elements = model["elements"]            # (n_elem, 2)

    # ========================================================
    # 网格构建（内部）
    # ========================================================

    def _build_mesh(self, displacement=None, scale=1.0):
        """构建 PyVista UnstructuredGrid（VTK_LINE 线单元）。"""
        if displacement is not None:
            disp_mat = self.getDisplacementMatrix(displacement)
            node_coords = self.nodes.copy().astype(np.float64)
            node_coords[:, 0] += disp_mat[:, 0] * scale  # UX
            node_coords[:, 1] += disp_mat[:, 1] * scale  # UY
        else:
            node_coords = self.nodes.copy().astype(np.float64)

        elem_conn = self.elements
        celltypes = np.full(self.n_elem, vtk.VTK_LINE, dtype=np.uint8)
        head = np.full((self.n_elem, 1), 2, dtype=int)
        cells = np.hstack([head, elem_conn])

        return pv.UnstructuredGrid(cells, celltypes, node_coords)

    # ========================================================
    # 数据提取
    # ========================================================

    def getDisplacementMatrix(self, displacement):
        """将 (n_dofs,) 位移向量 reshape 为 (n_nodes, dof_per_node) 矩阵。

        列顺序: [UX, UY, ROTZ]
        """
        return np.reshape(displacement, (self.n_nodes, self.dof_per_node))

    def _get_displacement_components(self, displacement):
        """从位移向量提取各分量 (ux, uy, uz, rotx, roty, rotz)。"""
        disp = self.getDisplacementMatrix(displacement)
        return (disp[:, 0], disp[:, 1], disp[:, 2],
                disp[:, 3], disp[:, 4], disp[:, 5])

    # ========================================================
    # 标量显示
    # ========================================================

    def showScalar(self, scalar, name='scalar', showEdge=True):
        """在网格上显示任意节点标量场。"""
        mesh = self._build_mesh()
        mesh.point_data[name] = scalar

        dargs = dict(
            cmap="coolwarm",
            show_scalar_bar=True,
            scalar_bar_args={'title': name, 'color': 'k'},
        )

        pl = pv.Plotter()
        pl.add_mesh(mesh, scalars=scalar, **dargs, show_edges=showEdge)
        pl.background_color = 'white'
        pl.add_camera_orientation_widget()
        pl.add_text(
            f"nNodes = {self.n_nodes}\nnElem = {self.n_elem}",
            color='k'
        )
        pl.show()

    # ========================================================
    # 位移显示
    # ========================================================

    def showDisplacement(self, displacement, direction='all',
                         showEdge=True, show_deformed=False, scale=1.0,
                         show_shade=True, opacity=0.08, shade_color='lightgray',
                         smooth=True, div=9, save_path=None,
                         save_html=None, window_size=None):
        """显示位移场。

        参数:
            displacement:  (n_dofs,) 全局位移向量
            direction:     'all' | 'x' | 'y' | 'rotz' | 'sum'
            show_deformed: 是否显示变形后的网格
            scale:         变形放大系数
            show_shade:    是否叠加未变形对照
        """
        disp_mat = self.getDisplacementMatrix(displacement)

        if show_deformed:
            mesh = self._build_mesh(displacement, scale)
        else:
            mesh = self._build_mesh()

        ux, uy, uz, rotx, roty, rotz = self._get_displacement_components(displacement)
        mesh.point_data['UX (mm)'] = ux * 1000
        mesh.point_data['UY (mm)'] = uy * 1000
        mesh.point_data['UZ (mm)'] = uz * 1000
        mesh.point_data['ROTX (rad)'] = rotx
        mesh.point_data['ROTY (rad)'] = roty
        mesh.point_data['ROTZ (rad)'] = rotz
        usum = calculateSum(disp_mat[:, :2])
        mesh.point_data['USUM (mm)'] = usum * 1000

        if smooth:
            dargs_base = dict(
                cmap="coolwarm", show_scalar_bar=True,
                interpolate_before_map=True,
                scalar_bar_args={'n_labels': 7, 'fmt': '%.4f',
                                 'label_font_size': 10,
                                 'title_font_size': 12},
            )
        else:
            dargs_base = dict(
                cmap="coolwarm", show_scalar_bar=True,
                interpolate_before_map=False, n_colors=div,
                scalar_bar_args={'n_labels': 7, 'fmt': '%.4f',
                                 'label_font_size': 10,
                                 'title_font_size': 12},
            )

        if show_shade:
            mesh_undef = self._build_mesh()

        if direction == 'all':
            pl = pv.Plotter(shape=(1, 3))
            for idx, (name, title) in enumerate([
                ('UX (mm)', 'UX Displacement'),
                ('UY (mm)', 'UY Displacement'),
                ('ROTZ (rad)', 'ROTZ Rotation'),
            ]):
                pl.subplot(0, idx)
                m = mesh.copy()
                m.point_data.active_scalars_name = name
                pl.add_mesh(m, **dargs_base, show_edges=showEdge)
                if show_shade:
                    pl.add_mesh(mesh_undef, color=shade_color,
                                opacity=opacity, show_edges=False)
                pl.add_text(title, color='k')
            pl.link_views()
        else:
            pl = pv.Plotter()
            name_map = {
                'x': ('UX (mm)', 'UX Displacement'),
                'y': ('UY (mm)', 'UY Displacement'),
                'rotz': ('ROTZ (rad)', 'ROTZ Displacement'),
                'sum': ('USUM (mm)', 'Total Displacement'),
            }
            if direction not in name_map:
                raise ValueError(
                    f"Unknown direction '{direction}'. "
                    f"Choose from: {list(name_map.keys())}"
                )
            field_name, title = name_map[direction]
            mesh.point_data.active_scalars_name = field_name
            dargs = dict(dargs_base)
            dargs['scalar_bar_args'] = {
                'title': direction.upper(), 'color': 'k',
                'n_labels': 7, 'fmt': '%.4f',
                'label_font_size': 10, 'title_font_size': 12,
            }
            pl.add_mesh(mesh, **dargs, show_edges=showEdge)
            if show_shade:
                pl.add_mesh(mesh_undef, color=shade_color,
                            opacity=opacity, show_edges=False)
            pl.add_text(title, color='k')

        pl.background_color = 'white'
        pl.add_camera_orientation_widget()
        if save_html:
            pl.export_html(save_html)
            pl.close()
        elif save_path:
            if window_size:
                pl.window_size = window_size
            pl.show(screenshot=save_path)
        else:
            pl.show()

    # ========================================================
    # 3D 实体梁可视化（截面扫掠成六面体）
    # ========================================================

    def _build_solid_mesh(self, displacement=None, scale=1.0):
        """将矩形截面沿梁轴扫掠，构建六面体实体网格。"""
        sections = self.model["sections"]
        b_height = sections["height"]
        b_width = sections["area"] / b_height

        hw = b_width / 2.0
        hh = b_height / 2.0

        if displacement is not None:
            disp_mat = self.getDisplacementMatrix(displacement)
            beam_xyz = self.nodes.copy().astype(np.float64)
            beam_xyz[:, 0] += disp_mat[:, 0] * scale
            beam_xyz[:, 1] += disp_mat[:, 1] * scale
        else:
            beam_xyz = self.nodes.copy().astype(np.float64)

        offsets = np.array([
            [-hh, -hw], [+hh, -hw], [+hh, +hw], [-hh, +hw],
        ], dtype=np.float64)

        n_per_node = 4
        n_beam = self.n_nodes
        solid_pts = np.zeros((n_beam * n_per_node, 3), dtype=np.float64)

        for i in range(n_beam):
            x, y, z = beam_xyz[i]
            for j in range(n_per_node):
                dy, dz = offsets[j]
                solid_pts[i * n_per_node + j] = [x, y + dy, z + dz]

        n_hex = self.n_elem
        celltypes = np.full(n_hex, vtk.VTK_HEXAHEDRON, dtype=np.uint8)
        head = np.full((n_hex, 1), 8, dtype=int)

        cells_data = np.zeros((n_hex, 8), dtype=int)
        for e in range(n_hex):
            b0, b1, b2, b3 = [e * n_per_node + k for k in range(4)]
            t0, t1, t2, t3 = [(e + 1) * n_per_node + k for k in range(4)]
            cells_data[e] = [b0, b1, b2, b3, t0, t1, t2, t3]

        cells = np.hstack([head, cells_data])
        mesh = pv.UnstructuredGrid(cells, celltypes, solid_pts)
        return mesh, n_per_node

    def showBeamSolid(self, displacement, direction='sum',
                      showEdge=True, show_deformed=True, scale=200,
                      show_shade=True, opacity=0.2, shade_color='lightgray',
                      smooth=True, div=9, save_path=None,
                      save_html=None, window_size=None):
        """以 3D 实体梁形式显示位移场。

        参数:
            displacement:  (n_dofs,) 全局位移向量
            direction:     'all' | 'x' | 'y' | 'rotz' | 'sum'
            show_deformed: 是否显示变形（默认 True）
            scale:         变形放大系数（默认 200）
        """
        if show_deformed:
            mesh, n_per_node = self._build_solid_mesh(displacement, scale)
        else:
            mesh, n_per_node = self._build_solid_mesh()

        ux, uy, uz, rotx, roty, rotz = self._get_displacement_components(displacement)
        usum = calculateSum(self.getDisplacementMatrix(displacement)[:, :2])

        # 位移 m→mm，数值更可读；转角保持 rad
        mesh.point_data['UX (mm)']   = np.repeat(ux * 1000, n_per_node)
        mesh.point_data['UY (mm)']   = np.repeat(uy * 1000, n_per_node)
        mesh.point_data['UZ (mm)']   = np.repeat(uz * 1000, n_per_node)
        mesh.point_data['ROTX (rad)'] = np.repeat(rotx, n_per_node)
        mesh.point_data['ROTY (rad)'] = np.repeat(roty, n_per_node)
        mesh.point_data['ROTZ (rad)'] = np.repeat(rotz, n_per_node)
        mesh.point_data['USUM (mm)'] = np.repeat(usum * 1000, n_per_node)

        if smooth:
            dargs_base = dict(
                cmap="coolwarm", show_scalar_bar=True,
                interpolate_before_map=True,
                scalar_bar_args={'n_labels': 7, 'fmt': '%.4f',
                                 'label_font_size': 10,
                                 'title_font_size': 12},
            )
        else:
            dargs_base = dict(
                cmap="coolwarm", show_scalar_bar=True,
                interpolate_before_map=False, n_colors=div,
                scalar_bar_args={'n_labels': 7, 'fmt': '%.4f',
                                 'label_font_size': 10,
                                 'title_font_size': 12},
            )

        # ---- 未变形对照网格（预构建，所有子图复用） ----
        if show_shade:
            mesh_undef, _ = self._build_solid_mesh()

        # ---- 相机位置: 从 Y 方向侧视，使梁高 (Z=0.3m) 在屏幕上竖直可见 ----
        L_span = self.model["_L"]
        camera_pos = [(L_span / 2, -L_span * 0.6, L_span * 0.3),
                      (L_span / 2, 0, 0),
                      (0, 0, 1)]

        if direction == 'all':
            pl = pv.Plotter(shape=(1, 3))
            for idx, (name, title) in enumerate([
                ('UX (mm)', 'UX Displacement'),
                ('UY (mm)', 'UY Displacement'),
                ('ROTZ (rad)', 'ROTZ Rotation'),
            ]):
                pl.subplot(0, idx)
                m = mesh.copy()
                m.point_data.active_scalars_name = name
                pl.add_mesh(m, **dargs_base, show_edges=showEdge)
                # 每个子图单独加未变形虚影
                if show_shade:
                    pl.add_mesh(mesh_undef, color=shade_color,
                                opacity=opacity, show_edges=False)
                pl.add_text(title, color='k')
            pl.link_views()
        else:
            name_map = {
                'x': ('UX (mm)', 'UX Displacement'),
                'y': ('UY (mm)', 'UY Displacement'),
                'rotz': ('ROTZ (rad)', 'ROTZ Displacement'),
                'sum': ('USUM (mm)', 'Total Displacement'),
            }
            if direction not in name_map:
                raise ValueError(
                    f"Unknown direction '{direction}'. "
                    f"Choose from: {list(name_map.keys())}"
                )
            field_name, title = name_map[direction]
            mesh.point_data.active_scalars_name = field_name
            vals = mesh.point_data[field_name]
            vmin, vmax = float(vals.min()), float(vals.max())
            dargs = dict(dargs_base, clim=[vmin, vmax])
            dargs['scalar_bar_args'] = {
                'title': direction.upper(), 'color': 'k',
                'n_labels': 7, 'fmt': '%.4f',
                'label_font_size': 10, 'title_font_size': 12,
            }
            pl = pv.Plotter()
            pl.add_mesh(mesh, **dargs, show_edges=showEdge)
            if show_shade:
                pl.add_mesh(mesh_undef, color=shade_color,
                            opacity=opacity, show_edges=False)
            pl.add_text(title, color='k')

        # ---- 标注最大/最小值 ----
        if direction != 'all':
            scalars = mesh.point_data.active_scalars
            if scalars is not None:
                vmin, vmax = float(scalars.min()), float(scalars.max())
                pl.add_text(f'Max: {vmax:.4f}\nMin: {vmin:.4f}',
                            position='upper_right', font_size=8, color='k')

        pl.background_color = 'white'
        pl.camera_position = camera_pos
        pl.add_camera_orientation_widget()
        if save_html:
            pl.export_html(save_html)
            pl.close()
        elif save_path:
            if window_size:
                pl.window_size = window_size
            pl.show(screenshot=save_path)
        else:
            pl.show()

    # ========================================================
    # Mises 应力
    # ========================================================

    def computeMisesStress(self, displacement, K_e=None, dof_map=None):
        """从位移解反算节点 von Mises 应力 (BEAM4)。

        BEAM4 内力 f^e (12 DOF):
          [N1,V1y,V1z,T1,M1y,M1z, N2,V2y,V2z,T2,M2y,M2z]
        XY 平面弯曲: M_z = f_e[5], f_e[11]
        σ_vm = |M| · y_max / I_z
        """
        from beam_element import beam4_stiffness, DOF_PER_ELEM, DOF_PER_NODE

        E = self.model["materials"]["E"]; nu = self.model["materials"]["nu"]
        G = E / (2.0 * (1.0 + nu))
        I_z = self.model["sections"]["Iz"]; I_y = self.model["sections"]["Iy"]
        J_v = self.model["sections"]["J"]; A_sec = self.model["sections"]["area"]
        h = self.model["sections"]["height"]; y_max = h / 2.0
        dx = self.model["_dx"]
        elements = self.model["elements"]; n_elem = self.n_elem

        if dof_map is None:
            dof_map = np.zeros((n_elem, DOF_PER_ELEM), dtype=int)
            for e in range(n_elem):
                n1, n2 = elements[e]
                for d in range(DOF_PER_NODE):
                    dof_map[e, d] = n1 * DOF_PER_NODE + d
                    dof_map[e, DOF_PER_NODE + d] = n2 * DOF_PER_NODE + d

        if K_e is None:
            K_single = beam4_stiffness(E, G, A_sec, I_z, I_y, J_v, dx)
            K_e = np.tile(K_single[np.newaxis, :, :], (n_elem, 1, 1))

        u_global = np.asarray(displacement).flatten()
        M_nodal = np.zeros(self.n_nodes); count = np.zeros(self.n_nodes)

        for e in range(n_elem):
            n1, n2 = elements[e]
            u_e = u_global[dof_map[e]]
            f_e = K_e[e] @ u_e  # 12-DOF
            M_nodal[n1] += -f_e[5]   # M_z at node 1
            count[n1] += 1
            M_nodal[n2] += f_e[11]   # M_z at node 2
            count[n2] += 1

        M_nodal /= np.maximum(count, 1)
        sigma_vm = np.abs(M_nodal) * y_max / I_z
        return sigma_vm

    def showMises(self, displacement, K_e=None, dof_map=None,
                  showEdge=True, show_deformed=True,
                  scale=200, show_shade=True, opacity=0.2,
                  shade_color='lightgray', smooth=True, div=9,
                  save_path=None, save_html=None,
                  window_size=None):
        """显示 Mises 应力云图（3D 实体梁 + 变形 + 应力着色）。"""
        sigma_vm = self.computeMisesStress(displacement, K_e=K_e,
                                           dof_map=dof_map)

        if show_deformed:
            mesh, n_per_node = self._build_solid_mesh(displacement, scale)
        else:
            mesh, n_per_node = self._build_solid_mesh()

        # Pa → MPa
        mesh.point_data['von Mises (MPa)'] = np.repeat(sigma_vm / 1e6, n_per_node)
        mesh.point_data.active_scalars_name = 'von Mises (MPa)'

        if smooth:
            dargs = dict(
                cmap="coolwarm", show_scalar_bar=True,
                interpolate_before_map=True,
                scalar_bar_args={'title': 'von Mises (MPa)', 'color': 'k',
                                     'n_labels': 7, 'fmt': '%.4f',
                                     'label_font_size': 10,
                                     'title_font_size': 12},
            )
        else:
            dargs = dict(
                cmap="coolwarm", show_scalar_bar=True,
                interpolate_before_map=False, n_colors=div,
                scalar_bar_args={'title': 'von Mises (MPa)', 'color': 'k',
                                     'n_labels': 7, 'fmt': '%.4f',
                                     'label_font_size': 10,
                                     'title_font_size': 12},
            )

        # ---- 相机: 侧视角度，显示梁高 (Z) ----
        L_span = self.model["_L"]

        # 显式设 clim 确保色条覆盖完整数据范围
        vmin, vmax = float(sigma_vm.min() / 1e6), float(sigma_vm.max() / 1e6)
        dargs['clim'] = [vmin, vmax]

        pl = pv.Plotter()
        pl.add_mesh(mesh, **dargs, show_edges=showEdge)
        pl.add_text('von Mises Stress', color='k')

        # 标注 Mises 应力最值
        pl.add_text(f'Max: {vmax:.4f} MPa\nMin: {vmin:.4f} MPa',
                    position='upper_right', font_size=8, color='k')

        if show_shade:
            mesh_undef, _ = self._build_solid_mesh()
            pl.add_mesh(mesh_undef, color=shade_color, opacity=opacity,
                        show_edges=False)

        pl.background_color = 'white'
        pl.camera_position = [(L_span / 2, -L_span * 0.6, L_span * 0.3),
                              (L_span / 2, 0, 0), (0, 0, 1)]
        pl.add_camera_orientation_widget()
        if save_html:
            pl.export_html(save_html)
            pl.close()
        elif save_path:
            if window_size:
                pl.window_size = window_size
            pl.show(screenshot=save_path)
        else:
            pl.show()


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="统一 EBE-PCG 求解 + 可视化"
    )
    parser.add_argument("--solver", choices=["jax", "numpy"], default="jax",
                        help="求解器后端 (default: jax)")
    parser.add_argument("--n-elem", type=int, default=200,
                        help="单元数 (default: 200)")
    parser.add_argument("--L", type=float, default=10.0,
                        help="梁跨度 m (default: 100)")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="最大迭代数 (default: 自适应)")
    parser.add_argument("--tol", type=float, default=1e-8,
                        help="收敛容差 (default: 1e-8)")
    parser.add_argument("--no-viz", action="store_true",
                        help="跳过可视化，仅求解")
    parser.add_argument("--direction", default="y",
                        choices=["all", "x", "y", "rotz", "sum"],
                        help="位移分量 (default: y)")
    parser.add_argument("--scale", type=float, default=200,
                        help="变形放大系数 (default: 200)")
    parser.add_argument("--mises", action="store_true",
                        help="显示 Mises 应力")
    parser.add_argument("--save", type=str, default=None,
                        help="保存截图到指定路径 (png/pdf/svg)")
    parser.add_argument("--save-html", type=str, default=None,
                        help="保存交互式HTML到指定路径 (可插入PPT)")
    parser.add_argument("--window-size", type=int, nargs=2, default=None,
                        metavar=("W", "H"),
                        help="截图分辨率 (default: 1024 768)")
    args = parser.parse_args()

    # ---- 导入模型 ----
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ansys.ansys_parser import build_native_model, print_model_summary
    from beam_element import print_beam4_info

    print("=" * 60)
    print(f"  EBE-PCG 简支梁 — {args.solver.upper()} 后端")
    print("=" * 60)

    model = build_native_model(L=args.L, n_elem=args.n_elem)
    print_model_summary(model)
    print_beam4_info(model)
    theory = analytic_deflection(model)
    est_iter = estimate_iterations(model["n_elem"])
    print(f"  自适应 max_iter 估计: {est_iter}")

    # ---- 求解 ----
    if args.max_iter is None:
        args.max_iter = est_iter

    u, n_iter, info = solve(model, solver=args.solver,
                             tol=args.tol, max_iter=args.max_iter,
                             verbose=True)

    # ---- 验证 ----
    u_direct = direct_solve(model)
    err = np.linalg.norm(u - u_direct) / np.linalg.norm(u_direct) * 100
    mid_dof = model["_mid_node"] * model["dof_per_node"] + 1  # UY

    print(f"\n  {'='*60}")
    print(f"  求解结果")
    print(f"  {'='*60}")
    print(f"  求解器:       {args.solver.upper()} EBE-PCG")
    print(f"  迭代次数:     {n_iter} / {args.max_iter}")
    print(f"  耗时:         {info['time_ms']:.1f} ms")
    print(f"  跨中挠度:     {u[mid_dof]*1000:.4f} mm")
    print(f"  理论值:       {theory*1000:.4f} mm")
    print(f"  直接解:       {u_direct[mid_dof]*1000:.4f} mm")
    print(f"  vs 直接解:    {err:.6f} %")

    if err > 1.0:
        print(f"  ⚠ 误差较大！max_iter={args.max_iter} 可能不足。")
        print(f"    建议: --max-iter {estimate_iterations(args.n_elem) + 1000}")

    # ---- 可视化 ----
    ws = tuple(args.window_size) if args.window_size else None
    if not args.no_viz and _HAS_PYVISTA:
        post = Post(model)
        if args.mises:
            print(f"\n>>> Mises 应力")
            post.showMises(u, show_deformed=True, scale=args.scale,
                          save_path=args.save, save_html=args.save_html,
                          window_size=ws)
        else:
            print(f"\n>>> 3D 实体梁 {args.direction.upper()} 位移")
            post.showBeamSolid(u, direction=args.direction,
                              show_deformed=True, scale=args.scale,
                              save_path=args.save, save_html=args.save_html,
                              window_size=ws)
            if not args.save and not args.save_html:
                print(f"\n>>> 三分量总览")
                post.showBeamSolid(u, direction="all", show_deformed=True,
                                  scale=args.scale)
    elif not _HAS_PYVISTA and not args.no_viz:
        print("\n  ⚠ pyvista 未安装，跳过可视化。pip install pyvista")