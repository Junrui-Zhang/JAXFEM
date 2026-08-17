"""
beam_element.py — BEAM4 3D 梁单元刚度矩阵
===========================================
ANSYS BEAM4 等效单元：2 节点，每节点 6 DOF:
  [UX, UY, UZ, ROTX, ROTY, ROTZ]

局部坐标系下 12×12 刚度矩阵，包含：
  - 轴向刚度 (EA/L)
  - 扭转刚度 (GJ/L)
  - XY 平面弯曲 (I_z)
  - XZ 平面弯曲 (I_y)

截面属性通过 sectionproperties 计算，支持任意截面形状。

用法:
    from beam_element import beam4_stiffness, compute_section_props
    props = compute_section_props(b=0.2, h=0.3)        # 矩形
    K = beam4_stiffness(E, G, props['area'], props['Iz'],
                        props['Iy'], props['J'], L)
"""

import numpy as np

# ---- sectionproperties 可用性 ----
try:
    import sectionproperties as sp
    from sectionproperties.pre.library.primitive_sections import rectangular_section
    _HAS_SP = True
except ImportError:
    _HAS_SP = False


# ============================================================
# 截面属性计算
# ============================================================

def compute_section_props(b: float = None, h: float = None,
                          geometry=None, mesh_size: float = None) -> dict:
    """
    使用 sectionproperties 计算任意截面的工程属性。

    两种调用方式:
      1. 矩形截面:   compute_section_props(b=0.2, h=0.3)
      2. 自定义形状:  compute_section_props(geometry=my_geom, mesh_size=0.01)

    返回:
        dict: area, Iz (绕Z轴=XY面内弯曲), Iy (绕Y轴=XZ面内弯曲),
              J (扭转常数), cy, cz (形心偏移)
              Izz=Ix, Iyy=Iy, centroid_x, centroid_y (sp原始字段)

    坐标映射 (sp → beam):
        sp.x (截面水平) → beam.Y (梁宽方向)
        sp.y (截面竖直) → beam.Z (梁高方向)
        sp.Ixx (绕x轴) → beam.Iz  (XY面内弯曲)
        sp.Iyy (绕y轴) → beam.Iy  (XZ面内弯曲)
    """
    if _HAS_SP:
        return _compute_section_props_sp(b, h, geometry, mesh_size)
    else:
        return _compute_section_props_numpy(b, h)


def _compute_section_props_sp(b: float = None, h: float = None,
                               geometry=None, mesh_size: float = None) -> dict:
    """使用 sectionproperties 计算截面属性"""
    if geometry is not None:
        geom = geometry
    elif b is not None and h is not None:
        geom = rectangular_section(b=b, d=h)
    else:
        raise ValueError("必须提供 (b, h) 或 geometry")

    # 自动网格尺寸: min(b,h)/100，确保翘曲函数 J 精度 <0.5%
    if mesh_size is None:
        if b is not None and h is not None:
            mesh_size = min(b, h) / 100.0
        else:
            mesh_size = 0.002

    # 网格划分 + 分析
    geom.create_mesh(mesh_sizes=[mesh_size])
    section = sp.analysis.section.Section(geometry=geom)
    section.calculate_geometric_properties()
    section.calculate_warping_properties()
    props = section.section_props

    return {
        "area":     props.area,
        "Iz":       props.ixx_c,    # 绕 x 轴弯曲 → 梁 Iz
        "Iy":       props.iyy_c,    # 绕 y 轴弯曲 → 梁 Iy
        "J":        props.j,        # 扭转常数 (精确解)
        "Ixx":      props.ixx_c,
        "Iyy":      props.iyy_c,
        "Ixy":      props.ixy_c,
        "cy":       props.cx,       # 形心 x 坐标 (sp)
        "cz":       props.cy,       # 形心 y 坐标 (sp)
        "centroid_x": props.cx,
        "centroid_y": props.cy,
        "method":   "sectionproperties",
    }


def _compute_section_props_numpy(b: float, h: float) -> dict:
    """
    矩形截面属性 (NumPy 解析解, 无 sectionproperties 时的备选)。

    矩形扭转常数近似公式（弹性力学级数解截断）:
        J ≈ b³h · [1/3 - 0.21·(b/h)·(1 - (b/h)⁴/12)]   (b ≤ h)
    """
    if b is None or h is None:
        raise ValueError("NumPy 备选仅支持矩形截面，需提供 b 和 h")

    area = b * h
    Iz = b * h**3 / 12.0
    Iy = h * b**3 / 12.0

    ratio = min(b, h) / max(b, h)
    J_approx = max(b, h) * min(b, h)**3 * (
        1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0)
    )

    return {
        "area":     area,
        "Iz":       Iz,
        "Iy":       Iy,
        "J":        J_approx,
        "Ixx":      Iz,
        "Iyy":      Iy,
        "Ixy":      0.0,
        "cy":       b / 2.0,
        "cz":       h / 2.0,
        "centroid_x": b / 2.0,
        "centroid_y": h / 2.0,
        "method":   "numpy_approx",
    }


# 保留旧接口兼容
def rect_section_props(b: float, h: float) -> dict:
    """矩形截面属性（兼容旧接口，内部调用 compute_section_props）"""
    return compute_section_props(b=b, h=h)


# ============================================================
# BEAM4 单元刚度矩阵 (12×12)
# ============================================================

def beam4_stiffness(E: float, G: float, A: float,
                    I_z: float, I_y: float, J: float,
                    L: float) -> np.ndarray:
    """
    BEAM4 单元刚度矩阵 (12×12)，局部坐标系。

    DOF 顺序（每节点 6 个）:
        [UX, UY, UZ, ROTX, ROTY, ROTZ]

    节点 1 → DOF [0:6],  节点 2 → DOF [6:12]
    """
    # 轴向
    EA_L = E * A / L

    # XY 平面弯曲 (I_z → UY, ROTZ)
    EIz_L3 = 12.0 * E * I_z / L**3
    EIz_L2 = 6.0  * E * I_z / L**2
    EIz_L1 = 4.0  * E * I_z / L
    EIz_h  = 2.0  * E * I_z / L   # "half" coupling

    # XZ 平面弯曲 (I_y → UZ, ROTY)
    EIy_L3 = 12.0 * E * I_y / L**3
    EIy_L2 = 6.0  * E * I_y / L**2
    EIy_L1 = 4.0  * E * I_y / L
    EIy_h  = 2.0  * E * I_y / L

    # 扭转
    GJ_L = G * J / L

    # ---- 组装 12×12 ----
    # 上三角逐行定义，矩阵对称
    K = np.zeros((12, 12), dtype=np.float64)

    # Row 0: UX1
    K[0, 0] = EA_L
    K[0, 6] = -EA_L

    # Row 1: UY1
    K[1, 1] = EIz_L3
    K[1, 5] = EIz_L2
    K[1, 7] = -EIz_L3
    K[1, 11] = EIz_L2

    # Row 2: UZ1
    K[2, 2] = EIy_L3
    K[2, 4] = -EIy_L2
    K[2, 8] = -EIy_L3
    K[2, 10] = -EIy_L2

    # Row 3: ROTX1
    K[3, 3] = GJ_L
    K[3, 9] = -GJ_L

    # Row 4: ROTY1
    K[4, 4] = EIy_L1
    K[4, 8] = EIy_L2
    K[4, 10] = EIy_h

    # Row 5: ROTZ1
    K[5, 5] = EIz_L1
    K[5, 7] = -EIz_L2
    K[5, 11] = EIz_h

    # Row 6: UX2
    K[6, 6] = EA_L

    # Row 7: UY2
    K[7, 7] = EIz_L3
    K[7, 11] = -EIz_L2

    # Row 8: UZ2
    K[8, 8] = EIy_L3
    K[8, 10] = EIy_L2

    # Row 9: ROTX2
    K[9, 9] = GJ_L

    # Row 10: ROTY2
    K[10, 10] = EIy_L1

    # Row 11: ROTZ2
    K[11, 11] = EIz_L1

    # 对称填充下三角
    K = K + K.T - np.diag(np.diag(K))

    return K


# ============================================================
# 批量计算（均匀网格）
# ============================================================

def compute_all_stiffness(model: dict) -> np.ndarray:
    """
    计算所有单元的 BEAM4 刚度矩阵。

    参数:
        model: build_native_model() 输出

    返回:
        K_e: (n_elem, 12, 12) ndarray
    """
    n_elem = model["n_elem"]
    E = model["materials"]["E"]
    nu = model["materials"]["nu"]
    G = E / (2.0 * (1.0 + nu))
    A = model["sections"]["area"]
    I_z = model["sections"]["Iz"]
    I_y = model["sections"]["Iy"]
    J_val = model["sections"]["J"]
    dx = model["_dx"]

    K_single = beam4_stiffness(E, G, A, I_z, I_y, J_val, dx)
    return np.tile(K_single[np.newaxis, :, :], (n_elem, 1, 1))


# ============================================================
# DOF 映射
# ============================================================

DOF_PER_NODE = 6
DOF_PER_ELEM = 12

# DOF 名称
DOF_NAMES = ["UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"]

# 各分量索引（在每节点 6DOF 中的位置）
DOF_UX = 0
DOF_UY = 1
DOF_UZ = 2
DOF_RX = 3
DOF_RY = 4
DOF_RZ = 5


def build_dof_map(elements: np.ndarray) -> np.ndarray:
    """
    构建单元 DOF → 全局 DOF 映射 (n_elem, 12)。

    全局 DOF 编号: global_dof = node_id * 6 + local_dof
    """
    n_elem = len(elements)
    dof_map = np.zeros((n_elem, DOF_PER_ELEM), dtype=int)
    for e in range(n_elem):
        n1, n2 = elements[e]
        for d in range(DOF_PER_NODE):
            dof_map[e, d] = n1 * DOF_PER_NODE + d
            dof_map[e, DOF_PER_NODE + d] = n2 * DOF_PER_NODE + d
    return dof_map


# ============================================================
# 诊断
# ============================================================

def print_beam4_info(model: dict):
    """打印 BEAM4 单元信息"""
    E = model["materials"]["E"]
    nu = model["materials"]["nu"]
    G = E / (2.0 * (1.0 + nu))
    sec = model["sections"]
    dx = model["_dx"]

    print(f"  BEAM4 单元 (3D, 12×12 刚度矩阵)")
    print(f"  截面: {sec['area']/sec.get('height',0.3):.3f} × {sec.get('height',0.3):.3f} m")
    print(f"  G = {G:.3e} Pa")
    print(f"  J = {sec['J']:.3e} m⁴")
    print(f"  I_z = {sec['Iz']:.3e} m⁴  (XY平面弯曲)")
    print(f"  I_y = {sec['Iy']:.3e} m⁴  (XZ平面弯曲)")
    print(f"  单元长度 dx = {dx:.4f} m")

    # 刚度矩阵数值范围
    K = beam4_stiffness(E, G, sec["area"], sec["Iz"], sec["Iy"], sec["J"], dx)
    diag = np.diag(K)
    print(f"  diag(K) 范围: [{diag.min():.2e}, {diag.max():.2e}]")
