"""
ansys_parser.py — ANSYS 导出文件解析器 + 原生模型生成器
==========================================================
两种使用方式：
  1. parse_ansys_export(dir) → 读取 ANSYS 导出的 7 个 .txt 文件
  2. build_native_model() → 直接在 Python 中生成相同模型（不需要 ANSYS）

输出统一为一个 dict，供 EBE-PCG 求解器使用。
"""

import re
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ============================================================
# 方式 1：解析 ANSYS 导出文件
# ============================================================

def parse_ansys_export(export_dir: str) -> dict:
    """
    读取 ANSYS 导出的全部 .txt 文件，返回统一模型数据结构。
    
    参数:
        export_dir: 包含 nodes.txt, elements.txt, ... 的目录
    
    返回:
        model: dict，结构见 DATA_FORMAT.md
    """
    if not os.path.isdir(export_dir):
        raise FileNotFoundError(f"目录不存在: {export_dir}")
    
    model = {}
    
    # (1) 节点坐标
    nodes_raw = _read_lines(os.path.join(export_dir, "nodes.txt"))
    n_nodes = _parse_first_int(nodes_raw[0])
    nodes = np.zeros((n_nodes, 3))
    for i, line in enumerate(nodes_raw[1:n_nodes+1]):
        parts = line.split()
        node_id = int(float(parts[0]))  # ANSYS 用 F 格式写整数，可能带 "1."
        idx = node_id - 1
        nodes[idx] = [float(parts[1]), float(parts[2]), float(parts[3])]
    model["n_nodes"] = n_nodes
    model["nodes"] = nodes
    
    # (2) 单元连接
    elems_raw = _read_lines(os.path.join(export_dir, "elements.txt"))
    n_elem = _parse_first_int(elems_raw[0])
    elements = np.zeros((n_elem, 2), dtype=int)
    for i, line in enumerate(elems_raw[1:n_elem+1]):
        parts = line.split()
        elem_id = int(float(parts[0]))
        n1 = int(float(parts[1])) - 1
        n2 = int(float(parts[2])) - 1
        elements[elem_id - 1] = [n1, n2]
    model["n_elem"] = n_elem
    model["elements"] = elements
    
    # (3) 材料属性
    mat_raw = _read_lines(os.path.join(export_dir, "materials.txt"))[0]
    model["materials"] = {
        "E":   _extract_float(mat_raw, r'E=\s*([\d\.E\+\-]+)'),
        "nu":  _extract_float(mat_raw, r'nu=\s*([\d\.E\+\-]+)'),
        "rho": _extract_float(mat_raw, r'rho=\s*([\d\.E\+\-]+)'),
    }
    
    # (4) 截面属性
    sec_raw = _read_lines(os.path.join(export_dir, "sections.txt"))[0]
    model["sections"] = {
        "area":   _extract_float(sec_raw, r'AREA=\s*([\d\.E\+\-]+)'),
        "Iz":     _extract_float(sec_raw, r'IZ=\s*([\d\.E\+\-]+)'),
        "height": _extract_float(sec_raw, r'H=\s*([\d\.E\+\-]+)'),
    }
    
    # (5) 边界条件
    bc_raw = _read_lines(os.path.join(export_dir, "boundary.txt"))
    n_bc = _parse_first_int(bc_raw[0])
    dof_map_label = {"UX": 0, "UY": 1, "ROTZ": 2}
    boundary = []
    for line in bc_raw[1:n_bc+1]:
        node_id = _extract_float(line, r'NODE\s+([\d\.]+)')
        dof_label = _extract_str(line, r'DOF=(\w+)')
        boundary.append((int(node_id) - 1, dof_map_label[dof_label]))
    model["boundary"] = boundary
    
    # (6) 荷载
    load_raw = _read_lines(os.path.join(export_dir, "loads.txt"))
    n_load = _parse_first_int(load_raw[0])
    load_dof_map = {"FX": 0, "FY": 1, "MZ": 2}
    loads = []
    for line in load_raw[1:n_load+1]:
        node_id = _extract_float(line, r'NODE\s+([\d\.]+)')
        dof_label = _extract_str(line, r'DOF=(\w+)')
        value = _extract_float(line, r'VALUE=\s*([\-\d\.E\+]+)')
        loads.append((int(node_id) - 1, load_dof_map[dof_label], value))
    model["loads"] = loads
    
    # (7) ANSYS 位移解（验证用）
    disp_path = os.path.join(export_dir, "displacements.txt")
    if os.path.exists(disp_path):
        disp_raw = _read_lines(disp_path)
        n_disp = _parse_first_int(disp_raw[0])
        ansys_disp = np.zeros((n_disp, 3))
        for i, line in enumerate(disp_raw[1:n_disp+1]):
            parts = line.split()
            node_id = int(float(parts[0]))
            ansys_disp[node_id - 1] = [float(parts[1]), float(parts[2]), float(parts[3])]
        model["ansys_disp"] = ansys_disp
    
    # 派生数据
    model["dof_per_node"] = 3   # BEAM3
    model["dof_per_elem"] = 6   # 2 节点 × 3DOF
    model["n_dofs"] = n_nodes * 3
    
    return model


# ============================================================
# 方式 2：纯 Python 生成相同模型（不需要 ANSYS）
# ============================================================

def build_native_model(
    L: float = 10.0,
    n_elem: int = 20,
    b_width: float = 0.2,
    b_height: float = 0.3,
    E: float = 210.0e9,
    nu: float = 0.3,
    rho: float = 7850.0,
    P: float = 0.0,
) -> dict:
    """
    直接在 Python 中生成简支梁模型 (BEAM4 3D 梁单元)。

    BEAM4: 每节点 6 DOF [UX, UY, UZ, ROTX, ROTY, ROTZ]
    单元刚度矩阵 12×12，自动约束面外自由度 (UZ, ROTX)。

    参数:
        L: 梁全长 (m)
        n_elem: 单元数
        b_width, b_height: 矩形截面 Y×Z 尺寸 (m)
        E: 弹性模量 (Pa), nu: 泊松比, rho: 密度 (kg/m³)
        P: 跨中集中力 (N)，负值向下
    """
    from beam_element import rect_section_props, DOF_PER_NODE, DOF_PER_ELEM

    n_nodes = n_elem + 1
    dx = L / n_elem

    # 节点坐标 (XY 平面：梁沿 X 轴)
    nodes = np.zeros((n_nodes, 3))
    nodes[:, 0] = np.arange(n_nodes) * dx

    # 单元连接（0-based）
    elements = np.column_stack([
        np.arange(n_elem),
        np.arange(1, n_elem + 1)
    ])

    # 截面属性 (BEAM4: 含 I_y, J)
    sec = rect_section_props(b_width, b_height)
    area = sec["area"]
    Iz = sec["Iz"]     # XY 平面弯曲
    Iy = sec["Iy"]     # XZ 平面弯曲
    J_val = sec["J"]   # 扭转常数

    mid_node = n_elem // 2

    # 边界条件 (BEAM4 DOF: 0=UX, 1=UY, 2=UZ, 3=ROTX, 4=ROTY, 5=ROTZ)
    boundary = [
        (0, 0),                    # node 0: UX=0
        (0, 1),                    # node 0: UY=0
        (0, 2),                    # node 0: UZ=0  (面外约束)
        (0, 3),                    # node 0: ROTX=0 (扭转约束)
        (n_nodes - 1, 1),          # node N: UY=0
        (n_nodes - 1, 2),          # node N: UZ=0  (面外约束)
        (n_nodes - 1, 3),          # node N: ROTX=0 (扭转约束)
    ]

    loads = [
        (mid_node, 1, P),          # 跨中节点: FY = P (DOF index 1)
    ]

    return {
        "n_nodes": n_nodes,
        "n_elem": n_elem,
        "dof_per_node": DOF_PER_NODE,    # 6
        "dof_per_elem": DOF_PER_ELEM,    # 12
        "n_dofs": n_nodes * DOF_PER_NODE,
        "nodes": nodes,
        "elements": elements,
        "materials": {"E": E, "nu": nu, "rho": rho},
        "sections": {"area": area, "Iz": Iz, "Iy": Iy, "J": J_val,
                     "height": b_height, "width": b_width},
        "boundary": boundary,
        "loads": loads,
        "ansys_disp": None,
        "_dx": dx,
        "_mid_node": mid_node,
        "_L": L,
    }


# ============================================================
# 工具函数
# ============================================================

def _read_lines(path: str) -> List[str]:
    """读取文件所有行，跳过空行"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_first_int(line: str) -> int:
    """提取行中的第一个整数"""
    match = re.search(r'[\-\d]+', line)
    if match:
        return int(float(match.group()))
    raise ValueError(f"无法解析整数: {line}")


def _extract_float(text: str, pattern: str) -> float:
    """用正则提取一个浮点数"""
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    raise ValueError(f"模式 '{pattern}' 在 '{text}' 中未匹配")


def _extract_str(text: str, pattern: str) -> str:
    """用正则提取一个字符串"""
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    raise ValueError(f"模式 '{pattern}' 在 '{text}' 中未匹配")


# ============================================================
# 模型信息打印
# ============================================================

def print_model_summary(model: dict):
    """打印模型摘要信息"""
    print("=" * 55)
    print("  ANSYS 简支梁模型摘要")
    print("=" * 55)
    print(f"  节点数:          {model['n_nodes']}")
    print(f"  单元数:          {model['n_elem']}")
    print(f"  总自由度数:      {model['n_dofs']}")
    print(f"  单元DOF/节点DOF: {model['dof_per_elem']}/{model['dof_per_node']}")
    
    mat = model["materials"]
    print(f"  E  (Pa):         {mat['E']:.3e}")
    print(f"  ν:               {mat['nu']:.4f}")
    print(f"  ρ  (kg/m³):      {mat['rho']:.1f}")
    
    sec = model["sections"]
    print(f"  面积 (m²):       {sec['area']:.4f}")
    print(f"  惯性矩 Iz (m⁴):  {sec['Iz']:.3e}")
    
    print(f"  边界条件 ({len(model['boundary'])} 个):")
    dof_names = {0: "UX", 1: "UY", 2: "UZ", 3: "ROTX", 4: "ROTY", 5: "ROTZ"}
    for node, dof in model["boundary"]:
        print(f"    节点 {node}: {dof_names.get(dof, str(dof))}=0")

    print(f"  荷载 ({len(model['loads'])} 个):")
    ldof_names = {0: "FX", 1: "FY", 2: "FZ", 3: "MX", 4: "MY", 5: "MZ"}
    for node, dof, val in model["loads"]:
        print(f"    节点 {node}: {ldof_names[dof]}={val:.0f} N")
    
    # 理论挠度
    L = model.get("_L", 10.0)
    E = mat["E"]
    Iz = sec["Iz"]
    P_abs = abs(model["loads"][0][2])
    delta_theory = P_abs * L**3 / (48 * E * Iz)
    print(f"\n  理论跨中挠度:    {delta_theory*1000:.4f} mm")
    print(f"  (δ = PL³/(48EI))")
    print("=" * 55)


# ============================================================
# 命令行测试
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  测试 1: 原生模型（纯 Python，无需 ANSYS）")
    print("=" * 55)
    model_native = build_native_model()
    print_model_summary(model_native)
    
    # 检查 ANSYS 导出文件是否存在
    export_dir = os.path.join(os.path.dirname(__file__), "..", "ansys")
    has_export = all(
        os.path.exists(os.path.join(export_dir, f))
        for f in ["nodes.txt", "elements.txt", "materials.txt",
                   "sections.txt", "boundary.txt", "loads.txt"]
    )
    
    if has_export:
        print("\n" + "=" * 55)
        print("  测试 2: 解析 ANSYS 导出文件")
        print("=" * 55)
        model_ansys = parse_ansys_export(export_dir)
        print_model_summary(model_ansys)
        
        # 验证一致性
        assert model_ansys["n_nodes"] == model_native["n_nodes"]
        assert model_ansys["n_elem"] == model_native["n_elem"]
        assert np.allclose(model_ansys["nodes"], model_native["nodes"])
        print("\n✅ 原生模型与 ANSYS 导出模型一致！")
    else:
        print("\n⚠ 未找到 ANSYS 导出文件（需在 ANSYS 中运行 simple_beam.inp）")
        print(f"  预期路径: {export_dir}/")
        print("  提示：原生模型可直接用于求解器，无需 ANSYS。")
