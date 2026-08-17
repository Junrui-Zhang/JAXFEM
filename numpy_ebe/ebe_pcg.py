"""
ebe_pcg.py — NumPy EBE-PCG 求解器（串行循环版, BEAM4）
========================================================
支持 BEAM4 3D 梁单元 (12×12 刚度矩阵, 6 DOF/节点)。

BEAM4 DOF 顺序 (每节点): [UX, UY, UZ, ROTX, ROTY, ROTZ]
"""

import numpy as np

DOF_PER_NODE = 6
DOF_PER_ELEM = 12


def build_dof_map(elements: np.ndarray, dof_per_node: int = DOF_PER_NODE) -> np.ndarray:
    """单元 DOF → 全局 DOF 映射 (n_elem, 12)"""
    n_elem = len(elements)
    dof_map = np.zeros((n_elem, 2 * dof_per_node), dtype=int)
    for e in range(n_elem):
        n1, n2 = elements[e]
        for d in range(dof_per_node):
            dof_map[e, d] = n1 * dof_per_node + d
            dof_map[e, dof_per_node + d] = n2 * dof_per_node + d
    return dof_map


def build_adjacency(elements: np.ndarray) -> list:
    """EBE 邻接表（链式梁：每单元最多 2 个邻居）"""
    n_elem = len(elements)
    adj = [[] for _ in range(n_elem)]
    for e in range(n_elem):
        n_start, n_end = elements[e]
        for other in range(n_elem):
            if other == e:
                continue
            o_start, o_end = elements[other]
            if o_start == n_start or o_end == n_start or o_start == n_end or o_end == n_end:
                adj[e].append(other)
    return adj


def compute_element_stiffness(model: dict) -> np.ndarray:
    """BEAM4 单元刚度矩阵 (n_elem, 12, 12)"""
    from beam_element import beam4_stiffness

    n_elem = model["n_elem"]
    E = model["materials"]["E"]
    nu = model["materials"]["nu"]
    G = E / (2.0 * (1.0 + nu))
    A = model["sections"]["area"]
    Iz = model["sections"]["Iz"]
    Iy = model["sections"]["Iy"]
    J_val = model["sections"]["J"]
    dx = float(model["nodes"][1, 0]) - float(model["nodes"][0, 0])

    K_single = beam4_stiffness(E, G, A, Iz, Iy, J_val, dx)
    return np.tile(K_single[np.newaxis, :, :], (n_elem, 1, 1))


def build_force_vector(model: dict) -> np.ndarray:
    """全局力向量"""
    F = np.zeros(model["n_dofs"])
    for node, dof, val in model["loads"]:
        F[node * model["dof_per_node"] + dof] = val
    return F


def get_constrained_dofs(model: dict) -> list:
    """约束 DOF 索引列表"""
    result = []
    for node, dof in model["boundary"]:
        result.append(node * model["dof_per_node"] + dof)
    return sorted(result)


def apply_boundary_to_elements(K_e: np.ndarray, dof_map: np.ndarray,
                                constrained_dofs: list, large: float = 1e12) -> np.ndarray:
    """单元级施加边界条件"""
    n_elem, _, dof_per_elem = K_e.shape
    K_bc = K_e.copy()
    mask = np.zeros((n_elem, dof_per_elem), dtype=bool)
    for g in constrained_dofs:
        mask = mask | (dof_map == g)
    for e in range(n_elem):
        for local_dof in range(dof_per_elem):
            if mask[e, local_dof]:
                K_bc[e, local_dof, local_dof] = large
    return K_bc


# ============================================================
# EBE 核心运算（NumPy 串行版）
# ============================================================

_HALF = 6  # DOF_PER_ELEM // 2


def assemble_global(v_e: np.ndarray, dof_map: np.ndarray, n_dofs: int) -> np.ndarray:
    """单元向量 → 全局向量"""
    result = np.zeros(n_dofs)
    n_elem = len(v_e)
    for e in range(n_elem):
        for i in range(v_e.shape[1]):
            result[dof_map[e, i]] += v_e[e, i]
    return result


def scatter_global(v_global: np.ndarray, dof_map: np.ndarray) -> np.ndarray:
    """全局向量 → 单元向量（按份额分配）"""
    n_elem = dof_map.shape[0]
    n_dofs = len(v_global)
    dof_per_elem = dof_map.shape[1]

    share_count = np.zeros(n_dofs, dtype=int)
    for e in range(n_elem):
        for i in range(dof_per_elem):
            share_count[dof_map[e, i]] += 1

    v_e = np.zeros((n_elem, dof_per_elem))
    for e in range(n_elem):
        for i in range(dof_per_elem):
            g = dof_map[e, i]
            v_e[e, i] = v_global[g] / share_count[g]
    return v_e


def real_to_fake(v_e: np.ndarray, adj: list) -> np.ndarray:
    """
    真向量 → 伪向量（BEAM4: 12 DOF/元素）。

    v^(e) = v^e + Σ_{相邻单元 j} v^j 在共享节点处的贡献
    """
    n_elem = len(v_e)
    v_fake = v_e.copy()

    for e in range(n_elem):
        for other in adj[e]:
            if other < e:
                # other 是左邻居: other 的右节点(HALF:12) → e 的左节点(0:HALF)
                v_fake[e, 0:_HALF] += v_e[other, _HALF:2*_HALF]
            else:
                # other 是右邻居: other 的左节点(0:HALF) → e 的右节点(HALF:12)
                v_fake[e, _HALF:2*_HALF] += v_e[other, 0:_HALF]

    return v_fake


def ebe_inner_product(v_e: np.ndarray, adj: list) -> float:
    """EBE 内积: Σ_e (v^e)^T v^(e)"""
    v_fake = real_to_fake(v_e, adj)
    total = 0.0
    for e in range(len(v_e)):
        total += np.dot(v_e[e], v_fake[e])
    return total


def ebe_pAp(p_e: np.ndarray, K_e: np.ndarray, adj: list) -> float:
    """EBE (p, Ap): Σ_e (p^(e))^T K^e p^(e)"""
    p_fake = real_to_fake(p_e, adj)
    total = 0.0
    for e in range(len(p_e)):
        Kp = K_e[e] @ p_fake[e]
        total += np.dot(p_fake[e], Kp)
    return total


# ============================================================
# EBE-PCG 求解器（NumPy 串行版）
# ============================================================

def ebe_pcg_solve(K_e: np.ndarray, F_global: np.ndarray,
                  dof_map: np.ndarray, adj: list,
                  constrained_dofs: list,
                  tol: float = 1e-8, max_iter: int = 2000,
                  verbose: bool = True) -> tuple:
    """
    EBE-PCG 求解器 — NumPy 串行版 (BEAM4, 12 DOF/元素)。
    """
    n_elem = len(K_e)
    n_dofs = len(F_global)
    dof_per_elem = K_e.shape[2]

    F_e = scatter_global(F_global, dof_map)
    x_e = np.zeros((n_elem, dof_per_elem))
    r_e = F_e.copy()

    # 对角预处理器
    M_inv_e = np.zeros((n_elem, dof_per_elem))
    for e in range(n_elem):
        for i in range(dof_per_elem):
            d = K_e[e, i, i]
            if abs(d) > 1e-15 and dof_map[e, i] not in constrained_dofs:
                M_inv_e[e, i] = 1.0 / d

    h_e = M_inv_e * r_e
    gamma = _ebe_inner_rh(r_e, h_e, adj)
    F_norm = np.linalg.norm(F_global)

    if gamma < 1e-30:
        if verbose:
            print("  [EBE-PCG] 初始残量接近零")
        return assemble_global(x_e, dof_map, n_dofs), 0

    p_e = h_e.copy()

    n_iter = 0
    for k in range(max_iter):
        pAp_val = ebe_pAp(p_e, K_e, adj)

        if abs(pAp_val) < 1e-16:
            if verbose:
                print(f"  [EBE-PCG] (p,Ap)≈0 at iter {k+1}, stopping")
            break

        alpha = gamma / pAp_val

        # x += α p,  r -= α K^(e) p^(e)
        x_e += alpha * p_e

        p_fake = real_to_fake(p_e, adj)
        for e in range(n_elem):
            r_e[e] -= alpha * (K_e[e] @ p_fake[e])

        # 收敛检查
        r_global = assemble_global(r_e, dof_map, n_dofs)
        rel_res = np.linalg.norm(r_global) / max(F_norm, 1.0)
        n_iter = k + 1

        if rel_res < tol:
            if verbose:
                print(f"  [EBE-PCG] 收敛于第 {n_iter} 步, "
                      f"||r||/||F|| = {rel_res:.2e}")
            break

        # CG 更新
        h_e = M_inv_e * r_e
        gamma_new = _ebe_inner_rh(r_e, h_e, adj)
        beta = gamma_new / max(gamma, 1e-30)
        p_e = h_e + beta * p_e
        gamma = gamma_new

    x_global = assemble_global(x_e, dof_map, n_dofs)
    return x_global, n_iter


def _ebe_inner_rh(r_e: np.ndarray, h_e: np.ndarray, adj: list) -> float:
    """(r, h) = Σ_e (r^e)^T h^(e)"""
    h_fake = real_to_fake(h_e, adj)
    total = 0.0
    for e in range(len(r_e)):
        total += np.dot(r_e[e], h_fake[e])
    return total
