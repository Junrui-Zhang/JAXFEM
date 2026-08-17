"""
ebe_pcg.py — JAX EBE-PCG 求解器（vmap 并行版本, BEAM4）
=========================================================
支持 BEAM4 3D 梁单元 (12×12 刚度矩阵, 6 DOF/节点)。
通过 jax.vmap 实现单元间并行计算。

与 NumPy 版本相同的 EBE 算法:
  - NumPy 版: for e in range(n_elem): ...  → O(n_elem) 串行
  - JAX 版:   vmap(fn)(data)               → GPU 并行

BEAM4 DOF 顺序 (每节点): [UX, UY, UZ, ROTX, ROTY, ROTZ]
"""

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax import vmap, jit

_I = jnp.int32

# ---- BEAM4 常量 ----
DOF_PER_NODE = 6
DOF_PER_ELEM = 12


def build_dof_map(elements: jnp.ndarray, dof_per_node: int = DOF_PER_NODE) -> jnp.ndarray:
    """构建单元 DOF → 全局 DOF 映射 (n_elem, 12)"""
    n1 = elements[:, 0]
    n2 = elements[:, 1]
    offsets = jnp.arange(dof_per_node, dtype=jnp.int32)

    first_half = n1[:, None] * dof_per_node + offsets[None, :]
    second_half = n2[:, None] * dof_per_node + offsets[None, :]

    return jnp.concatenate([first_half, second_half], axis=1)


def build_adjacency(elements: jnp.ndarray) -> list:
    """EBE 邻接表（兼容旧接口）"""
    n_elem = elements.shape[0]
    adj = [[] for _ in range(n_elem)]
    for e in range(n_elem):
        n_start, n_end = int(elements[e, 0]), int(elements[e, 1])
        for other in range(n_elem):
            if other == e:
                continue
            o_start, o_end = int(elements[other, 0]), int(elements[other, 1])
            if o_start == n_start or o_end == n_start:
                adj[e].append(other)
            elif o_start == n_end or o_end == n_end:
                adj[e].append(other)
    return adj


def build_adjacency_matrices(elements: jnp.ndarray):
    """
    EBE 邻居索引数组 (BEAM4: 每节点 6 DOF)。

    替换稠密邻接矩阵为 O(1) 索引构建（梁链特化）:
      - 每个单元至多 1 个左邻 + 1 个右邻
      - 使用哨兵 n_elem 表示「无邻居」

    返回:
        left_idx:  (n_elem,) int32, left_idx[e]=左邻索引或 n_elem
        right_idx: (n_elem,) int32, right_idx[e]=右邻索引或 n_elem

    注: 对通用网格，可替换为基于节点共享的 O(n²) 构建。
    """
    n_elem = elements.shape[0]
    # 梁链: 单元 e 的左邻 = e-1, 右邻 = e+1
    left_idx = jnp.where(jnp.arange(n_elem) > 0,
                         jnp.arange(n_elem) - 1, n_elem)
    right_idx = jnp.where(jnp.arange(n_elem) < n_elem - 1,
                          jnp.arange(n_elem) + 1, n_elem)
    return left_idx, right_idx


def compute_element_stiffness(model: dict) -> jnp.ndarray:
    """BEAM4 单元刚度矩阵 (n_elem, 12, 12) — JAX 版本"""
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

    K_single_np = beam4_stiffness(E, G, A, Iz, Iy, J_val, dx)
    K_single = jnp.array(K_single_np)
    return jnp.tile(K_single[jnp.newaxis, :, :], (n_elem, 1, 1))


def build_force_vector(model: dict) -> jnp.ndarray:
    """构建全局力向量"""
    F = jnp.zeros(model["n_dofs"])
    for node, dof, val in model["loads"]:
        g = _I(int(node) * model["dof_per_node"] + int(dof))
        F = F.at[g].set(val)
    return F


def get_constrained_dofs(model: dict) -> list:
    """约束 DOF 索引列表"""
    result = []
    for node, dof in model["boundary"]:
        result.append(node * model["dof_per_node"] + dof)
    return sorted(result)


def apply_boundary_to_elements(K_e: jnp.ndarray, dof_map: jnp.ndarray,
                                constrained_dofs: list, large: float = 1e12):
    """单元级施加边界条件（向量化）"""
    n_elem = K_e.shape[0]
    dof_per_elem = K_e.shape[2]
    constrained_mask = compute_constrained_mask(dof_map, constrained_dofs, n_elem, dof_per_elem)

    diag_add = jnp.where(constrained_mask, large, 0.0)
    add_3d = vmap(jnp.diag)(diag_add)
    return K_e + add_3d


# ============================================================
# EBE 核心运算（JAX 并行 + 向量化版）
# ============================================================

# half 大小 = DOF_PER_ELEM // 2 = 6
_HALF = 6


@jit
def real_to_fake(v_e: jnp.ndarray, left_idx: jnp.ndarray, right_idx: jnp.ndarray) -> jnp.ndarray:
    """
    真向量 → 伪向量（BEAM4, O(n) 索引 gather）。

    用邻居索引直接收集相邻单元的共享节点贡献:
      v_fake[e, 0:6]  += v_e[left_idx[e], 6:12]   (左邻右半 → 当前左半)
      v_fake[e, 6:12] += v_e[right_idx[e], 0:6]   (右邻左半 → 当前右半)

    哨兵 n_elem → 取零行（无邻居时加零）。
    替换原 O(n²) 稠密邻接矩阵乘法 adj @ v。
    """
    # 末尾补一行零，哨兵索引 n_elem 映射到此行
    pad_row = jnp.zeros((1,) + v_e.shape[1:], dtype=v_e.dtype)
    v_pad = jnp.concatenate([v_e, pad_row], axis=0)

    left_contrib = v_pad[left_idx, _HALF:2*_HALF]  # (n_elem, 6)
    right_contrib = v_pad[right_idx, 0:_HALF]       # (n_elem, 6)

    v_fake = v_e
    v_fake = v_fake.at[:, 0:_HALF].add(left_contrib)
    v_fake = v_fake.at[:, _HALF:2*_HALF].add(right_contrib)
    return v_fake


@jit
def ebe_inner_product(v_e: jnp.ndarray, left_idx: jnp.ndarray, right_idx: jnp.ndarray) -> jnp.ndarray:
    """EBE 内积: Σ_e (v^e)^T v^(e) — 返回标量"""
    v_fake = real_to_fake(v_e, left_idx, right_idx)
    dots = vmap(jnp.dot)(v_e, v_fake)
    return jnp.sum(dots)


@jit
def ebe_pAp(p_e: jnp.ndarray, K_e: jnp.ndarray,
            left_idx: jnp.ndarray, right_idx: jnp.ndarray) -> jnp.ndarray:
    """EBE (p, Ap): Σ_e (p^(e))^T K^e p^(e) — 返回标量"""
    p_fake = real_to_fake(p_e, left_idx, right_idx)

    def per_element(p, K):
        Kp = K @ p
        return jnp.dot(p, Kp)

    per_elem_results = vmap(per_element)(p_fake, K_e)
    return jnp.sum(per_elem_results)


def assemble_global(v_e: jnp.ndarray, dof_map: jnp.ndarray, n_dofs: int) -> jnp.ndarray:
    """单元向量 → 全局向量（向量化）"""
    flat_indices = dof_map.ravel()
    flat_values = v_e.ravel()
    return jnp.zeros(n_dofs).at[flat_indices].add(flat_values)


def scatter_global(v_global: jnp.ndarray, dof_map: jnp.ndarray) -> jnp.ndarray:
    """全局向量 → 单元向量（按份额分配）"""
    n_elem = dof_map.shape[0]
    n_dofs = len(v_global)
    dof_per_elem = dof_map.shape[1]

    flat_indices = dof_map.ravel()

    share_count = jnp.zeros(n_dofs, dtype=jnp.int32)
    share_count = share_count.at[flat_indices].add(
        jnp.ones(n_elem * dof_per_elem, dtype=jnp.int32)
    )

    scattered_values = v_global[flat_indices] / share_count[flat_indices]
    return scattered_values.reshape(n_elem, dof_per_elem)


# ============================================================
# EBE-PCG 求解器（JAX 全 JIT 向量化版）
# ============================================================

def ebe_pcg_solve(K_e: jnp.ndarray, F_global: jnp.ndarray,
                  dof_map: jnp.ndarray,
                  left_idx: jnp.ndarray, right_idx: jnp.ndarray,
                  constrained_dofs: list,
                  tol: float = 1e-8, max_iter: int = 2000,
                  verbose: bool = True):
    """
    EBE-PCG 求解器（BEAM4, 12 DOF/元素）。
    """
    n_elem = K_e.shape[0]
    n_dofs = len(F_global)
    dof_per_elem = K_e.shape[2]  # 12

    F_e = scatter_global(F_global, dof_map)
    x_e = jnp.zeros((n_elem, dof_per_elem))
    r_e = F_e

    # 对角预处理器
    diag_K = jnp.diagonal(K_e, axis1=1, axis2=2)
    constrained_mask = compute_constrained_mask(dof_map, constrained_dofs, n_elem, dof_per_elem)
    near_zero_mask = jnp.abs(diag_K) < 1e-15
    M_inv_e = jnp.where(constrained_mask | near_zero_mask, 0.0, 1.0 / diag_K)

    h_e = M_inv_e * r_e
    gamma = _ebe_inner_rh(r_e, h_e, left_idx, right_idx)
    F_norm = jnp.linalg.norm(F_global)

    if float(gamma) < 1e-30:
        if verbose:
            print("  [EBE-PCG JAX] 初始残量接近零")
        return assemble_global(x_e, dof_map, n_dofs), 0

    p_e = h_e
    _n_dofs = n_dofs
    _n_elem = n_elem

    @jit
    def _pcg_step(x_e, r_e, p_e, h_e, gamma):
        pAp_val = ebe_pAp(p_e, K_e, left_idx, right_idx)
        safe_beta = jnp.where(jnp.abs(pAp_val) < 1e-16, 1.0, pAp_val)
        alpha = gamma / safe_beta

        alpha_vec = jnp.full((_n_elem,), alpha)
        x_e_new = x_e + alpha_vec[:, None] * p_e

        p_fake = real_to_fake(p_e, left_idx, right_idx)
        r_e_new = r_e - alpha_vec[:, None] * vmap(lambda K, p: K @ p)(K_e, p_fake)

        r_global = assemble_global(r_e_new, dof_map, _n_dofs)
        rel_res = jnp.linalg.norm(r_global) / jnp.maximum(F_norm, 1.0)

        h_e_new = M_inv_e * r_e_new
        gamma_new = _ebe_inner_rh(r_e_new, h_e_new, left_idx, right_idx)
        beta_cg = gamma_new / jnp.maximum(gamma, 1e-30)

        p_e_new = h_e_new + beta_cg * p_e

        return x_e_new, r_e_new, p_e_new, h_e_new, gamma_new, pAp_val, rel_res

    n_iter = 0
    for k in range(max_iter):
        x_e, r_e, p_e, h_e, gamma, pAp_val, rel_res = _pcg_step(
            x_e, r_e, p_e, h_e, gamma
        )
        n_iter = k + 1

        if float(jnp.abs(pAp_val)) < 1e-16:
            if verbose:
                print(f"  [EBE-PCG JAX] (p,Ap)≈0 at iter {n_iter}, stopping")
            break

        if float(rel_res) < tol:
            if verbose:
                print(f"  [EBE-PCG JAX] 收敛于第 {n_iter} 步, "
                      f"||r||/||F|| = {float(rel_res):.2e}")
            break

    x_global = assemble_global(x_e, dof_map, n_dofs)
    return x_global, n_iter


def compute_constrained_mask(dof_map: jnp.ndarray, constrained_dofs: list,
                              n_elem: int, dof_per_elem: int = DOF_PER_ELEM) -> jnp.ndarray:
    """约束 DOF 布尔掩码 (n_elem, dof_per_elem)"""
    mask = jnp.zeros((n_elem, dof_per_elem), dtype=bool)
    for g in constrained_dofs:
        mask = mask | (dof_map == _I(g))
    return mask


def _ebe_inner_rh(r_e: jnp.ndarray, h_e: jnp.ndarray,
                  left_idx: jnp.ndarray, right_idx: jnp.ndarray) -> jnp.ndarray:
    """(r, h) = Σ_e (r^e)^T h^(e)"""
    h_fake = real_to_fake(h_e, left_idx, right_idx)
    dots = vmap(jnp.dot)(r_e, h_fake)
    return jnp.sum(dots)
