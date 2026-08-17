# ANSYS → Python 单元信息导出格式说明

## 概述

ANSYS 将简支梁模型导出为 7 个文本文件，Python 端通过 `ansys_parser.py` 读取并转为字典/数组供 EBE-PCG 求解器使用。

---

## 文件清单

| 文件名 | 内容 | EBE-PCG 用途 |
|--------|------|-------------|
| `nodes.txt` | 节点坐标 | 计算单元长度、方向向量 |
| `elements.txt` | 单元-节点连接表 | 构建 DOF 映射、邻接表 |
| `materials.txt` | 材料属性 (E, ν, ρ) | 计算 K_e、M_e、C_e |
| `sections.txt` | 截面属性 (A, Iz, H) | 计算 K_e |
| `boundary.txt` | 边界条件 | 修改 K_e 对角元素、置零 F |
| `loads.txt` | 节点荷载 | 构建力向量 F |
| `displacements.txt` | ANSYS 解 | Python 结果验证 |

---

## 各文件格式详解

### 1. `nodes.txt`

```
NNODES=       11.     ← 第1行：节点总数（用于内存预分配）
     1    0.000000    0.000000    0.000000
     2    1.000000    0.000000    0.000000
     ...
    11   10.000000    0.000000    0.000000
```

- 第 1 行：`NNODES=<N>`（float 格式）
- 后续每行：`<node_id> <x> <y> <z>`（I6, 3F14.6）
- node_id 从 1 开始（ANSYS 惯例）

### 2. `elements.txt`

```
NELEMS=    10.
     1     1     2     1     1     1
     2     2     3     1     1     1
     ...
    10    10    11     1     1     1
```

- 第 1 行：`NELEMS=<N>`
- 后续每行：`<elem_id> <node_i> <node_j> <type_id> <real_id> <mat_id>`
- 此模型所有单元 type_id=1, real_id=1, mat_id=1

### 3. `materials.txt`

```
1_mat E=  0.210000E+12 nu=0.3000 rho=7850.00
```

- 单行，需要正则解析出数值
- 获取：`E`, `nu`, `rho`
- 多材料模型需扩展

### 4. `sections.txt`

```
1_sec AREA=  0.600000E-01 IZ=  0.450000E-03 H=0.300
```

- 单行，需要正则解析
- 获取：`AREA`, `IZ`（惯性矩）, `H`（梁高）

### 5. `boundary.txt`

```
NBC=   3.
NODE       1. DOF=UX
NODE       1. DOF=UY
NODE      11. DOF=UY
```

- 第 1 行：`NBC=<N>`
- 后续每行：`NODE <node_id>. DOF=<label>`
- DOF label: `UX`, `UY`, `ROTZ`（对 BEAM3 只有这 3 种）
- Python 中映射：UX→0, UY→1, ROTZ→2

### 6. `loads.txt`

```
NLOAD=   1.
NODE       6. DOF=FY VALUE= -0.100000E+05
```

- 第 1 行：`NLOAD=<N>`
- 后续每行：`NODE <node_id>. DOF=<label> VALUE=<force>`
- DOF label: `FX`, `FY`, `MZ`

### 7. `displacements.txt`（验证用）

```
NDISP=    11.
     1  0.000000E+00 -0.000000E+00  0.971556E-04
     2  0.000000E+00 -1.448675E-03  0.868643E-03
     ...
```

- 第 1 行：`NDISP=<N>`
- 后续每行：`<node_id> <UX> <UY> <ROTZ>`（3E16.6）

---

## Python 解析后的数据结构

```python
model = {
    "n_nodes": 11,
    "n_elem": 10,
    "dof_per_node": 3,
    "dof_per_elem": 6,
    "nodes": np.array([[x1,y1,z1], ...]),      # (n_nodes, 3)
    "elements": np.array([[n1, n2], ...]),       # (n_elem, 2)  0-based
    "materials": {"E": 210e9, "nu": 0.3, "rho": 7850},
    "sections": {"area": 0.06, "Iz": 0.00045, "height": 0.3},
    "boundary": [(node_id, dof_idx), ...],       # dof_idx: 0=UX,1=UY,2=ROTZ
    "loads": [(node_id, dof_idx, value), ...],
    "ansys_disp": np.array([[ux,uy,rotz], ...])  # ANSYS 解（验证用）
}

# 求解器据此推导：
#   dof_map[elem, local] → global_dof      (n_elem, 6)
#   adj[elem] → [neighbor_elem, ...]       邻接表
#   K_e[elem] → 6×6 刚度矩阵               在 Python/JAX 中计算
```

---

## ANSYS DOF → Python DOF 索引映射

| ANSYS Label | Python 含义 | 节点内偏移 |
|-------------|------------|-----------|
| UX | 轴向位移 | 0 |
| UY | 竖向位移 | 1 |
| ROTZ | 面内转角 | 2 |

全局 DOF 编号 = `node_id * 3 + local_offset`（0-based node_id）

---

## 扩展到 3D 梁（BEAM188）时的变化

当实际项目使用 BEAM188（3D 梁，每节点 6DOF）时：

| 新增 DOF | Python 偏移 |
|----------|------------|
| UZ | 2 |
| ROTX | 3 |
| ROTY | 4 |
| ROTZ | 5 |

单元 DOF = 12（2 节点 × 6DOF），K_e 为 12×12。

额外需要的截面参数：`Iy`, `J`（扭转常数）。

导出文件格式不变，只需扩展解析器。
