# Demo 使用手册

> 简支梁 EBE-PCG 求解器 — NumPy（串行）与 JAX（GPU 并行）双版本

---

## 快速开始（零依赖，无需 ANSYS）

```bash
# 使用 jaxfem 虚拟环境
PYTHON=/home/zjr/anaconda3/envs/jaxfem/bin/python3
DEMO="/home/zjr/桌面/Keyan/Master's paper/研究路线/demo"

# NumPy 版本（串行 EBE-PCG）
$PYTHON "$DEMO/numpy_ebe/main.py"

# JAX GPU 版本（vmap 并行 EBE-PCG）
$PYTHON "$DEMO/jax_ebe/main.py"
```

两个脚本都会：
1. 在 Python 内自动生成一个 10 单元简支梁模型
2. 用 NumPy 直接求解（`np.linalg.solve`，作为基准）
3. 用 EBE-PCG 迭代求解
4. 对比解析解、直接解、EBE-PCG 解三者的跨中挠度和全部节点位移

---

## 输出示例

```
============================================================
  JAX EBE-PCG 简支梁求解 — 并行可行性验证
============================================================
  JAX 设备: [CudaDevice(id=0)]
  ✅ GPU 可用 — cuda:0
=======================================================
  ANSYS 简支梁模型摘要
=======================================================
  节点数:          11
  单元数:          10
  总自由度数:      33
  ...
  理论跨中挠度:    2.2046 mm
  (δ = PL³/(48EI))

[1] NumPy 直接求解（基准）...
    跨中挠度: -2.2046 mm  |  理论: 2.2046 mm

[2] JAX EBE-PCG 求解（vmap 单元并行）...
    迭代次数:     10
    跨中挠度 UY:  -2.2046 mm
    vs 直接解:    0.000000 %

============================================================
  ✅ JAX EBE-PCG 验证通过！与直接解一致（偏差 0.000000%）
============================================================
```

---

## 模型参数自定义

编辑 `main.py` 中 `build_native_model()` 的调用：

```python
model = build_native_model(
    L=10.0,          # 梁全长 (m)
    n_elem=100,      # 单元数（越多规模越大）
    b_width=0.2,     # 矩形截面宽 (m)
    b_height=0.3,    # 矩形截面高 (m)
    E=210.0e9,       # 弹性模量 (Pa)
    nu=0.3,          # 泊松比
    rho=7850.0,      # 密度 (kg/m³)
    P=-10000.0,      # 跨中集中力 (N)，负值 = 向下
)
```

---

## 模型数据来源：两种方式

### 方式一：纯 Python 生成（默认，推荐）

```python
from ansys.ansys_parser import build_native_model
model = build_native_model(n_elem=50)  # 无需 ANSYS
```

两个 `main.py` 默认使用此方式。

### 方式二：ANSYS 导出（为后续大规模桥梁模型准备）

| 步骤 | 操作 |
|------|------|
| 1 | 在 ANSYS Mechanical APDL 中：`File → Read Input from → ansys/simple_beam.inp` |
| 2 | 求解完成后，工作目录下生成 6-7 个 `.txt` 文件 |
| 3 | 将 `.txt` 复制到 `demo/ansys/` 目录 |
| 4 | 在 `main.py` 中改用：`model = parse_ansys_export("ansys/")` |

导出文件格式详见 `ansys/DATA_FORMAT.md`。

---

## 项目结构

```
demo/
├── USAGE.md                  # 本手册
├── README.md                 # 项目说明
├── ansys/
│   ├── simple_beam.inp       # ANSYS APDL 脚本（仅在 ANSYS 中运行）
│   ├── ansys_parser.py       # 模型解析器（两种构建方式）
│   └── DATA_FORMAT.md        # 导出 .txt 文件格式说明
├── numpy_ebe/
│   ├── ebe_pcg.py            # EBE-PCG 求解器（纯 NumPy，单元间 for 循环串行）
│   ├── main.py               # 主程序入口
│   └── __init__.py
└── jax_ebe/
    ├── ebe_pcg.py            # EBE-PCG 求解器（JAX + vmap 并行 + @jit 编译）
    ├── main.py               # 主程序入口
    └── __init__.py
```

---

## 两个求解器的区别

| | NumPy 版 (`numpy_ebe/`) | JAX 版 (`jax_ebe/`) |
|---|---|---|
| 单元矩阵计算 | `for e in range(n_elem)` 串行 | `jax.vmap` 全部单元同时计算 |
| EBE-PCG 迭代 | Python `for` 循环 | `jax.lax.while_loop`（JIT 内编译） |
| 邻接信息交换 | Python `list` 索引 | `jnp.take` 向量化索引 |
| 运行环境 | CPU（NumPy） | GPU（CUDA），自动回退 CPU |
| 小模型耗时 | ~1.7 ms | ~1.5 s（JIT 编译开销为主） |
| 适用场景 | 调试、验证 | 大规模模型（DOF > 10⁴ 才体现 GPU 优势） |

> **注意**：当前 33 DOF 模型太小，JAX 版的 1.5s 耗时几乎全部是 JIT 编译时间，
> 实际计算在微秒级。模型规模达到 10⁴~10⁵ DOF 时 GPU 加速才会显著。

---

## 环境

| 项目 | 值 |
|------|-----|
| Python | 3.11.14 (`jaxfem` conda env) |
| JAX | 0.9.0 + CUDA jaxlib |
| GPU | NVIDIA GeForce RTX 4060 (8GB) |
| CUDA | 13.0, Driver 580.173.02 |
| JAX float64 | 已启用 |
