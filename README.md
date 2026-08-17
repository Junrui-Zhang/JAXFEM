# 简支梁 EBE-PCG 可行性验证 Demo

## 概述

本 demo 是硕士论文《大跨度桥梁抖振时域分析的GPU加速并行有限元算法》步骤 1.4 的可行性预验证。

验证目标：对一简支梁模型，分别用 NumPy 串行和 JAX GPU 并行两种方式实现 EBE-PCG 求解器，对比结果精度和计算速度。

## 模型参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 跨度 L | 10 m | 梁全长 |
| 单元数 | 10 | 均匀划分 |
| 截面 | 0.2m × 0.3m | 矩形截面 |
| 弹性模量 E | 210 GPa | 钢材 |
| 密度 ρ | 7850 kg/m³ | - |
| 集中力 P | -10 kN | 跨中竖向 |
| 支座 | 简支 | 两端约束 UY |
| 单元类型 | 2D Euler-Bernoulli 梁 | 每节点 3DOF (UX, UY, ROTZ) |

**理论跨中挠度**：δ = PL³/(48EI) = **2.204 mm**（向下）

## 目录结构

```
demo/
├── README.md                    # 本文件
├── ansys/
│   └── simple_beam.inp          # ANSYS APDL 建模脚本（含导出）
├── numpy_ebe/
│   ├── beam_model.py            # 梁模型定义 + 全局矩阵组装
│   ├── ebe_pcg.py               # NumPy EBE-PCG 求解器（串行循环）
│   └── main.py                  # 运行入口，输出结果对比
└── jax_ebe/
    ├── beam_model.py            # 梁模型定义（JAX 版本）
    ├── ebe_pcg.py               # JAX EBE-PCG 求解器（vmap 并行）
    └── main.py                  # 运行入口，输出结果和加速比
```

## 运行方式

### 1. NumPy 串行版本
```bash
cd numpy_ebe
python3 main.py
```

### 2. JAX 并行版本
```bash
cd jax_ebe
python3 main.py
```

### 3. ANSYS 验证（需要安装 ANSYS）
```bash
cd ansys
# 启动 ANSYS APDL，读取 simple_beam.inp
# File → Read Input from... → simple_beam.inp
```

## 预期验证结果

| 指标 | NumPy EBE-PCG | JAX EBE-PCG | ANSYS 参考 | 解析解 |
|------|:---:|:---:|:---:|:---:|
| 跨中挠度 | ≈2.204 mm | ≈2.204 mm | ≈2.204 mm | 2.204 mm |
| 求解迭代次数 | - | - | - | - |

## 关键概念

### EBE (Element-by-Element) 核心运算

- **真向量** $x^e$：只包含单元自身节点的值（6维）
- **伪向量** $x^{(e)}$：包含单元自身 + 相邻单元共享节点的值
- **内积分解**：$(r,r) = \sum_{e=1}^E (r^e)^T r^{(e)}$
- **$p^TAp$ 分解**：$(p,Ap) = \sum_{e=1}^E (p^{(e)})^T A^e p^{(e)}$

### 邻接关系

对梁单元链（单元 e 连接节点 e 和 e+1）：
- 单元 e 与单元 e-1 共享节点 e
- 单元 e 与单元 e+1 共享节点 e+1
