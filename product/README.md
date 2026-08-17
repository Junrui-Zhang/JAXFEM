# JAXFEM · 简支梁 EBE-PCG GPU 并行有限元演示平台

> 硕士论文《大跨度桥梁抖振时域分析的 GPU 加速并行有限元算法》可行性验证的产品化封装:
> 把 JAX GPU 并行 EBE-PCG 求解器包装成 **浏览器可交互的 Web 应用**,支持参数调节、
> 实时求解、2D 曲线与 3D 变形云图,用于求职演示与答辩展示。

![界面占位](docs/screenshot.png)

---

## 快速开始

```bash
cd "/home/zjr/桌面/求职/自研产品/JAXFEM"
/home/zjr/anaconda3/envs/jaxfem/bin/python3 product/app.py
# 浏览器打开 http://127.0.0.1:8050
```

首次访问后台会自动预热(默认参数的 JAX 求解 + 3D 渲染,约 3s),
预热完成后点击「运行求解」几乎瞬间出结果。

## 功能清单

| 功能 | 说明 |
|---|---|
| 参数面板 | 几何(跨度/单元数/矩形截面)、材料(E/ν/ρ)、荷载(跨中集中力)、求解器(容差/迭代上限)全部可调,中文界面,实时校验 |
| 双后端求解 | JAX GPU 并行(vmap)/ NumPy 串行两种 EBE-PCG 后端,一键切换 |
| 结果卡片 | 跨中挠度(FEM/理论/直接解三方对照)、求解精度、迭代次数、耗时、模型规模 |
| 2D 曲线 | 沿梁挠度 UY 与转角 ROTZ 曲线,可叠加理论解(δ=PL³/(48EI) 分布) |
| 3D 变形云图 | pyvista 渲染的实体梁(截面扫掠),可旋转/缩放;显示场可在 UY / USUM / von Mises 应力间切换,变形放大系数可调 |
| 性能对比 | 三种求解方式(直接解 / NumPy EBE-PCG / JAX EBE-PCG)耗时柱状图 + 加速比;叠加历史 benchmark 曲线与本次运行标记点 |
| 智能缓存 | 相同参数重复运行零求解;切换显示场/放大系数只重渲染不重求解 |

## 界面说明

左侧面板分组:

- **几何参数**:跨度 L(m)、单元数 n_elem(10~2000,偶数)、截面宽/高(mm)
- **材料参数**:E(GPa)、ν、ρ(kg/m³)
- **荷载**:跨中集中力 P(kN),负值 = 竖直向下
- **求解器**:后端(JAX GPU / NumPy 串行)、收敛容差 tol、最大迭代数(留空 = 自适应)
- **显示设置**:云图显示场、变形放大系数、理论解叠加开关(切换不重新求解)

## 3~5 分钟求职演示脚本

1. **开屏**(30s):介绍平台与论文背景 —— 大跨度桥梁抖振时域分析需要大规模并行
   有限元,本项目用 JAX vmap 实现 EBE-PCG 单元级并行,先以简支梁验证精度与加速。
2. **精度验证**(1min):默认参数点击「运行求解」,指出跨中挠度三方一致
   (2.2046 mm),FEM 与理论曲线完全重合 —— 算法正确性。
3. **GPU 加速**(1min):勾选/对比「耗时对比图」三根柱(直接解 / NumPy EBE / JAX EBE),
   口播加速比;把 n_elem 拉到 1000~2000 再跑一次,指出 NumPy 串行已需数分钟
   (面板显示拟合估计值),而 JAX 仅数秒 —— GPU 并行的规模优势。
4. **3D 云图**(1min):切换显示场 UY → USUM → von Mises 应力,旋转/放大模型,
   展示实体梁变形与应力分布;拖动放大系数演示变形 2000×。
5. **收尾**(30s):架构一句话 —— JAX 函数式 + vmap 单元并行 + 全 JIT 编译,
   为后续大跨度桥梁整桥模型(DOF 10⁵+)铺路;现有 benchmark 曲线显示 JAX
   在 n_elem=2000 时约 6.4s,而 NumPy 需 36 分钟。

> 注意:面试现场网络可能受限,本平台**完全本地运行、零外部依赖**(无 CDN 资源),
> 离线可用。3D 云图为 pyvista 导出的自包含 HTML。

## 项目结构

```
JAXFEM/                           # 项目根:核心求解代码 + 科研文件(未改动)
├── beam_element.py               # BEAM4 单元刚度矩阵(核心)
├── post.py                       # 统一求解入口 + Post 可视化类(核心)
├── ansys/  jax_ebe/  numpy_ebe/  # 模型生成与求解器(核心)
├── benchmark.py  benchmark_data/ # 性能基准(Web 对比图数据源)
└── product/                      # ★ 产品目录(本 README 所在)
    ├── app.py                    # Web 应用入口
    ├── assets/style.css          # 界面样式
    ├── webapp/                   # Web 应用包
    │   ├── config.py             # 参数边界/配色/默认值
    │   ├── layout.py             # 页面布局
    │   ├── callbacks.py          # 3 个回调(求解/显示切换/滑条回显)
    │   └── services/
    │       ├── solver_service.py # 参数校验 + 求解管线 + 结果缓存
    │       ├── export3d.py       # 无头 3D 云图导出(pyvista → HTML 字符串)
    │       ├── benchmark_data.py # benchmark 数据 + NumPy 耗时拟合
    │       └── charts.py         # plotly 图表公共样式
    ├── tests/
    │   ├── smoke_test.py         # 无头冒烟测试(不依赖服务)
    │   └── e2e_test.py           # 端到端回调测试(需服务已启动)
    └── requirements.txt
```

## 测试

```bash
# 无头冒烟测试(模拟无显示器环境,验证求解精度与 3D 导出)
cd "/home/zjr/桌面/求职/自研产品/JAXFEM"
env -u DISPLAY /home/zjr/anaconda3/envs/jaxfem/bin/python3 product/tests/smoke_test.py

# 端到端测试(先启动服务,再另开终端执行)
env -u DISPLAY /home/zjr/anaconda3/envs/jaxfem/bin/python3 product/tests/e2e_test.py
```

## 已知限制

| 限制 | 原因 |
|---|---|
| n_elem 上限 2000 | 直接解(基准)的 K_global 在 n=2000 时约 1.15GB 内存 |
| NumPy 后端 n_elem ≤ 500 | NumPy 串行 EBE-PCG 在 n=2000 需约 36 分钟 |
| 大模型时 NumPy 对比为拟合估计值 | n_elem > 400 时对比图 NumPy 柱为 benchmark 数据 log-log 外推(斜纹表示) |
| 首次运行(或改单元数后)有 1-2s JIT 编译 | JAX 按形状缓存编译结果,同规模再次运行瞬间完成 |
| 3D 场景内文字为英文 | VTK 默认字体无中文字形 |

## 故障排查

| 现象 | 处理 |
|---|---|
| 顶栏徽章显示「CPU 模式」 | GPU 未识别;检查 CUDA 驱动,应用仍可运行(JAX 自动回退 CPU) |
| 求解失败弹红框 | 看错误信息(参数校验为中文);大模型 NumPy 后端会被拦截 |
| 端口被占用 | `python product/app.py --port 8051` |
| 3D 云图空白 | 服务端日志查看 VTK 报错;重启应用(后台预热会自动重试) |
| 想跳过预热 | `python product/app.py --no-prewarm` |
