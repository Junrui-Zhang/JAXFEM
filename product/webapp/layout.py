"""
layout.py — 页面布局(纯 dash html 组件 + assets/style.css)
=============================================================
结构:顶栏(GPU 徽章)/ 左侧参数面板 / 主内容区(卡片+曲线+3D+对比图)。
组件 id 与 callbacks.py 严格对应。
"""

import plotly.graph_objects as go
from dash import dcc, html

from webapp import config
from webapp.services.benchmark_data import build_benchmark_figure
from webapp.services.export3d import placeholder_html

GRAPH_CONFIG = {'displayModeBar': False}

FIELD_OPTIONS = [
    {'label': 'UY 竖向位移 (mm)', 'value': 'uy'},
    {'label': 'USUM 总位移 (mm)', 'value': 'usum'},
    {'label': 'von Mises 应力 (MPa)', 'value': 'mises'},
]


def _empty_figure(message: str) -> go.Figure:
    """结果占位图。"""
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False,
        font=dict(color=config.CHART_MUTED, size=14))
    fig.update_layout(
        paper_bgcolor=config.CHART_SURFACE, plot_bgcolor=config.CHART_SURFACE,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ============================================================
# 顶栏
# ============================================================

def _topbar(gpu_info: dict) -> html.Header:
    ok = gpu_info.get('available', False)
    return html.Header(className='topbar', children=[
        html.Div(className='brand', children=[
            html.Span('JAXFEM', className='brand-name'),
            html.Span('简支梁 EBE-PCG GPU 并行有限元演示平台', className='brand-sub'),
        ]),
        html.Div(className=f'gpu-badge {"ok" if ok else "warn"}', children=[
            html.Span(className='gpu-dot'),
            html.Span(f"计算设备:{gpu_info.get('name', '未知')}"
                      + (' · JAX GPU 并行' if ok else ' · CPU 模式(JAX 自动回退)')),
        ]),
    ])


# ============================================================
# 侧栏:参数面板
# ============================================================

def _group(title: str, children: list) -> html.Div:
    return html.Div(className='group', children=[
        html.Div(title, className='group-title'),
        *children,
    ])


def _num_field(pid: str, label: str, value, unit: str, hint: str = None) -> html.Div:
    return html.Div(className='field', children=[
        html.Label(label, htmlFor=pid, className='field-label'),
        html.Div(className='field-row', children=[
            dcc.Input(id=pid, type='number', value=value, className='num-input'),
            html.Span(unit, className='field-unit'),
        ]),
        html.Div(hint, className='field-hint') if hint else None,
    ])


def _sidebar() -> html.Aside:
    return html.Aside(className='sidebar', children=[
        _group('几何参数', [
            _num_field('L', '跨度 L', config.DEFAULT_UI['L'], 'm'),
            html.Div(className='field', children=[
                html.Label('单元数 n_elem', htmlFor='n-elem', className='field-label'),
                dcc.Slider(
                    id='n-elem', min=config.N_ELEM_MIN, max=config.N_ELEM_MAX,
                    step=2, value=config.DEFAULT_UI['n_elem'],
                    marks={10: '10', 200: '200', 500: '500',
                           1000: '1000', 1500: '1500', 2000: '2000'},
                    updatemode='mouseup', className='slider'),
                html.Div(id='n-elem-label', className='field-hint',
                         children=f"{config.DEFAULT_UI['n_elem']} 个单元 "
                                  f"({6 * (config.DEFAULT_UI['n_elem'] + 1):,} DOF)"),
            ]),
            _num_field('b-width', '截面宽(梁宽)', config.DEFAULT_UI['b_width'], 'mm'),
            _num_field('b-height', '截面高(梁高)', config.DEFAULT_UI['b_height'], 'mm'),
        ]),
        _group('材料参数', [
            _num_field('E', '弹性模量 E', config.DEFAULT_UI['E'], 'GPa'),
            _num_field('nu', '泊松比 ν', config.DEFAULT_UI['nu'], ''),
            _num_field('rho', '密度 ρ', config.DEFAULT_UI['rho'], 'kg/m³'),
        ]),
        _group('荷载', [
            _num_field('P', '跨中集中力 P', config.DEFAULT_UI['P'], 'kN',
                       hint='负值 = 竖直向下'),
        ]),
        _group('求解器', [
            html.Div(className='field', children=[
                html.Label('后端', className='field-label'),
                dcc.RadioItems(
                    id='solver', className='radio-group',
                    inputClassName='radio-input', labelClassName='radio-label',
                    options=[
                        {'label': ' JAX GPU 并行 (vmap)', 'value': 'jax'},
                        {'label': ' NumPy 串行(基线)', 'value': 'numpy'},
                    ],
                    value=config.DEFAULT_UI['solver']),
            ]),
            _num_field('tol', '收敛容差 tol', config.DEFAULT_UI['tol'], ''),
            _num_field('max-iter', '最大迭代数', '', '',
                       hint='留空 = 自适应'),
        ]),
        html.Button('▶  运行求解', id='run-button', className='run-button'),
        _group('显示设置(不重新求解)', [
            html.Div(className='field', children=[
                html.Label('云图显示场', htmlFor='field', className='field-label'),
                dcc.Dropdown(id='field', options=FIELD_OPTIONS,
                             value=config.DEFAULT_UI['field'],
                             clearable=False, className='dropdown'),
            ]),
            html.Div(className='field', children=[
                html.Label('变形放大系数', htmlFor='scale', className='field-label'),
                dcc.Slider(id='scale', min=config.SCALE_MIN, max=config.SCALE_MAX,
                           step=config.SCALE_STEP,
                           value=config.DEFAULT_UI['scale'],
                           marks={1: '1×', 500: '500×', 1000: '1000×', 2000: '2000×'},
                           updatemode='mouseup', className='slider'),
            ]),
            html.Div(className='field', children=[
                dcc.Checklist(id='theory-toggle',
                              options=[{'label': ' 曲线叠加理论解(虚线)', 'value': 'theory'}],
                              value=['theory'] if config.DEFAULT_UI['theory'] else [],
                              className='checklist', inputClassName='check-input',
                              labelClassName='check-label'),
            ]),
        ]),
        html.Div(className='sidebar-note', children=[
            html.Div('提示:'),
            html.Div('· 首次运行含 JAX JIT 编译(约 1-2s),之后同规模瞬间完成'),
            html.Div('· NumPy 后端仅支持 n_elem ≤ 500(串行太慢)'),
            html.Div('· n_elem 增大后建议同步放大变形系数'),
        ]),
    ])


# ============================================================
# 主内容区
# ============================================================

def _card(title: str, value: str = '—', subs: list = None) -> html.Div:
    return html.Div(className='card', children=[
        html.Div(title, className='card-title'),
        html.Div(value, className='card-value'),
        html.Div([html.Div(s, className='card-sub') for s in (subs or [])]),
    ])


def _content() -> html.Main:
    empty_msg = '点击左侧「运行求解」后显示'
    return html.Main(className='content', children=[
        html.Div(id='warning-box', className='warning-box hidden'),
        dcc.Loading(id='loading', type='circle', color=config.COL_FEM, children=[
            html.Div(className='results', children=[
                html.Div(id='status-badge', className='status-badge hidden'),
                html.Div(id='result-cards', className='cards-grid', children=[
                    _card('跨中挠度'),
                    _card('求解精度'),
                    _card('迭代次数'),
                    _card('求解耗时'),
                    _card('模型规模'),
                ]),
                html.Div(id='speedup-text', className='speedup-text'),
                html.Div(className='chart-grid', children=[
                    dcc.Graph(id='graph-deflection', config=GRAPH_CONFIG,
                              figure=_empty_figure(empty_msg)),
                    dcc.Graph(id='graph-rotz', config=GRAPH_CONFIG,
                              figure=_empty_figure(empty_msg)),
                ]),
                html.Div(className='panel-title', children=[
                    html.Span('三维变形云图'),
                    html.Span('(鼠标左键旋转 / 滚轮缩放 / 右键平移)', className='panel-sub'),
                ]),
                html.Iframe(id='iframe-3d', className='iframe-3d',
                            srcDoc=placeholder_html()),
                html.Div(className='chart-grid', children=[
                    dcc.Graph(id='graph-compare-bar', config=GRAPH_CONFIG,
                              figure=_empty_figure(empty_msg)),
                    dcc.Graph(id='graph-benchmark', config=GRAPH_CONFIG,
                              figure=build_benchmark_figure()),
                ]),
            ]),
        ]),
    ])


# ============================================================
# 组装
# ============================================================

def build_layout(gpu_info: dict) -> html.Div:
    return html.Div(className='app-root', children=[
        _topbar(gpu_info),
        html.Div(className='app-body', children=[
            _sidebar(),
            _content(),
        ]),
        dcc.Store(id='result-store', storage_type='memory'),
    ])
