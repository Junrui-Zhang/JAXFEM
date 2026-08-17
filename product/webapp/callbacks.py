"""
callbacks.py — Dash 回调
=========================
回调 1(运行求解):参数校验 → SOLVE_LOCK 内求解(命中 ResultCache 则跳过)
                  → 预热导出当前显示场(入 ExportCache)→ 卡片/对比图/状态
回调 2(显示切换):只重渲染不重求解 —— 从 ResultCache 取结果,
                  ExportCache 查 (键, 场, 系数),未命中才重新导出 3D
回调 3(滑条回显):n_elem 数值 + DOF 数
"""

import traceback

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, html, no_update

from webapp import config
from webapp.services import charts
from webapp.services.benchmark_data import build_benchmark_figure
from webapp.services.export3d import (EXPORT_CACHE, FIELD_SCALARS,
                                      export_solid_html, placeholder_html)
from webapp.services.solver_service import (RESULT_CACHE, SOLVE_LOCK,
                                            parse_params, parse_scale,
                                            run_case, node_curves,
                                            theory_curves)

SOLVER_LABELS = {'jax': 'JAX GPU 并行', 'numpy': 'NumPy 串行'}


# ============================================================
# 小工具
# ============================================================

def _fmt_time(ms: float) -> str:
    if ms >= 1000:
        return f'{ms / 1e3:.2f} s'
    return f'{ms:.1f} ms'


def _warning_children(errors: list) -> list:
    return [html.Span('⚠', className='warning-icon'),
            html.Div([html.Div(e) for e in errors], className='warning-list')]


def _status(text: str, cls: str):
    return text, f'status-badge {cls}'


def _card(title: str, value, subs: list) -> html.Div:
    """subs:字符串或 (文本, 附加class) 元组。"""
    rows = []
    for s in subs:
        if isinstance(s, tuple):
            text, cls = s
            rows.append(html.Div(text, className=f'card-sub {cls}'))
        else:
            rows.append(html.Div(s, className='card-sub'))
    return html.Div(className='card', children=[
        html.Div(title, className='card-title'),
        html.Div(value, className='card-value'),
        *rows,
    ])


# ============================================================
# 结果卡片
# ============================================================

def _build_cards(result) -> list:
    m = result.model
    mid = m['_mid_node']
    uy_mid = float(result.u[mid * m['dof_per_node'] + 1])
    uy_direct_mid = float(result.u_direct[mid * m['dof_per_node'] + 1])

    conv_txt = '✅ 收敛' if result.converged else '⚠ 达到迭代上限'
    if result.err_vs_direct_pct > 1.0:
        conv_txt = '⚠ 精度不足'
    conv_cls = 'ok' if result.converged and result.err_vs_direct_pct <= 1.0 else 'warn'

    return [
        _card('跨中挠度',
              [f'{uy_mid * 1e3:.4f}', html.Span('mm', className='unit')],
              [f'理论解: {result.theory * 1e3:.4f} mm',
               f'直接解: {uy_direct_mid * 1e3:.4f} mm']),
        _card('求解精度',
              [f'{result.err_vs_theory_pct:.4f}',
               html.Span('%', className='unit')],
              [f'vs 理论解: {result.err_vs_theory_pct:.4f} %',
               f'vs 直接解: {result.err_vs_direct_pct:.6f} %']),
        _card('迭代次数', [str(result.n_iter)],
              [f'上限: {result.info["max_iter"]:,}',
               (conv_txt, conv_cls)]),
        _card('求解耗时', [_fmt_time(result.time_wall_ms)],
              [f'{SOLVER_LABELS.get(result.params.solver, "?")} · '
               f'计算 {_fmt_time(result.info["time_ms"])}',
               '壁钟含 JIT 编译(首次)']),
        _card('模型规模', [f'{m["n_dofs"]:,}'],
              [f'节点 {m["n_nodes"]:,} · 单元 {m["n_elem"]:,}',
               f'单元长度 {m["_dx"]:.4f} m']),
    ]


# ============================================================
# 图表
# ============================================================

def _curve_figures(result, show_theory: bool):
    """挠度 / 转角两张曲线图(FEM 蓝实线 + 理论橙虚线)。"""
    m = result.model
    nc = node_curves(m, result.u)

    fig_uy = go.Figure()
    fig_uy.add_trace(go.Scatter(x=nc['x'], y=nc['uy_mm'],
                                name='FEM 数值解', mode='lines+markers',
                                line=dict(color=config.COL_FEM, width=2),
                                marker=dict(color=config.COL_FEM, size=8)))
    if show_theory:
        tc = theory_curves(m)
        fig_uy.add_trace(go.Scatter(x=tc['x'], y=tc['uy_mm'],
                                    name='理论解', mode='lines',
                                    line=dict(color=config.COL_THEORY, width=2,
                                              dash='dash')))
    charts.apply_chrome(fig_uy, '梁轴向坐标 x (m)', '挠度 UY (mm)',
                        legend=show_theory, height=320)

    fig_rz = go.Figure()
    fig_rz.add_trace(go.Scatter(x=nc['x'], y=nc['rotz'],
                                name='FEM 数值解', mode='lines+markers',
                                line=dict(color=config.COL_FEM, width=2),
                                marker=dict(color=config.COL_FEM, size=8)))
    if show_theory:
        tc = theory_curves(m)
        fig_rz.add_trace(go.Scatter(x=tc['x'], y=tc['rotz'],
                                    name='理论解', mode='lines',
                                    line=dict(color=config.COL_THEORY, width=2,
                                              dash='dash')))
    charts.apply_chrome(fig_rz, '梁轴向坐标 x (m)', '转角 ROTZ (rad)',
                        legend=show_theory, height=320)
    return fig_uy, fig_rz


def _compare_figure(result) -> go.Figure:
    """三种求解方式耗时对比(对数纵轴,估算值用斜纹表示)。"""
    npc = result.numpy_compare
    jaxc = result.jax_compare
    names = ['NumPy 直接解', 'NumPy EBE-PCG', 'JAX EBE-PCG']
    colors = [config.COL_FEM, config.COL_THEORY, config.COL_JAX]
    times = [result.time_direct_ms, npc['time_ms'], jaxc['time_ms']]
    live = [True, npc['live'], jaxc['live']]

    fig = go.Figure()
    for name, color, t, is_live in zip(names, colors, times, live):
        fig.add_trace(charts.bar_trace(
            x=[name], y=[t], name=name, color=color,
            opacity=1.0 if is_live else 0.55,
            estimated=not is_live,
            text=[f'{_fmt_time(t)}{"" if is_live else "(估计)"}'],
        ))
    charts.apply_chrome(fig, None, '耗时 (ms, 对数轴)', y_log=True,
                        legend=False, height=320)
    fig.update_xaxes(tickfont=dict(color=config.CHART_INK2, size=12))
    fig.update_layout(margin=dict(l=56, r=20, t=44, b=46))
    return fig


def _benchmark_with_marker(result) -> go.Figure:
    """静态基准曲线 + 本次运行的星标点。"""
    fig = build_benchmark_figure()
    n = result.params.n_elem
    npc = result.numpy_compare
    jaxc = result.jax_compare
    for name, color, t, is_live in [
        ('本次 · 直接解', config.COL_FEM, result.time_direct_ms, True),
        ('本次 · NumPy EBE-PCG', config.COL_THEORY, npc['time_ms'], npc['live']),
        ('本次 · JAX EBE-PCG', config.COL_JAX, jaxc['time_ms'], jaxc['live']),
    ]:
        fig.add_trace(go.Scatter(
            x=[n], y=[t], mode='markers', name=name,
            marker=dict(size=13, color=color, symbol='star',
                        line=dict(width=1, color='#ffffff')),
        ))
    fig.update_layout(height=340)
    return fig


def _speedup_text(result) -> str:
    npc = result.numpy_compare
    jaxc = result.jax_compare
    ratio = npc['time_ms'] / max(jaxc['time_ms'], 1e-9)
    if npc['live']:
        if ratio >= 1.0:
            return (f'⚡ JAX GPU 并行加速 {ratio:.1f}× '
                    f'(相对 NumPy EBE-PCG 实跑,n_elem={result.params.n_elem})')
        return (f'ℹ 该规模下 JAX 计算尚未显现优势 '
                f'(JAX {_fmt_time(jaxc["time_ms"])} vs NumPy {_fmt_time(npc["time_ms"])});'
                f'GPU 加速优势随模型规模增大而显现')
    return (f'⚡ 预计 JAX GPU 加速 ≈ {ratio:.0f}× '
            f'(NumPy 耗时基于 benchmark 拟合外推,n_elem={result.params.n_elem})')


# ============================================================
# 回调注册
# ============================================================

def register_callbacks(app) -> None:
    # ---- 回调 1:运行求解 ----
    @app.callback(
        Output('result-store', 'data'),
        Output('result-cards', 'children'),
        Output('status-badge', 'children'),
        Output('status-badge', 'className'),
        Output('warning-box', 'children'),
        Output('warning-box', 'className'),
        Output('graph-compare-bar', 'figure'),
        Output('graph-benchmark', 'figure'),
        Output('speedup-text', 'children'),
        Input('run-button', 'n_clicks'),
        State('L', 'value'), State('n-elem', 'value'),
        State('b-width', 'value'), State('b-height', 'value'),
        State('E', 'value'), State('nu', 'value'), State('rho', 'value'),
        State('P', 'value'), State('solver', 'value'),
        State('tol', 'value'), State('max-iter', 'value'),
        State('field', 'value'), State('scale', 'value'),
        prevent_initial_call=True,
    )
    def on_run(n_clicks, L, n_elem, b_width, b_height, E, nu, rho,
               P, solver, tol, max_iter, field, scale):
        if not n_clicks:
            raise RuntimeError('unreachable')   # prevent_initial_call 已挡
        raw = dict(L=L, n_elem=n_elem, b_width=b_width, b_height=b_height,
                   E=E, nu=nu, rho=rho, P=P, solver=solver,
                   tol=tol, max_iter=max_iter)
        try:
            params = parse_params(raw)
        except ValueError as exc:
            errors = [e for e in str(exc).split('；') if e]
            return (no_update, no_update,
                    *_status('❌ 参数有误,请检查左侧输入', 'error'),
                    _warning_children(errors), 'warning-box',
                    no_update, no_update, no_update)

        scale_val = parse_scale({'scale': scale})
        field_val = field if field in FIELD_SCALARS else 'uy'
        key = RESULT_CACHE.key(params)

        try:
            with SOLVE_LOCK:
                result = RESULT_CACHE.get(key)
                if result is None:
                    result = run_case(params)
                    RESULT_CACHE.put(key, result)
                # 预热导出当前显示场 → 回调 2 直接命中 ExportCache
                ek = (key, field_val, scale_val)
                if EXPORT_CACHE.get(ek) is None:
                    EXPORT_CACHE.put(ek, export_solid_html(
                        result.model, result.u, field_val, scale_val))
        except Exception as exc:
            traceback.print_exc()
            return (no_update, no_update,
                    *_status(f'❌ 求解失败:{exc}', 'error'),
                    _warning_children([f'求解过程中出现异常:{exc}']),
                    'warning-box',
                    no_update, no_update, no_update)

        # 状态以「与直接解的偏差」为准:残差停滞但解已达标时仍算成功
        err = result.err_vs_direct_pct
        if err <= 1.0:
            note = '' if result.converged else '(迭代达上限,精度仍达标)'
            status, cls = _status(
                f'✅ 求解完成 · {SOLVER_LABELS.get(params.solver, "?")} · '
                f'与直接解偏差 {err:.2e} % {note}', 'success')
        else:
            status, cls = _status(
                f'⚠ 精度警示:与直接解偏差 {err:.4f} %'
                + ('' if result.converged else '(未收敛)'),
                'warning')
        return (key,
                _build_cards(result),
                status, cls,
                [], 'warning-box hidden',
                _compare_figure(result),
                _benchmark_with_marker(result),
                _speedup_text(result))

    # ---- 回调 2:显示切换(只重渲染,不重求解) ----
    @app.callback(
        Output('iframe-3d', 'srcDoc'),
        Output('graph-deflection', 'figure'),
        Output('graph-rotz', 'figure'),
        Input('field', 'value'),
        Input('scale', 'value'),
        Input('theory-toggle', 'value'),
        Input('result-store', 'data'),
        prevent_initial_call=True,
    )
    def on_display(field, scale, theory, store_key):
        if not store_key:
            return placeholder_html(), no_update, no_update
        result = RESULT_CACHE.get(store_key)
        if result is None:
            return (placeholder_html('结果缓存已失效,请重新点击「运行求解」'),
                    no_update, no_update)

        scale_val = parse_scale({'scale': scale})
        field_val = field if field in FIELD_SCALARS else 'uy'
        ek = (store_key, field_val, scale_val)
        html_str = EXPORT_CACHE.get(ek)
        if html_str is None:
            html_str = export_solid_html(result.model, result.u,
                                         field_val, scale_val)
            EXPORT_CACHE.put(ek, html_str)

        show_theory = bool(theory) and 'theory' in (theory or [])
        fig_uy, fig_rz = _curve_figures(result, show_theory)
        return html_str, fig_uy, fig_rz

    # ---- 回调 3:单元数滑条回显 ----
    @app.callback(
        Output('n-elem-label', 'children'),
        Input('n-elem', 'value'),
    )
    def on_n_elem(n):
        if n is None:
            return '10 个单元(66 DOF)'
        return f'{n} 个单元({6 * (n + 1):,} DOF)'
