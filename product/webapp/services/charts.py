"""
charts.py — Plotly 图表公共样式(浅色 chrome,配色槽位见 config)
==================================================================
规则(遵循 dataviz 规范):
  - 白底/发丝网格线/次级墨色坐标轴;≥2 序列必带图例(水平,图上方)
  - 线条 2px、marker 8px;tooltip 由 plotly 原生 hover 提供
  - 不用双 y 轴;序列颜色跟随实体而非排序
"""

import plotly.graph_objects as go

from webapp import config


def _axis_common(title: str = None, log: bool = False) -> dict:
    a = dict(
        gridcolor=config.CHART_GRID,
        zerolinecolor=config.CHART_AXIS,
        linecolor=config.CHART_AXIS,
        ticks='outside',
        tickcolor=config.CHART_AXIS,
        tickfont=dict(color=config.CHART_MUTED, size=11),
        title=dict(text=title, font=dict(color=config.CHART_INK2, size=12)) if title else None,
    )
    if log:
        a['type'] = 'log'
        a['dtick'] = 1          # 对数轴每十倍程一个刻度
    return a


def apply_chrome(fig: go.Figure, x_title: str = None, y_title: str = None,
                 x_log: bool = False, y_log: bool = False,
                 legend: bool = True, height: int = None) -> go.Figure:
    """统一图表 chrome:白底、发丝网格、墨色文字、图例位置。"""
    fig.update_layout(
        paper_bgcolor=config.CHART_SURFACE,
        plot_bgcolor=config.CHART_SURFACE,
        font=dict(family='system-ui, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif',
                  color=config.CHART_INK, size=12),
        margin=dict(l=56, r=20, t=34, b=46),
        hoverlabel=dict(bgcolor=config.CHART_SURFACE,
                        font=dict(color=config.CHART_INK, size=12),
                        bordercolor=config.CHART_AXIS),
        legend=dict(orientation='h', yanchor='bottom', y=1.0,
                    xanchor='left', x=0, font=dict(color=config.CHART_INK2, size=12))
        if legend else dict(),
        showlegend=legend,
    )
    fig.update_xaxes(**_axis_common(x_title, x_log))
    fig.update_yaxes(**_axis_common(y_title, y_log))
    if height:
        fig.update_layout(height=height)
    return fig


def line_trace(x, y, name: str, color: str, dash: str = 'solid',
               marker_size: int = 8) -> go.Scatter:
    """标准线轨迹:2px 线 + 8px 标记。"""
    return go.Scatter(
        x=x, y=y, mode='lines+markers', name=name,
        line=dict(color=color, width=2, dash=dash),
        marker=dict(size=marker_size, color=color),
    )


def bar_trace(x, y, name: str, color: str, opacity: float = 1.0,
              text: list = None, estimated: bool = False) -> go.Bar:
    """标准柱轨迹;estimated=True 时半透明 + 斜纹样式表示拟合估算值。"""
    base = dict(
        x=x, y=y, name=name,
        marker=dict(color=color, opacity=opacity,
                    pattern=dict(shape='/', size=6, solidity=0.5) if estimated else None),
        text=text, textposition='outside',
        textfont=dict(color=config.CHART_INK2, size=11),
        cliponaxis=False,
    )
    return go.Bar(**base)
