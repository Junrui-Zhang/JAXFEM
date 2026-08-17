"""
smoke_test.py — 无头冒烟测试
==============================
用法:
    cd "/home/zjr/桌面/求职/自研产品/JAXFEM"
    env -u DISPLAY /home/zjr/anaconda3/envs/jaxfem/bin/python3 product/tests/smoke_test.py

任何断言失败 → exit code 非 0。
覆盖:求解精度(理论/直接解)、三场 3D 导出、参数校验、耗时拟合单调性、布局构建。
"""

import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from webapp import config
from webapp.services.benchmark_data import estimate_numpy_ms
from webapp.services.export3d import export_solid_html, placeholder_html
from webapp.services.solver_service import detect_gpu, parse_params, run_case


def test_run_case_accuracy():
    raw = dict(config.DEFAULT_UI)
    p = parse_params(raw)
    r = run_case(p)
    assert r.n_iter > 0, '迭代次数必须 > 0'
    assert r.err_vs_theory_pct < 1.0, f'vs 理论解误差 {r.err_vs_theory_pct:.4f}% ≥ 1%'
    assert r.err_vs_direct_pct < 1.0, f'vs 直接解误差 {r.err_vs_direct_pct:.4f}% ≥ 1%'
    assert r.converged, '默认参数应当收敛'
    assert r.jax_compare['live'] and r.numpy_compare['live'], 'n=100 双后端应实跑'
    print(f'  [1] 精度 OK:理论偏差 {r.err_vs_theory_pct:.4f}%,'
          f'直接解偏差 {r.err_vs_direct_pct:.2e}%,迭代 {r.n_iter}')
    return r


def test_export3d(r):
    for field in ('uy', 'usum', 'mises'):
        html_str = export_solid_html(r.model, r.u, field=field, scale=200.0)
        assert '<html' in html_str, f'{field}: 导出内容不含 <html>'
        assert len(html_str) > 100_000, f'{field}: 导出长度 {len(html_str)} 过短'
    ph = placeholder_html()
    assert '<html' in ph, '占位 HTML 不合法'
    print('  [2] 3D 导出 OK:uy / usum / mises 三场均 >100KB')


def test_validate_params():
    def expect_error(**kw):
        raw = dict(config.DEFAULT_UI)
        raw.update(kw)
        try:
            parse_params(raw)
            return None
        except ValueError as e:
            return str(e)

    err = expect_error(n_elem=101)
    assert err and '偶数' in err, '奇数 n_elem 应报错'
    err = expect_error(P=0)
    assert err and '不能为 0' in err, 'P=0 应报错'
    err = expect_error(nu=0.5)
    assert err and '泊松比' in err, 'ν=0.5 应报错'
    err = expect_error(solver='numpy', n_elem=600)
    assert err and 'NumPy' in err, 'numpy + 600 单元应被拦截'
    # dcc.Input 空值返回 None:必须等同留空(自动),不得报错
    p_none = parse_params(dict(config.DEFAULT_UI, max_iter=None))
    assert p_none.max_iter is None, 'max_iter=None 应解析为自动'
    print('  [3] 参数校验 OK:奇 n_elem / P=0 / ν=0.5 / numpy+600 均被拦截;'
          'max_iter=None 解析为自动')


def test_estimate_monotonic():
    t200 = estimate_numpy_ms(200)
    t500 = estimate_numpy_ms(500)
    t1000 = estimate_numpy_ms(1000)
    assert 0 < t200 < t500 < t1000, f'{t200} < {t500} < {t1000} 不成立'
    print(f'  [4] 耗时拟合 OK:n=200/500/1000 → '
          f'{t200:.0f}/{t500:.0f}/{t1000:.0f} ms')


def test_layout_builds():
    from webapp.layout import build_layout
    gpu = detect_gpu()
    layout = build_layout(gpu)
    assert layout is not None
    print(f'  [5] 布局构建 OK:GPU = {gpu}')


def test_large_model_converges():
    """n=600 JAX:webapp 自适应迭代上限应足够收敛(实测需 8380 步)。"""
    raw = dict(config.DEFAULT_UI)
    raw['n_elem'] = 600
    p = parse_params(raw)
    r = run_case(p)
    assert r.converged, f'n=600 应在自适应上限内收敛(实际 {r.n_iter} 步)'
    assert r.err_vs_direct_pct < 1.0, f'误差 {r.err_vs_direct_pct:.4f}% ≥ 1%'
    print(f'  [6] 大模型收敛 OK:n=600 → {r.n_iter} 步收敛,'
          f'偏差 {r.err_vs_direct_pct:.2e}%')


if __name__ == '__main__':
    print('JAXFEM 冒烟测试(无头模式)')
    r = test_run_case_accuracy()
    test_export3d(r)
    test_validate_params()
    test_estimate_monotonic()
    test_layout_builds()
    test_large_model_converges()
    print('\n全部通过 ✅')
