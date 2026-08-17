"""
e2e_test.py — 端到端回调测试(需先启动服务)
============================================
用法:
    # 终端 1(在 JAXFEM 项目根目录)
    /home/zjr/anaconda3/envs/jaxfem/bin/python3 product/app.py --port 8050
    # 终端 2(等预热完成后)
    /home/zjr/anaconda3/envs/jaxfem/bin/python3 product/tests/e2e_test.py

通过 HTTP 直接调用 /_dash-update-component 模拟浏览器交互,
覆盖:缓存命中、大模型收敛、三场 3D 导出、连点并发、NumPy 拦截、估算柱。

注意 Dash 4 请求协议:
  - 多输出回调:"output" 为 '..a.prop...b.prop..' 拼接键(见 app.callback_map),
    "outputs" 为扁平 spec 列表;单输出回调 "outputs" 为单个 spec dict
  - State 值在 "state" 字段(与 "inputs" 分开)
"""

import sys
import time

import requests

BASE = 'http://127.0.0.1:8050'
ENDPOINT = f'{BASE}/_dash-update-component'

RUN_KEY = ('..result-store.data...result-cards.children...status-badge.children'
           '...status-badge.className...warning-box.children...warning-box.className'
           '...graph-compare-bar.figure...graph-benchmark.figure...speedup-text.children..')
RUN_OUTPUTS = [{"id": "result-store", "property": "data"},
               {"id": "result-cards", "property": "children"},
               {"id": "status-badge", "property": "children"},
               {"id": "status-badge", "property": "className"},
               {"id": "warning-box", "property": "children"},
               {"id": "warning-box", "property": "className"},
               {"id": "graph-compare-bar", "property": "figure"},
               {"id": "graph-benchmark", "property": "figure"},
               {"id": "speedup-text", "property": "children"}]
DISPLAY_KEY = '..iframe-3d.srcDoc...graph-deflection.figure...graph-rotz.figure..'
DISPLAY_OUTPUTS = [{"id": "iframe-3d", "property": "srcDoc"},
                   {"id": "graph-deflection", "property": "figure"},
                   {"id": "graph-rotz", "property": "figure"}]


def states(n_elem, solver):
    return [
        {"id": "L", "property": "value", "value": 10.0},
        {"id": "n-elem", "property": "value", "value": n_elem},
        {"id": "b-width", "property": "value", "value": 200.0},
        {"id": "b-height", "property": "value", "value": 300.0},
        {"id": "E", "property": "value", "value": 210.0},
        {"id": "nu", "property": "value", "value": 0.3},
        {"id": "rho", "property": "value", "value": 7850.0},
        {"id": "P", "property": "value", "value": -10.0},
        {"id": "solver", "property": "value", "value": solver},
        {"id": "tol", "property": "value", "value": 1e-8},
        {"id": "max-iter", "property": "value", "value": None},
        {"id": "field", "property": "value", "value": "uy"},
        {"id": "scale", "property": "value", "value": 200.0},
    ]


def run(n_clicks, n_elem, solver):
    payload = {"output": RUN_KEY, "outputs": RUN_OUTPUTS,
               "inputs": [{"id": "run-button", "property": "n_clicks",
                           "value": n_clicks}],
               "state": states(n_elem, solver),
               "changedPropIds": ["run-button.n_clicks"]}
    r = requests.post(ENDPOINT, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]


def display(field, scale, theory, key):
    payload = {"output": DISPLAY_KEY, "outputs": DISPLAY_OUTPUTS, "inputs": [],
               "state": [
                   {"id": "field", "property": "value", "value": field},
                   {"id": "scale", "property": "value", "value": scale},
                   {"id": "theory-toggle", "property": "value", "value": theory},
                   {"id": "result-store", "property": "data", "value": key},
               ], "changedPropIds": ["field.value"]}
    r = requests.post(ENDPOINT, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"]


if __name__ == '__main__':
    print('JAXFEM E2E 测试(需服务已启动)')

    t0 = time.time()
    r1 = run(1, 100, 'jax')
    dt1 = time.time() - t0
    assert dt1 < 1.0, f'预热缓存应秒回,实际 {dt1:.2f}s'
    assert '✅' in str(r1['status-badge']['children'])
    print(f'  [1] 默认参数缓存命中 {dt1:.2f}s ✓')

    r2 = run(2, 600, 'jax')
    assert '✅' in str(r2['status-badge']['children'])
    print(f'  [2] n=600 jax 收敛 ✓  {r2["status-badge"]["children"][:50]}')

    key2 = r2['result-store']['data']
    for f, sc in [('uy', 200), ('usum', 500), ('mises', 2000)]:
        d = display(f, sc, ['theory'], key2)
        doc = d['iframe-3d']['srcDoc']
        assert '<html' in doc and len(doc) > 100_000
        print(f'  [3] 场={f} 放大={sc}× → {len(doc) // 1024}KB ✓')

    for i in range(5):
        r = run(10 + i, 100, 'jax')
        assert '✅' in str(r['status-badge']['children']), f'连点 {i} 失败'
    print('  [4] 连点 5 次全部成功 ✓')

    r5 = run(20, 100, 'numpy')
    assert '✅' in str(r5['status-badge']['children'])
    print(f'  [5] numpy n=100 ✓  {str(r5["speedup-text"]["children"])[:46]}')

    r6 = run(21, 600, 'numpy')
    assert str(r6['status-badge']['children']).startswith('❌')
    assert 'NumPy' in str(r6['warning-box']['children'])
    print('  [6] numpy+600 拦截 ✓')

    r7 = run(22, 1000, 'jax')
    assert '预计' in str(r7['speedup-text']['children'])
    est = [b for b in r7['graph-compare-bar']['figure']['data']
           if b['name'] == 'NumPy EBE-PCG'][0]
    assert est['marker'].get('pattern') is not None
    print(f'  [7] n=1000 jax 估算柱 ✓  {str(r7["speedup-text"]["children"])[:56]}')

    print('\nE2E 全部通过 ✅')
