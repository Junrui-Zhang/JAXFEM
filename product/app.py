"""
app.py — JAXFEM 产品化 Web 应用入口
====================================
启动:  /home/zjr/anaconda3/envs/jaxfem/bin/python3 product/app.py
       (在 JAXFEM 项目根目录下执行)
可选:  --host 127.0.0.1 --port 8050 --no-prewarm

启动后台预热线程:默认参数 JAX 求解 + 一次 3D 导出,
消除首次点击的 JIT 编译延迟(约 1-2s)。
"""

import argparse
import os
import sys
import threading
import traceback

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from dash import Dash

from webapp import config
from webapp.callbacks import register_callbacks
from webapp.layout import build_layout
from webapp.services.solver_service import detect_gpu


def create_app() -> Dash:
    """创建 Dash 应用(GPU 徽章在启动时静态注入布局)。"""
    app = Dash(
        __name__,
        title=config.APP_TITLE,
        update_title=None,                      # Loading 时不改标签页标题
        assets_folder=os.path.join(_BASE, 'assets'),
    )
    app.layout = build_layout(detect_gpu())
    register_callbacks(app)
    return app


def _prewarm() -> None:
    """后台预热:默认参数 JAX 求解 + 3D 导出,结果入缓存。"""
    try:
        from webapp.services.export3d import EXPORT_CACHE, export_solid_html
        from webapp.services.solver_service import (RESULT_CACHE,
                                                    parse_params, run_case)
        params = parse_params(dict(config.DEFAULT_UI))
        result = run_case(params)
        key = RESULT_CACHE.key(params)
        RESULT_CACHE.put(key, result)
        html_str = export_solid_html(result.model, result.u, 'uy', 200.0)
        EXPORT_CACHE.put((key, 'uy', 200.0), html_str)
        print(f'[预热] 完成:JAX 求解 {result.n_iter} 次迭代,'
              f'3D 导出 {len(html_str) // 1024} KB')
    except Exception:
        print('[预热] 失败(不影响使用):')
        traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(description=config.APP_TITLE)
    parser.add_argument('--host', default='127.0.0.1',
                        help='监听地址(默认 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8050,
                        help='端口(默认 8050)')
    parser.add_argument('--no-prewarm', action='store_true',
                        help='跳过后台预热')
    args = parser.parse_args()

    app = create_app()
    if not args.no_prewarm:
        threading.Thread(target=_prewarm, daemon=True).start()
    print(f'* JAXFEM 已启动: http://{args.host}:{args.port}  (Ctrl+C 退出)')
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
