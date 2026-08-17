"""
webapp — JAXFEM 产品化 Web 应用包
==================================
零侵入设计:不修改项目任何既有模块,仅通过把项目根插入 sys.path
保证导入链完整(from beam_element / post / ansys.ansys_parser / jax_ebe ...)。
"""

import os
import sys

# 目录结构:JAXFEM/product/webapp/__init__.py
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # product/
_CORE_ROOT = os.path.dirname(_PKG_ROOT)                                    # JAXFEM 根(核心求解代码)
for _p in (_PKG_ROOT, _CORE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _patch_asyncio_signals() -> None:
    """
    兼容补丁:Dash 回调/预热线程中调用 pyvista export_html 时,
    其内部的 trame/wslink 会启动临时 aiohttp 服务器,而
    add_signal_handler 在非主线程必然抛 RuntimeError
    ("set_wakeup_fd only works in main thread of the main interpreter")。

    在非主线程吞掉该错误:损失的只是该临时服务器的信号优雅退出
    (主线程路径完全不受影响,信号注册照常成功)。
    """
    try:
        import asyncio
        import asyncio.unix_events
        _orig = asyncio.unix_events._UnixSelectorEventLoop.add_signal_handler

        def _safe_add(self, sig, callback=None, *args):
            try:
                return _orig(self, sig, callback, *args)
            except RuntimeError:
                return None   # 非主线程:无法注册信号处理,忽略

        asyncio.unix_events._UnixSelectorEventLoop.add_signal_handler = _safe_add
    except Exception:   # 平台差异等,补丁失败不影响核心功能
        pass


_patch_asyncio_signals()
