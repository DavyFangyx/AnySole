"""Shared matplotlib CJK font setup for display scripts.

Without this, Chinese labels render as tofu boxes because matplotlib
falls back to DejaVu Sans.  Call ``setup_cjk_fonts()`` once before
creating figures; the first installed CJK-capable family is registered
ahead of the default sans stack.
"""
from __future__ import annotations

import matplotlib
from matplotlib import font_manager

# SC first when available.  Hosts that register only the JP face of the
# Noto CJK TTC still get full unified-CJK coverage; WenQuanYi Zen Hei is
# the last resort.  Droid Sans Fallback is deliberately excluded: the
# system copy ships without Latin glyphs and breaks ASCII labels.
_PREFERRED = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
)


def setup_cjk_fonts() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in _PREFERRED:
        if family in installed:
            matplotlib.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
