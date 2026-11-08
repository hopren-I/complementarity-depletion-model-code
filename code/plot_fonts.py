"""Use the bundled Fandol font for reproducible Chinese plotting on every OS."""
import os
from pathlib import Path
from matplotlib import font_manager

def chinese_font():
    override = os.environ.get("CHINESE_FONT")
    bundled = Path(__file__).resolve().parents[1] / "assets/fonts/FandolHei-Regular.otf"
    candidates = [Path(override)] if override else [bundled]
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei", "SimHei", "FandolHei"]:
        if name in installed:
            return name
    raise RuntimeError("Chinese plotting needs a CJK font. Install Noto Sans CJK SC or set CHINESE_FONT to a font file. Existing paper figures can be used without regeneration.")
