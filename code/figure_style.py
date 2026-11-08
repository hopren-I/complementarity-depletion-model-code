"""Shared typography/export settings; never changes plotted scientific arrays."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
PROJECT = Path(__file__).resolve().parents[1]
def embedded_cjk_font():
    """Convert the needed open-license Fandol glyphs to a real TrueType subset.

    Matplotlib's Type 42 embedding of CFF OTF fonts can lose glyphs when the
    figure is imported by XeLaTeX. Keep the supplied font untouched; generate
    a renamed, quadratic-outline subset in the build cache instead.
    """
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.pens.cu2quPen import Cu2QuPen

    font = TTFont(PROJECT / "assets/fonts/FandolHei-Regular.otf", recalcTimestamp=False)
    sources = [PROJECT / 'code/plot_si.py']
    characters = set(range(32, 127)) | {ord(c) for source in sources for c in source.read_text() if ord(c) >= 128}
    selection = subset.Subsetter()
    selection.populate(unicodes=characters)
    selection.subset(font)
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in font.getGlyphOrder():
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    del font['CFF ']
    if 'VORG' in font:
        del font['VORG']
    font.sfntVersion = '\x00\x01\x00\x00'
    builder = FontBuilder(font=font)
    builder.isTTF = True
    builder.setupGlyf(glyphs)
    builder.setupMaxp()
    builder.setupPost()
    for name_id, value in [(1,'ECSFigureHei'), (2,'Regular'), (3,'ECSFigureHei-Regular'),
                           (4,'ECSFigureHei Regular'), (6,'ECSFigureHei-Regular'), (16,'ECSFigureHei'), (17,'Regular')]:
        font['name'].setName(value, name_id, 3, 1, 0x409)
        font['name'].setName(value, name_id, 1, 0, 0)
    destination = PROJECT / 'qa/mplconfig/ECSFigureHei-Regular.ttf'
    destination.parent.mkdir(parents=True, exist_ok=True)
    font.save(destination)
    font_manager.fontManager.addfont(str(destination))
    return font_manager.FontProperties(fname=str(destination)).get_name()


CJK_FONT = embedded_cjk_font()
_font = Path('/System/Library/Fonts/Helvetica.ttc')
if _font.exists():
    font_manager.fontManager.addfont(str(_font))
LATIN_FONT = 'Helvetica' if any(f.name == 'Helvetica' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'

def configure(lang):
    plt.rcParams.update({'font.family':[LATIN_FONT,CJK_FONT],
                        'font.size':9,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
                        'axes.titlesize':9,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,
                        'axes.unicode_minus':False,'svg.hashsalt':'ecs-jss-current',
                        'mathtext.fontset':'dejavusans','text.color':'black','axes.labelcolor':'black'})

def export(fig, stem, lang, width_mm, height_mm, layout=True):
    fig.set_size_inches(width_mm/25.4,height_mm/25.4,forward=True)
    family=[LATIN_FONT,CJK_FONT]
    # Preserve every label and annotation, changing typography only.
    for t in fig.findobj(match=Text):
        if not t.get_text(): continue
        t.set_fontfamily(CJK_FONT if any(ord(c)>127 for c in t.get_text()) and lang=='zh' else family)
        t.set_fontsize(min(11,max(8.5,t.get_fontsize())))
        t.set_color('black')
    for ax in fig.axes:
        ax.title.set_fontsize(9)
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
    if layout and fig.get_layout_engine() is not None:
        fig.get_layout_engine().set(w_pad=.04,h_pad=.06,wspace=.06,hspace=.10)
    fig.canvas.draw()
    # Record effective font sizes and data signatures for reproducibility.
    texts=[t.get_fontsize() for t in fig.findobj(match=Text) if t.get_text() and t.get_visible()]
    row={'width_mm':width_mm,'height_mm':height_mm,'minimum_font_pt':min(texts),'maximum_font_pt':max(texts),'raster_dpi':600}
    target=PROJECT/'qa/figure1_style.json'
    report=json.loads(target.read_text()) if target.exists() else {}
    report[f'{stem}_{lang}']=row
    target.write_text(json.dumps(report,indent=2)+'\n')
    for ext in ['pdf','png','svg']:
        metadata={'CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Date':None} if ext=='svg' else None)
        fig.savefig(PROJECT/f'paper/figs/{stem}_{lang}.{ext}',dpi=600,metadata=metadata)
