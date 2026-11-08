"""Regenerate Figure 1(a) with its approved original subplot geometry.

The paired original subplot is rendered to reproduce constrained-layout
spacing, then the left panel is returned for the current Figure 1 composite.
Both panels use recalculated data; no archived plot raster is used.
"""
from pathlib import Path
import csv,io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
P=Path(__file__).resolve().parents[1]

def render_panel_a(language):
    c=list(csv.DictReader((P/'data/validation/convergence.csv').open()))
    old=list(csv.DictReader((P/'data/validation/bounds_baseline.csv').open()))
    font_manager.fontManager.addfont(str(P/'assets/fonts/FandolHei-Regular.otf'))
    zh=language=='zh';plt.rcdefaults()
    plt.rcParams.update({'font.size':9,'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.titlecolor':'black','axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
    if zh:plt.rcParams['font.family']=['FandolHei','DejaVu Sans']
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.25),layout='constrained')
    for mode,label in [('fixed_steps','固定步长' if zh else 'Fixed steps'),('joint_refinement','联合缩步' if zh else 'Joint refinement')]:
        rr=[r for r in old if r['case']=='near_transition' and r['mode']==mode]
        axs[0].loglog([int(r['n']) for r in rr],[100*float(r['point_relative_halfwidth']) for r in rr],'o--',ms=3,label=label)
    rr=[r for r in c if float(r['b'])==.49]
    axs[0].loglog([int(r['n']) for r in rr],[100*float(r['relative_halfwidth']) for r in rr],'s-',ms=3,label='间隙搜索' if zh else 'Gap search')
    axs[0].set_title('(a) $B=0.49$')
    for b in [.49,.499,.501,.51]:
        rr=[r for r in c if float(r['b'])==b]
        axs[1].loglog([int(r['n']) for r in rr],[100*float(r['relative_halfwidth']) for r in rr],'o-',ms=3,label=f'$B={b}$')
    axs[1].set_title('(b) 四个位置' if zh else '(b) Four locations')
    for ax in axs:
        ax.set_xlabel('单元数' if zh else 'Elements');ax.set_ylabel('相对半宽（%）' if zh else 'Relative half-width (%)');ax.grid(alpha=.2);ax.legend(fontsize=7,labelcolor='black')
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=240);plt.close(fig);buffer.seek(0)
    return Image.open(buffer).convert('RGBA').crop((0,0,864,780))
