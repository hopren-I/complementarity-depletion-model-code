"""Render merged Figure 4 from archived GaN fits and predictive diagnostics.
No simulation or posterior fitting; all scientific inputs are read-only.
"""
from pathlib import Path
import os,csv,json,hashlib,argparse
P=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--language',choices=['en','zh','both'],default='both')
parser.add_argument('--output-dir',type=Path,default=P/'.work/figures')
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(P/'.work/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
font_manager.fontManager.addfont(str(P/'assets/fonts/FandolHei-Regular.otf'))
E=P/'data/validation'
read=lambda p:list(csv.DictReader(p.open()))
points=read(E/'film_points.csv')
med=read(E/'posterior_medians.csv')
post=json.loads((P/'data/results/posteriors.json').read_text())
meta=json.loads((P/'data/manifest.json').read_text())
BLUE='#25688E';RUST='#B45336';INK='#283640';GRAY='#EEF0F1'
order=['n1_500nm','n1_750nm','n2_750nm']
for language in (['zh','en'] if args.language=='both' else [args.language]):
    zh=language=='zh'
    plt.rcdefaults()
    plt.rcParams.update({'font.family':(['FandolHei','DejaVu Sans'] if zh else ['DejaVu Sans']),'font.size':8.5,
    'axes.labelsize':9,'axes.titlesize':9,'axes.unicode_minus':False,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
    'xtick.labelsize':8.5,'ytick.labelsize':8.5,'xtick.major.size':3,
    'ytick.major.size':3,'svg.fonttype':'path','mathtext.fontset':'dejavusans'})
    fig,axes=plt.subplots(2,3,figsize=(160/25.4,110/25.4),sharex='col')
    fig.subplots_adjust(left=.095,right=.985,bottom=.115,top=.865,hspace=.24,wspace=.31)
    audit=[]
    for j,series in enumerate(order):
        rows=sorted([r for r in points if r['series']==series],key=lambda r:float(r['V_app_V']))
        ar=lambda key:np.array([float(r[key]) for r in rows])
        v=ar('V_app_V');c=ar('C_observed_F');cm=ar('C_median_F')
        used=np.array([r['used_in_primary_fit']=='1' for r in rows])
        residual=100*ar('relative_obs_minus_model')
        lo=100*(ar('C_predictive_lower_F')/cm-1)
        hi=100*(ar('C_predictive_upper_F')/cm-1)
        raw=np.genfromtxt(P/meta['series'][series]['full_csv'],delimiter=',',names=True)
        sorter=np.argsort(raw['V_app_V'])
        np.testing.assert_allclose(v,raw['V_app_V'][sorter],rtol=0,atol=1e-13)
        np.testing.assert_allclose(c,raw['C_F'][sorter],rtol=1e-12)
        np.testing.assert_allclose(residual,100*(c/cm-1),atol=1e-12)
        assert len(v)==[51,51,21][j] and used.sum()==[16,16,11][j]
        xmin=min(v)-.13
        for ax in axes[:,j]:
            ax.axvspan(-2.5,-1,color=GRAY,zorder=0)
            ax.set_axisbelow(True);ax.grid(axis='y',color='#E5E9EC',lw=.55)
            ax.set_xlim(xmin,.10)
            ax.set_xticks([-4,-2,0] if j<2 else [-2,-1,0])
            ax.tick_params(width=.65)
            for spine in ['left','bottom']:ax.spines[spine].set_color('#8B969D')
        ax=axes[0,j]
        y=1e-18/c**2
        ax.scatter(v[~used],y[~used],s=13,color=RUST,edgecolors='white',lw=.3,zorder=3)
        ax.scatter(v[used],y[used],s=13,color=BLUE,edgecolors='white',lw=.3,zorder=3)
        b0,b1=post['series'][series]['fit']['OLS_beta_scaled']
        vg=np.array([v[used].min(),v[used].max()])
        ax.plot(vg,b0+b1*vg,color=INK,lw=1.2,zorder=4)
        title=series.replace('_',' ').replace('nm',' nm')
        ax.set_title(f'({chr(97+j)}) {title}',loc='left',pad=6)
        vm=float(next(r['median'] for r in med if r['series']==series and r['target']=='Vbi_V'))
        ax.text(.04,.05,rf'$V_{{\mathrm{{bi}}}}={vm:.3f}$ V'+'\n'+rf'$n={used.sum()}$',
            transform=ax.transAxes,fontsize=8.5,va='bottom',linespacing=1.3,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
        ax=axes[1,j]
        ax.fill_between(v,lo,hi,facecolor=BLUE,alpha=.18,zorder=1,lw=0)
        ax.plot(v,lo,color=BLUE,alpha=.5,lw=.65,zorder=2)
        ax.plot(v,hi,color=BLUE,alpha=.5,lw=.65,zorder=2)
        ax.axhline(0,color='#76838B',lw=.8,zorder=2)
        ax.scatter(v[~used],residual[~used],s=13,color=RUST,edgecolors='white',lw=.3,zorder=3)
        ax.scatter(v[used],residual[used],s=13,color=BLUE,edgecolors='white',lw=.3,zorder=3)
        ax.set_title(f'({chr(100+j)})',loc='left',pad=5)
        ax.set_xlabel('$V$ (V)',labelpad=4)
        ax.set_yticks([-2,-1,0,1,2] if j==1 else [-1,-.5,0,.5,1])
        audit.append(dict(series=series,n_observations=len(v),n_fit=int(used.sum()),
          n_nonfit=int((~used).sum()),full_data_agreement=True,
          residual_min=float(residual.min()),residual_max=float(residual.max()),
          fit_line=post['series'][series]['fit']['OLS_beta_scaled']))
    axes[0,0].set_ylabel(r'$1/C^2$ ($10^{18}$ F$^{-2}$)',labelpad=5)
    axes[1,0].set_ylabel('相对残差（%）' if zh else 'Relative residual (%)',labelpad=5)
    handles=[Line2D([],[],marker='o',color=BLUE,ls='',ms=4),
    Line2D([],[],marker='o',color=RUST,ls='',ms=4),
    Line2D([],[],color=INK,lw=1.2),
    Patch(facecolor=matplotlib.colors.to_rgba(BLUE,.18),edgecolor=matplotlib.colors.to_rgba(BLUE,.5),lw=.65),
    Patch(facecolor=GRAY,edgecolor='#C5CACD',lw=.65)]
    fig.legend(handles,(['拟合点','非拟合点','主窗口线性拟合','95% 预测区间','主拟合窗口'] if zh else ['Fit points','Non-fit points','Primary-window fit','95% prediction interval','Primary window']),
    loc='upper center',bbox_to_anchor=(.54,1),ncol=3,frameon=False,
    fontsize=8.5,handlelength=1.4,columnspacing=1.8,handletextpad=.55)
    fig.canvas.draw()
    # Verify visible texts remain within canvas; point counts checked above.
    from matplotlib.text import Text
    rend=fig.canvas.get_renderer();bounds=fig.bbox
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():continue
        # Invisible tick positions outside the axes can still carry a label object.
        box=text.get_window_extent(rend)
        assert box.x0>=bounds.x0-1 and box.x1<=bounds.x1+1,text.get_text()
        assert box.y0>=bounds.y0-1 and box.y1<=bounds.y1+1,text.get_text()
    out=OUT/f'fig_gan_merged_{language}'
    for ext in ['png','svg']:
        fig.savefig(out.with_suffix('.'+ext),dpi=600,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
auditdata={'width_mm':160,'height_mm':110,'minimum_font_size_pt':8.5,'dpi':600,
'sources':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [E/'film_points.csv',E/'posterior_medians.csv',P/'data/results/posteriors.json',P/'data/manifest.json']},'groups':audit,
'notes':'Top: existing OLS primary-window lines, archived posterior Vbi medians. Bottom: archived predictive endpoints and residuals; no draws or regression refitted.'}
report_path=(P/'qa/merged_gan_audit.json' if OUT==(P/'paper/figs').resolve() else OUT/'merged_gan_audit.json')
report_path.parent.mkdir(parents=True,exist_ok=True)
report_path.write_text(json.dumps(auditdata,indent=2,ensure_ascii=False))
print(json.dumps({'figure':4,'languages':args.language,'observations':123,'fit_members':43,'output':str(OUT)}))
