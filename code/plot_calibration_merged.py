"""Draw current Figure 3 from bundled Si posterior draws and area inputs.

No posterior fit or sampling is repeated. Both languages omit the former area
table and reserve separate space for axis labels and the shared legend.
All paths resolve relative to this movable project.
"""
from pathlib import Path
import argparse
import os
import csv
import json
import hashlib

P = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(P / 'qa' / 'mplconfig_current'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
font_manager.fontManager.addfont(str(P / 'assets/fonts/FandolHei-Regular.otf'))


def draw(language, output):
    zh = language == 'zh'
    plt.rcdefaults()
    plt.rcParams.update({'font.family':['FandolHei','DejaVu Sans'] if zh else ['DejaVu Sans'],'font.size':8.5,
     'axes.labelsize':8.5,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
     'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
     'mathtext.fontset':'dejavusans','svg.fonttype':'path','svg.hashsalt':'merged-calibration-preview',
     'axes.linewidth':.7,'xtick.major.size':3,'ytick.major.size':3})
    d=np.genfromtxt(P/'data/si_ms_inset_digitized.csv',delimiter=',',names=True)
    v=d['V_app_V']; y=d['invCspec2_cm4_per_F2']/1e17; fit=v<=-1.
    beta=np.load(P/'data/results/post_Si_beta.npy')
    assert len(v)==252 and sum(fit)==199 and len(beta)==200000
    vbi,lo,hi=np.quantile(-beta[:,0]/beta[:,1],[.5,.025,.975])
    def curve(x):
     return np.concatenate([np.quantile((beta[:,0,None]+beta[:,1,None]*xx)/1e17,[.025,.5,.975],axis=0) for xx in np.array_split(x,20)],axis=1)
    xf=np.linspace(v.min(),-1,200);xe=np.linspace(-1,.49,200);bf=curve(xf);be=curve(xe)
    blue='#2166AC';orange='#B45B32';gray='#59616B'
    fig=plt.figure(figsize=(160/25.4,88/25.4))
    main=fig.add_axes([.095,.255,.370,.585]); area=fig.add_axes([.585,.255,.39,.585]); inset=fig.add_axes([.282,.65,.168,.175])
    fig.text(.095,.91,('(a) 硅标准' if zh else '(a) Silicon standard'),fontweight='bold',fontsize=9.5)
    fig.text(.585,.91,('(b) 名义面积与厚度关系' if zh else '(b) Area and thickness'),fontweight='bold',fontsize=9.5)
    for ax in (main,inset):
     ax.axhline(0,color='#B0B7BD',lw=.65)
     for xx,bb,ls in ((xf,bf,'-'),(xe,be,'--')):
      ax.fill_between(xx,bb[0],bb[2],color=blue,alpha=.2,lw=0)
      ax.plot(xx,bb[1],color=blue,lw=1,ls=ls,zorder=4)
     ax.scatter(v[fit],y[fit],s=5 if ax==main else 8,color=gray,edgecolors='none',zorder=3)
     ax.scatter(v[~fit],y[~fit],s=7 if ax==main else 9,facecolor='white',edgecolor=orange,lw=.6,zorder=5)
    main.set(xlim=(-5.2,.58),ylim=(-.025,.93),xlabel=r'$V$ (V)',ylabel=r'$1/C_{\rm spec}^2$ ($10^{17}$ cm$^4$ F$^{-2}$)')
    main.set_xticks([-5,-4,-3,-2,-1,0]);main.set_yticks([0,.2,.4,.6,.8])
    main.axvline(-1,ymax=.50,color='#9099A1',ls=':',lw=.7)
    main.add_patch(Rectangle((-.45,-.006),.94,.146,fill=False,edgecolor='#777',linestyle=':',lw=.7))
    main.text(.04,.34,('主窗口：199/252 点' if zh else 'Fit: 199/252 points'),transform=main.transAxes,va='top',fontsize=8.5)
    inset.set(xlim=(-.45,.49),ylim=(-.006,.14))
    inset.set_xticks([-.4,0,.4]);inset.set_yticks([0,.1]);inset.tick_params(labelsize=8,pad=2)
    inset.set_facecolor('white')
    inset.errorbar(vbi,0,xerr=[[vbi-lo],[hi-vbi]],fmt='v',color='#333',ms=3.5,capsize=2,lw=.7,zorder=8)
    inset.set_title(('近零区域' if zh else 'Near zero'),fontsize=8.5,pad=4)
    rows=list(csv.DictReader((P/'data/results/table2_summary.csv').open()))
    t=np.array([float(r['thickness_nm']) for r in rows]);c=np.array([float(r['median_capacitance_F']) for r in rows]);a=np.array([float(r['area_cm2']) for r in rows]);eps=10.6*8.854187817e-14
    np.testing.assert_allclose(a,c*t*1e-7/eps,rtol=1e-13)
    mean=a.mean();sd=a.std(ddof=1);au=a[:2].mean();x=np.linspace(0,10.7,200);scale=eps*1e16
    area.fill_between(x,scale*(mean-sd)*x,scale*(mean+sd)*x,color='#dce2e6',alpha=.8,zorder=1)
    area.plot(x,scale*mean*x,color='#33424e',lw=1.2)
    area.plot(x,scale*au*x,color='#777',lw=1,ls='--')
    colors=['#1F6A9B','#267B71','#C47727','#805C9C'];marks=['o','s','^','D']
    for i,(xx,yy) in enumerate(zip(1000/t,c*1e12)):
     area.scatter(xx,yy,s=42 if i==2 else 22,marker=marks[i],facecolor='none' if i==2 else colors[i],edgecolor=colors[i] if i==2 else 'white',lw=.8,zorder=5)
    pred=eps*au/(100e-7)*1e12
    area.plot(10,pred,'+',color='#555',ms=7,mew=1.2)
    area.annotate(('参考：433 pF' if zh else 'Reference: 433 pF'),(10,pred),xytext=(10.75,520),fontsize=8.5 if zh else 8.0,arrowprops=dict(arrowstyle='-',lw=.65,color='#777'),ha='right')
    area.set(xlim=(-.3,11),ylim=(-15,565),xlabel=r'$1/t$ ($\mu$m$^{-1}$)',ylabel=('电容中位数 (pF)' if zh else 'Median capacitance (pF)'))
    area.set_xticks([0,2,4,6,8,10]);area.set_yticks([0,200,400]);area.grid(axis='y',lw=.5,color='#e9e9e9');area.set_axisbelow(True)
    # Keep the area unit explicit after removing the table that carried its unit header.
    area.text(.04,.96,('面积 ' if zh else 'Area ')+r'($10^{-3}$ cm$^2$)'+'\n'+r'$\bar A=4.35\pm0.33$'+'\n'+r'$A_u=4.617$',transform=area.transAxes,va='top',fontsize=8.5)
    handles=[Line2D([],[],marker='o',ls='',ms=3,color=gray),Line2D([],[],marker='o',ls='',ms=3,mfc='white',mec=orange),Line2D([],[],color=blue,lw=1),Patch(fc=blue,alpha=.2)]
    fig.legend(handles,(['拟合点','区外点','中位线 / 虚线外推','95% 条件可信带'] if zh else ['Fit points','Outside window','Median / extrapolation','95% credible band']),loc='lower center',bbox_to_anchor=(.52,.014),ncol=4,fontsize=8.5,frameon=False,columnspacing=1.3,handlelength=1.5,handletextpad=.4)
    # Area legend is local to avoid implying that the Si posterior band is an area interval.
    area.text(.04,.72,('实线：四片均值\n虚线：本征膜均值' if zh else 'Solid: four-film mean\nDashed: intrinsic mean'),transform=area.transAxes,fontsize=8.0,va='top')
    
    if not zh:
        assert not any('\u4e00' <= ch <= '\u9fff' for artist in fig.findobj(matplotlib.text.Text) for ch in artist.get_text())
    for ext in ('png','svg'):
     fig.savefig(output/f'fig_calibration_merged_{language}.{ext}',dpi=600,facecolor='white',metadata={'Date':None} if ext=='svg' else None)
    plt.close(fig)
    report={'dimensions_mm':[160,88],'Si_counts':{'all':len(v),'fit':int(fit.sum()),'outside':int((~fit).sum()),'posterior_draws':len(beta)},'Si_Vbi_median_interval':[vbi,lo,hi],'areas_cm2':a.tolist(),'source_sha256':{str(path.relative_to(P)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [P/'data/si_ms_inset_digitized.csv',P/'data/results/post_Si_beta.npy',P/'data/results/table2_summary.csv']},'area_prediction_pF':pred}

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=('en', 'zh', 'both'), default='both')
    parser.add_argument('--output-dir', type=Path, default=P / '.work/figures')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    languages = ('zh', 'en') if args.language == 'both' else (args.language,)
    reports = {language: draw(language, output) for language in languages}
    report_path = ((P / 'qa' / 'merged_calibration.json')
                   if output == (P / 'paper/figs').resolve()
                   else output / 'merged_calibration.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({'generator': 'code/plot_calibration_merged.py',
        'languages': reports}, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output_dir': str(output), 'languages': list(languages),
                      'points': 252, 'fit_points': 199}))


if __name__ == '__main__':
    main()
