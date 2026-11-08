"""Read-only use of frozen inputs; all new results live beside this script."""
from pathlib import Path
import csv,json,sys,time,hashlib,platform
import numpy as np
P=Path(__file__).resolve().parents[1];D=P/'data';OUT=D/'validation'
OUT.mkdir(parents=True,exist_ok=True)
def read(p):return list(csv.DictReader(p.open()))
def write(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def qsum(a):
 lo,med,hi=np.quantile(a,[.025,.5,.975]);batches=np.array([np.quantile(x,[.025,.5,.975]) for x in np.array_split(a,20)])
 return dict(median=float(med),lower=float(lo),upper=float(hi),median_mcse=float(np.std(batches[:,1],ddof=1)/np.sqrt(20)),endpoint_mcse_max=float(np.max(np.std(batches[:,[0,2]],axis=0,ddof=1)/np.sqrt(20))))
def numerical():
 import numerical_bounds as op
 selected=[];candidates=[]
 for b in [.49,.499,.501,.51]:
  for n in [64,128,256,512,1024,2048,4096]:
   st=time.perf_counter();lo,hi=op.energy(b,n);s=min(.002,.2*abs(b-.5));mid=lambda a:sum(op.energy(a,n))/2
   ce=max((mid(b+s)-2*mid(b)+mid(b-s))/s**2,.1);d0=np.sqrt((hi-lo)/ce);hc=.9*min(b,abs(b-.5));trial=[]
   for hf in [.25,.4,.6,.8,1.]:
    for df in [.25,.5,.75,1,1.5,2,3,4]:
     rr=op.bracket(b,n,hc*hf,d0*df);trial.append(rr);candidates.append(dict(**rr,h_factor=hf,delta_factor=df,C_est=ce))
   best=min(trial,key=lambda r:r['radius']);elapsed=time.perf_counter()-st
   audited=op.audit(best)
   selected.append(dict(**audited,search_seconds=elapsed,h_at_cap=np.isclose(best['h'],hc)))
 for rr in candidates:op.audit(rr)
 for rr in op.energy_rows:
  exact=op.truth(rr['b'])[0];assert rr['lower']-2e-10<=exact<=rr['upper']+2e-10
 write('convergence.csv',selected);write('all_candidates.csv',candidates);write('energy_nodes.csv',op.energy_rows)
 # Save selected fields for independent re-integration.
 fields=[]
 for r in selected:
  *_,u,sig=op.solve(r['b'],r['n'])
  fields.extend(dict(b=r['b'],n=r['n'],x=i/r['n'],u=float(u[i]),sigma=float(sig[i])) for i in range(r['n']+1))
 write('selected_fields.csv',fields)
 slopes=[]
 for b in [.49,.499,.501,.51]:
  rr=[r for r in selected if r['b']==b];slope=np.polyfit(np.log([r['n'] for r in rr]),np.log([r['relative_halfwidth'] for r in rr]),1)[0]
  slopes.append(dict(b=b,loglog_slope=float(slope),min_n=64,max_n=4096,last_halfwidth_pct=100*rr[-1]['relative_halfwidth']))
 write('convergence_slopes.csv',slopes)
 return dict(candidates=len(candidates),nodes=len(op.energy_rows),max_kkt=max(x['kkt'] for x in op.energy_rows),max_dual_violation=max(x['dual_violation'] for x in op.energy_rows),all_h_at_cap=all(x['h_at_cap'] for x in selected),max_search_seconds=max(x['search_seconds'] for x in selected))
def film():
 saved=np.load(D/'results/posterior_draws.npz');win=np.load(D/'window_results/posterior_draws.npz');meta=json.loads((D/'manifest.json').read_text());rows=[]
 for series in ['n1_500nm','n1_750nm','n2_750nm']:
  for name in ['Vbi_V','k_NA2','N_cm_minus3']:rows.append(dict(series=series,target=name,**qsum(saved[series+'__'+name])))
 sb=np.load(D/'results/post_Si_beta.npy');sv=-sb[:,0]/sb[:,1];sn=-2/(1.602176634e-19*11.7*8.854187817e-14*sb[:,1])
 rows.extend([dict(series='Si',target='Vbi_V',**qsum(sv)),dict(series='Si',target='N_cm_minus3',**qsum(sn))]);write('posterior_medians.csv',rows)
 wins=[]
 for key in win.files:
  if key.endswith('__beta_scaled'):
   w,series,_=key.split('__');b=win[key];wins.append(dict(window=w,series=series,**qsum(-b[:,0]/b[:,1]),nonphysical_slope_count=int(np.sum(b[:,1]>=0))))
 write('window_medians.csv',wins)
 points=[x for x in read(D/'secant_bridge_results/interval_points.csv') if float(x['h_V'])==.5]
 assert len(points)==len({(x['series'],x['observation_id']) for x in points})==123
 diagnostics=[]
 for series in ['pooled','n1_500nm','n1_750nm','n2_750nm']:
  subset=[x for x in points if series=='pooled' or x['series']==series]
  for region in ['fit','deep_nonfit','high_nonfit']:
   rr=[x for x in subset if (x['used_in_primary_fit']=='1' if region=='fit' else x['used_in_primary_fit']=='0' and (float(x['V_app_V']) < -2.5 if region=='deep_nonfit' else float(x['V_app_V']) > -1))]
   if not rr:continue
   diagnostics.append(dict(series=series,region=region,count=len(rr),inside=sum(int(x['inside_predictive']) for x in rr),inclusion=sum(int(x['inside_predictive']) for x in rr)/len(rr),mean_signed_residual_pct=100*np.mean([float(x['relative_obs_minus_model']) for x in rr]),mean_relative_halfwidth_pct=100*np.mean([float(x['relative_halfwidth_predictive']) for x in rr]),positive_residual_count=sum(float(x['relative_obs_minus_model'])>0 for x in rr)))
 write('film_diagnostics.csv',diagnostics);write('film_points.csv',points)
 sensitivity=[]
 reading=read(D/'digitization/reading_sensitivity.csv')
 for series in meta['series']:
  vv=[x for x in wins if x['series']==series];rd=[float(x['OLS_Vbi_V']) for x in reading if x['series']==series]
  sensitivity.append(dict(series=series,window_median_min=min(x['median'] for x in vv),window_median_max=max(x['median'] for x in vv),window_range=max(x['median'] for x in vv)-min(x['median'] for x in vv),reading_OLS_min=min(rd),reading_OLS_max=max(rd),reading_range=max(rd)-min(rd)))
 write('sensitivity.csv',sensitivity)
 area=json.loads((D/'baseline_calibration.json').read_text())['area_model'];q=1.602176634e-19;eps=10.6*8.854187817e-14;au=np.mean([x['area_cm2'] for x in area['films'][:2]]);area_rows=[];branches=[]
 for f in area['films']:
  a=np.genfromtxt(P/f['source'],delimiter=',',names=True);v=a['V_app_V'];c=a['C_F'];m=v<=-1.5
  area_rows.append(dict(source=f['source'],thickness_nm=f['thickness_nm'],selected_count=int(m.sum()),selected_V_min=float(v[m].min()),selected_V_max=float(v[m].max()),selected_median_C=float(np.median(c[m])),full_C_range_over_median_pct=100*float(np.ptp(c)/np.median(c)),area_cm2=f['area_cm2'],two_intrinsic_prediction_F=eps*au/(f['thickness_nm']*1e-7),obs_minus_prediction_pct=100*(np.median(c[m])/(eps*au/(f['thickness_nm']*1e-7))-1)))
 write('area_diagnostics.csv',area_rows)
 scenarios=[]
 for series in meta['series']:
  kk=saved[series+'__k_NA2'];nn=saved[series+'__N_cm_minus3'];vb=saved[series+'__Vbi_V'];a=np.genfromtxt(P/meta['series'][series]['full_csv'],delimiter=',',names=True)
  for name,aa in [('four_film_empirical',saved['area_cm2']),('four_film_fixed',area['mean_cm2']),('two_intrinsic_fixed',au)]:
   scenarios.append(dict(series=series,scenario=name,area_reference_cm2=area['mean_cm2'] if name!='two_intrinsic_fixed' else au,**qsum(kk/aa**2)))
  thickness=500 if series=='n1_500nm' else 750;t=thickness*1e-7
  bt=q*nn*t*t/(2*eps);vt=vb-bt;maxratio=np.sqrt(2*eps*np.maximum(vb-a['V_app_V'].min(),0)/(q*nn))/t
  branches.append(dict(series=series,assumption='own fitted parameters and four-film area scenario',thickness_nm=thickness,V_min=float(a['V_app_V'].min()),V_max=float(a['V_app_V'].max()),contact_threshold_median=float(np.median(bt)),applied_transition_median=float(np.median(vt)),maximum_unclipped_W_over_t_median=float(np.median(maxratio)),fraction_full_any_measured_V=float(np.mean(maxratio>=1))))
  if series in ['n1_500nm','n2_750nm']:
   t=100e-7;bt=q*nn*t*t/(2*eps);thin=np.genfromtxt(D/('gan_n1_100nm_cv.csv' if series=='n1_500nm' else 'gan_n2_100nm_cv.csv'),delimiter=',',names=True);vmin=float(thin['V_app_V'].min());vmax=float(thin['V_app_V'].max());ratio=np.sqrt(2*eps*(vb-vmin)/(q*nn))/t
   branches.append(dict(series=series.replace('500nm','100nm').replace('750nm','100nm'),assumption='conditional transfer of thick-film N and Vbi; NOT an independent thin-film estimate',thickness_nm=100,V_min=vmin,V_max=vmax,contact_threshold_median=float(np.median(bt)),applied_transition_median=float(np.median(vb-bt)),maximum_unclipped_W_over_t_median=float(np.median(ratio)),fraction_full_any_measured_V=float(np.mean(ratio>=1))))
 write('area_scenarios.csv',scenarios);write('branch_checks.csv',branches)
 # Si window sensitivity: explicit raw thresholds; only descriptive variation.
 a=np.genfromtxt(D/'si_ms_inset_digitized.csv',delimiter=',',names=True);sirows=[]
 for upper in [float(a['V_app_V'].max()),-.5,-1,-1.5,-2,-2.5,-3,-4]:
  mask=a['V_app_V']<=upper;v=a['V_app_V'][mask];y=a['invCspec2_cm4_per_F2'][mask]/1e14;x=np.column_stack([np.ones(len(v)),v]);coef=np.linalg.lstsq(x,y,rcond=None)[0]
  sirows.append(dict(V_upper=upper,n=len(v),OLS_intercept_V=float(-coef[0]/coef[1])))
 write('si_window_sensitivity.csv',sirows)
 # Physical support probabilities from exact marginal Student-t CDF, not zero-count inference.
 import math
 def student_sf(t,df):
  # Tail via regularized beta integral, transformed to remove endpoint power.
  a=df/2;xx=df/(df+t*t)
  def integrate(n):
   nodes,weights=np.polynomial.legendre.leggauss(n);u=(nodes+1)/2
   return .5*np.exp(a*np.log(xx)-np.log(a)-math.lgamma(a)-math.lgamma(.5)+math.lgamma(a+.5))*np.sum(weights/2/np.sqrt(1-xx*u**(1/a)))
  r1,r2=integrate(128),integrate(256);assert abs(r1-r2)<1e-5*max(r2,1e-300)
  return r2
 assert abs(student_sf(10,1)-math.atan(.1)/math.pi)<1e-10
 support=[]
 post=json.loads((D/'results/posteriors.json').read_text())
 for series in meta['series']:
  fit=post['series'][series]['fit'];mu=fit['OLS_beta_scaled'][1];scale=np.sqrt(fit['t_scale_matrix_scaled'][1][1]);prob=float(student_sf(-mu/scale,fit['posterior_df']))
  support.append(dict(series=series,nonnegative_slope_posterior_probability=prob,nonnegative_saved_count=int(np.sum(saved[series+'__beta_scaled'][:,1]>=0)),draw_count=len(saved[series+'__beta_scaled'])))
 write('posterior_support.csv',support)
 return dict(intrinsic_area_cm2=float(au),area_N_scale=(area['mean_cm2']/au)**2,physical_support=support)
def stress():
 from minimal_validation import wilson
 q=1.602176634e-19;eps=10.6*8.854187817e-14;ys=1e18;vbi=1.1;k=4.3e12;sig=.15
 beta1=-2/(q*eps*k*ys);beta=np.array([-vbi*beta1,beta1]);v=np.linspace(-2.5,-1,16);x=np.column_stack([np.ones(16),v]);inv=np.linalg.inv(x.T@x);chol=np.linalg.cholesky(inv)
 tv=np.array([-4.,-1.75,0.]);xt=np.column_stack([np.ones(3),tv]);mu=x@beta;mt=xt@beta;records=[];invalid=[]
 for arm,seed in [('heteroscedastic',2026110401),('correlated',2026110402)]:
  rng=np.random.default_rng(seed);badbeta=bady=0
  for rep in range(1000):
   z=rng.standard_normal(16)
   if arm=='correlated':
    for i in range(1,16):z[i]=.6*z[i-1]+np.sqrt(1-.6**2)*z[i]
   noise=sig*z*(mu/mu.mean() if arm=='heteroscedastic' else 1)
   y=mu+noise;bh=inv@x.T@y;res=y-x@bh;s2=res@res/14;var=s2*14/rng.chisquare(14,10000);bd=bh+(rng.standard_normal((10000,2))@chol.T)*np.sqrt(var)[:,None]
   latent=bd@xt.T;pred=latent+np.sqrt(var)[:,None]*rng.standard_normal((10000,3));new=mt+sig*rng.standard_normal(3)*(mt/mu.mean() if arm=='heteroscedastic' else 1)
   badbeta+=int(np.sum(bd[:,1]>=0));bady+=int(np.sum(latent<=0)+np.sum(pred<=0)+np.sum(new<=0))
   assert (latent>0).all() and (pred>0).all() and (new>0).all() and (bd[:,1]<0).all()
   targets=[('Vbi',-bd[:,0]/bd[:,1],vbi),('k',-2/(q*eps*bd[:,1]*ys),k)]
   for j,vv in enumerate(tv):targets.extend([(f'latent_C@{vv:g}V',1/np.sqrt(latent[:,j]*ys),1/np.sqrt(mt[j]*ys)),(f'new_C@{vv:g}V',1/np.sqrt(pred[:,j]*ys),1/np.sqrt(new[j]*ys))])
   for name,vals,true in targets:
    low,med,high=np.quantile(vals,[.025,.5,.975]);records.append(dict(arm=arm,rep=rep,target=name,truth=true,estimate=med,lower=low,upper=high,covered=int(low<=true<=high)))
  invalid.append(dict(arm=arm,seed=seed,nonphysical_slope_draws=badbeta,nonpositive_response_draws=bady))
  print('Completed',arm,flush=True)
 write('stress_repetitions.csv',records);summaries=[]
 for arm in ['heteroscedastic','correlated']:
  for target in dict.fromkeys(x['target'] for x in records):
   rr=[r for r in records if r['arm']==arm and r['target']==target];count=sum(r['covered'] for r in rr);low,high=wilson(count,len(rr))
   summaries.append(dict(arm=arm,target=target,repetitions=len(rr),covered=count,coverage=count/len(rr),wilson95_lower=low,wilson95_upper=high,mean_relative_error=float(np.mean([(r['estimate']-r['truth'])/r['truth'] for r in rr])),mean_relative_interval_width=float(np.mean([(r['upper']-r['lower'])/r['truth'] for r in rr]))))
 write('stress_summary.csv',summaries);return invalid
if __name__=='__main__':
 report=dict(numerical=numerical(),film=film(),stress=stress(),python=platform.python_version(),numpy=np.__version__)
 (OUT/'analysis_report.json').write_text(json.dumps(report,indent=2,default=lambda x:bool(x))+'\n');print(json.dumps(report,indent=2,default=lambda x:bool(x)))
