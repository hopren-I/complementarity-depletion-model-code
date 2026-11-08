"""Isolated numerical validation; never reads/writes archived paper results.
Run: python run_experiments.py [--repetitions 1000] [--draws 10000]
Dependencies: numpy, matplotlib. Model derivations and scope: protocol.txt.
"""
from pathlib import Path
import argparse, csv, hashlib, json, platform
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/validation/baseline'
OUT.mkdir(parents=True, exist_ok=True)

def csvout(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def tridiagonal(rhs):
    """Thomas solve of the dimensionless P1 stiffness system (2,-1,-1)."""
    n=len(rhs)
    if not n: return np.array([])
    diag=np.full(n,2.0); r=rhs.copy()
    for i in range(1,n):
        a=-1/diag[i-1]; diag[i]+=a; r[i]-=a*r[i-1]
    x=np.empty(n); x[-1]=r[-1]/diag[-1]
    for i in range(n-2,-1,-1): x[i]=(r[i]+x[i+1])/diag[i]
    return x

def solve_bounds(b,n):
    """P1 obstacle solve by active-set boundary search, no analytic solution used.
    E = integral (u'^2/2+u); u(0)=b,u(1)=0,u>=0.
    Dual sigma'=1-lambda<=1; D=-b*sigma(0)-integral sigma^2/2.
    """
    dx=1/n
    for m in range(n,0,-1):
        rhs=np.full(m-1,-dx*dx)
        if m>1: rhs[0]+=b
        free=tridiagonal(rhs)
        if free.size==0 or np.min(free)>=-1e-14: break
    u=np.zeros(n+1); u[0]=b; u[1:m]=free
    residual=(2*u[1:-1]-u[:-2]-u[2:])/dx+dx
    kkt=max(float(np.max(np.abs(residual[u[1:-1]>1e-13]),initial=0)),
            float(np.max(-residual[u[1:-1]<=1e-13],initial=0)))
    slopes=np.diff(u)/dx
    sigma=np.r_[slopes[0]-dx/2,(slopes[:-1]+slopes[1:])/2,
                slopes[-1]+dx/2 if m==n else 0.0]
    dual_violation=max(0,float(np.max(np.diff(sigma)/dx-1)))
    ep=float(np.sum(.5*slopes**2*dx+.5*(u[:-1]+u[1:])*dx))
    em=float(-b*sigma[0]-np.sum(dx/6*(sigma[:-1]**2+sigma[:-1]*sigma[1:]+sigma[1:]**2)))
    assert np.min(u)>=-1e-13 and kkt<1e-10 and dual_violation<1e-10
    return em,ep,kkt,dual_violation,u,sigma

def exact(b):
    if b<=.5:
        return (2*np.sqrt(2)/3*b**1.5,np.sqrt(2*b),1/np.sqrt(2*b))
    return (.5*b*b+.5*b-1/24,b+.5,1.)

def experiment_bounds():
    nodes=[]; rows=[]; cache={}; profiles={}
    def energy(b,n):
        key=(float(b),n)
        if key not in cache:
            lo,hi,kkt,dv,u,sig=solve_bounds(b,n)
            e,q,c=exact(b)
            violation=max(lo-e,e-hi,0)
            assert violation<2e-12 and hi-lo>0
            nodes.append(dict(n=n,b=b,E_lower=lo,E_exact=e,E_upper=hi,gap=hi-lo,
                min_u=float(u.min()),kkt_residual=kkt,dual_derivative_violation=dv,energy_violation=violation))
            cache[key]=(lo,hi)
        return cache[key]
    def charge(b,n,d):
        lo,hi=energy(b,n); lom,him=energy(b-d,n); lop,hip=energy(b+d,n)
        return (lo-him)/d,(hip-lo)/d
    for name,b in [('partial',.2),('near_transition',.49),('full',.8),('cross_transition',.5)]:
        for mode in ['fixed_steps','joint_refinement']:
            for n in [32,64,128,256]:
                scale=32/n
                h0=.04 if name in ['partial','full','cross_transition'] else .003
                h=h0*(scale**.4 if mode=='joint_refinement' else 1)
                d=h0*.25*(scale**.8 if mode=='joint_refinement' else 1)
                qlo,qhi=charge(b,n,d)
                qml,qmu=charge(b-h,n,d); qpl,qpu=charge(b+h,n,d)
                clo,chi=(qpl-qmu)/(2*h),(qpu-qml)/(2*h)
                e,q,c=exact(b); cm=(exact(b+h)[1]-exact(b-h)[1])/(2*h)
                same_branch=(b+h<.5 or b-h>.5)
                rem=(3*np.sqrt(2)/8*(b-h)**(-2.5)*h*h/6) if b+h<.5 else 0.
                pl,pu=(clo-rem,chi+rem) if same_branch else (float('nan'),float('nan'))
                assert qlo<=q+1e-10<=qhi+2e-10 and clo<=cm+1e-10<=chi+2e-10
                if same_branch: assert pl<=c+1e-10<=pu+2e-10
                elo,ehi=energy(b,n)
                rows.append(dict(case=name,mode=mode,n=n,b=b,h=h,delta=d,energy_gap=ehi-elo,
                    Q_lower=qlo,Q_exact=q,Q_upper=qhi,mean_C_lower=clo,mean_C_exact=cm,mean_C_upper=chi,
                    point_bound_applicable=same_branch,C_lower=pl,C_exact=c,C_upper=pu,remainder=rem,
                    mean_relative_halfwidth=(chi-clo)/(2*cm),
                    point_relative_halfwidth=(pu-pl)/(2*c) if same_branch else float('nan')))
        _,_,_,_,u,sig=solve_bounds(b,32); profiles[name]=(b,u,sig)
    fields=[]
    for name,(b,_,_) in profiles.items():
        for n in [32,64,128,256]:
            _,_,_,_,u,sig=solve_bounds(b,n)
            fields.extend(dict(case=name,n=n,b=b,x=i/n,u=float(u[i]),sigma=float(sig[i])) for i in range(n+1))
    csvout('primal_dual_fields.csv',fields)
    csvout('energy_nodes.csv',nodes);csvout('bounds_convergence.csv',rows)
    trends=[]
    for name in profiles:
        for mode in ['fixed_steps','joint_refinement']:
            r=[x for x in rows if x['case']==name and x['mode']==mode]
            trends.append(dict(case=name,mode=mode,widths=[x['mean_relative_halfwidth'] for x in r],
                strictly_decreasing=bool(np.all(np.diff([x['mean_relative_halfwidth'] for x in r])<0))))
    return rows,profiles,dict(unique_energy_evaluations=len(nodes),all_bounds_pass=True,
        max_energy_violation=max(x['energy_violation'] for x in nodes),
        max_kkt_residual=max(x['kkt_residual'] for x in nodes),
        max_dual_derivative_violation=max(x['dual_derivative_violation'] for x in nodes),
        minimum_gap=min(x['gap'] for x in nodes),convergence=trends,
        scope='Deterministic floating-point implementation check; not interval-arithmetic certification. Cross-transition pointwise remainder deliberately not applied.')

def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return mid-half,mid+half

def experiment_bayes(reps,draws):
    q=1.602176634e-19;eps=10.6*8.854187817e-14;yscale=1e18
    vbi=1.1;k=4.3e12;sigma=.15
    b1=-2/(q*eps*k*yscale);beta=np.array([-vbi*b1,b1])
    v=np.linspace(-2.5,-1,16);x=np.column_stack([np.ones(16),v]);inv=np.linalg.inv(x.T@x);chol=np.linalg.cholesky(inv)
    test_v=np.array([-4.,-1.75,0.]);xt=np.column_stack([np.ones(3),test_v]);truth_y=xt@beta
    rng=np.random.default_rng(2026110302); rows=[]; invalid_beta=0;invalid_y=0
    for rep in range(reps):
        y=x@beta+rng.normal(0,sigma,len(v));bh=inv@x.T@y;res=y-x@bh;s2=res@res/14
        variance=s2*14/rng.chisquare(14,draws)
        bd=bh+(rng.standard_normal((draws,2))@chol.T)*np.sqrt(variance)[:,None]
        invalid_beta+=int(np.sum(bd[:,1]>=0))
        estimates={'Vbi':(-bd[:,0]/bd[:,1],vbi),'k':(-2/(q*eps*bd[:,1]*yscale),k)}
        for target,(vals,truth) in estimates.items():
            lo,med,hi=np.quantile(vals,[.025,.5,.975])
            rows.append(dict(rep=rep,target=target,truth=truth,estimate=med,lower=lo,upper=hi,
                covered=int(lo<=truth<=hi)))
        latent=bd@xt.T;predictive=latent+np.sqrt(variance)[:,None]*rng.standard_normal((draws,3))
        new_y=truth_y+rng.normal(0,sigma,3)
        invalid_y+=int(np.sum(latent<=0)+np.sum(predictive<=0)+np.sum(new_y<=0))
        # No positivity clipping or sample rejection: the chosen high-SNR experiment
        # records domain violations and aborts if the inverse transform is undefined.
        assert np.all(latent>0) and np.all(predictive>0) and np.all(new_y>0)
        for j,tv in enumerate(test_v):
            for kind,vals,truth in [('latent_C',1/np.sqrt(latent[:,j]*yscale),1/np.sqrt(truth_y[j]*yscale)),
                                    ('new_C',1/np.sqrt(predictive[:,j]*yscale),1/np.sqrt(new_y[j]*yscale))]:
                lo,med,hi=np.quantile(vals,[.025,.5,.975])
                rows.append(dict(rep=rep,target=f'{kind}@{tv:g}V',truth=truth,estimate=med,lower=lo,upper=hi,
                    covered=int(lo<=truth<=hi)))
    assert invalid_beta==0
    summaries=[]
    for target in dict.fromkeys(r['target'] for r in rows):
        rr=[r for r in rows if r['target']==target];count=sum(r['covered'] for r in rr);low,high=wilson(count,reps)
        err=np.array([(r['estimate']-r['truth'])/r['truth'] for r in rr])
        summaries.append(dict(target=target,repetitions=reps,covered=count,coverage=count/reps,
            coverage_MC_SE=float(np.sqrt(count/reps*(1-count/reps)/reps)),wilson95_lower=low,wilson95_upper=high,
            mean_relative_error=float(err.mean()),relative_error_mean_SE=float(err.std(ddof=1)/np.sqrt(reps)),
            mean_relative_interval_width=float(np.mean([(r['upper']-r['lower'])/r['truth'] for r in rr]))))
    csvout('bayes_repetitions.csv',rows);csvout('bayes_summary.csv',summaries)
    return summaries,dict(seed=2026110302,repetitions=reps,posterior_draws_per_repeat=draws,
        truth_Vbi_V=vbi,truth_k_cm=k,sigma_y_scaled=sigma,Y_scale=yscale,train_voltages=v.tolist(),test_voltages=test_v.tolist(),
        nonphysical_slope_draws=invalid_beta,nonpositive_y_draws=invalid_y,
        scope='Correctly specified iid Gaussian noise in y=1/C^2; conditional on fixed design. No N coverage claim or area calibration simulation. Pointwise coverage, not simultaneous bands. Finite draws do not establish posterior support admissibility.')

