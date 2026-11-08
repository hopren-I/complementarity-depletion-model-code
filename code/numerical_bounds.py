"""Optimize only numerical steps; preserve baseline and paper. numpy/matplotlib."""
from pathlib import Path
import csv,json,hashlib,sys,platform
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def solve(b,n):
    """Exact algebraic solution of the discrete P1 system, NOT continuum U.
    Free-region recurrence u[i+1]-2u[i]+u[i-1]=dx² and boundary values.
    Select m by discrete active-node KKT: m(m+1)>=2*b*n².
    Binary integer search; validation against frozen Thomas solver below.
    """
    dx=1/n;left,right=1,n
    while left<right:
        mid=(left+right)//2
        if mid*(mid+1)>=2*b*n*n:right=mid
        else:left=mid+1
    m=left;i=np.arange(m+1,dtype=float)
    u=np.zeros(n+1);u[:m+1]=b*(1-i/m)+.5*dx*dx*i*(i-m)
    slope=np.diff(u)/dx
    sig=np.r_[slope[0]-dx/2,(slope[:-1]+slope[1:])/2,slope[-1]+dx/2 if m==n else 0.]
    violation=max(0.,float(np.max(np.diff(sig)/dx-1)))
    # Tiny conservative derivative repair for floating-point reconstruction.
    rho=violation+8*np.finfo(float).eps*n
    sig-=rho*np.arange(n+1)*dx
    dv=max(0.,float(np.max(np.diff(sig)/dx-1)))
    r=(2*u[1:-1]-u[:-2]-u[2:])/dx+dx
    free=u[1:-1]>1e-13
    kkt=max(float(np.max(np.abs(r[free]),initial=0)),float(np.max(-r[~free],initial=0)))
    ep=float(np.sum(.5*slope*slope*dx+.5*(u[:-1]+u[1:])*dx))
    em=float(-b*sig[0]-np.sum(dx/6*(sig[:-1]**2+sig[:-1]*sig[1:]+sig[1:]**2)))
    assert u.min()>=-1e-13 and kkt<2e-10 and dv<2e-10 and ep>em
    return em,ep,kkt,dv,rho,u,sig

# Analytic solution confined to audit, never objective/step selection.
def truth(b):
    if b<=.5:return 2*np.sqrt(2)*b**1.5/3,np.sqrt(2*b),1/np.sqrt(2*b)
    return .5*b*b+.5*b-1/24,b+.5,1.

cache={};energy_rows=[]
def energy(b,n):
    key=(float(b),n)
    if key not in cache:
        lo,hi,kkt,dv,rho,_,_=solve(b,n)
        cache[key]=(lo,hi)
        energy_rows.append(dict(b=b,n=n,lower=lo,upper=hi,gap=hi-lo,kkt=kkt,dual_violation=dv,repair_slope=rho))
    return cache[key]

def bracket(b,n,h,d):
    def q(x):
        el,eu=energy(x,n);lm,um=energy(x-d,n);lp,up=energy(x+d,n)
        return (el-um)/d,(up-el)/d
    qml,qmu=q(b-h);qpl,qpu=q(b+h)
    al,au=(qpl-qmu)/(2*h),(qpu-qml)/(2*h)
    assert b-h>0 and (b+h<.5 or b-h>.5)
    R=3*np.sqrt(2)/8*(b-h)**(-2.5)*h*h/6 if b<.5 else 0.
    low,high=al-R,au+R;radius=(high-low)/2
    # Width decomposition, using midpoint energies and gap contributions.
    def components(x):
        el,eu=energy(x,n);lm,um=energy(x-d,n);lp,up=energy(x+d,n)
        gap_q=((up-lp)+2*(eu-el)+(um-lm))/(2*d)
        return gap_q
    gap_radius=(components(b-h)+components(b+h))/(4*h)
    return dict(b=b,n=n,h=h,delta=d,lower=low,upper=high,radius=radius,
        conservative_relative_radius=radius/low if low>0 else float('inf'),
        energy_gap_radius=gap_radius,secant_radius=radius-R-gap_radius,remainder=R,
        average_lower=al,average_upper=au)

def audit(row):
    b=row['b'];h=row['h'];_,_,c=truth(b);cm=(truth(b+h)[1]-truth(b-h)[1])/(2*h)
    violation=max(row['lower']-c,c-row['upper'],row['average_lower']-cm,cm-row['average_upper'],0)
    assert violation<2e-8
    return dict(**row,exact_C=c,relative_halfwidth=row['radius']/c,violation=violation)

