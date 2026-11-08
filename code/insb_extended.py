"""Paired same-voltage target refit, common cohorts and independent source audit."""
import sys
sys.dont_write_bytecode=True
import json, hashlib
from zipfile import ZipFile
import numpy as np
import pandas as pd
from insb_io import ROOT,load_devices,valid
from insb_analyze import write_csv,summarize
from insb_stats import fit_line,score,t975

def main():
    ds,audit=load_devices();lookup={d['device']:d for d in ds}
    fits=pd.read_csv(ROOT/'insb_reference_fits.csv');scores=pd.read_csv(ROOT/'insb_device_scores.csv')
    points=pd.read_csv(ROOT/'insb_prediction_points.csv')
    extended=[];refit_points=[];exclusions=[]
    labels=['same_2MHz','forward_100000Hz','forward_5000Hz','reverse_2e+06Hz']
    for row in fits[fits.window=='low_04_08'].itertuples():
        d=lookup[row.device];v=d['v'];ix=np.flatnonzero((v>=row.Vmin-1e-8)&(v<=row.Vmax+1e-8));train=ix[np.arange(len(ix))%3!=2];test=ix[np.arange(len(ix))%3==2]
        reference=fit_line(v[train],(row.Cref_F/d['c'][train,-1])**2)
        for label in labels:
            if label=='reverse_2e+06Hz': cv,cc=d['reverse'][2e6]
            else:
                freq={'same_2MHz':2e6,'forward_100000Hz':1e5,'forward_5000Hz':5e3}[label]
                j=int(np.flatnonzero(d['freq']==freq)[0]);cv,cc=v,d['c'][:,j]
            mapping={round(float(x),8):i for i,x in enumerate(cv)}
            tr=np.array([mapping.get(round(float(v[i]),8),-1) for i in train]);te=np.array([mapping.get(round(float(v[i]),8),-1) for i in test])
            tr=tr[tr>=0];te=te[te>=0];tr=tr[valid(cc[tr])];te=te[valid(cc[te])]
            if len(tr)<8 or len(te)<5:
                exclusions.append(dict(device=row.device,condition=label,n_train=len(tr),n_test=len(te),reason='insufficient valid paired samples'));continue
            assert not set(tr)&set(te)
            target=fit_line(cv[tr],(row.Cref_F/cc[tr])**2)
            for model,fit in [('frozen_reference',reference),('target_refit_same_voltage',target)]:
                metrics=score(fit,cv[te],cc[te],row.Cref_F)
                extended.append(dict(device=row.device,group=row.group,window='low_04_08',condition=label,model=model,n_train=len(tr),**metrics))
            for member,indices in [('train',tr),('test',te)]:
                for i in indices: refit_points.append(dict(device=row.device,condition=label,membership=member,V_G_V=cv[i],C_F=cc[i],Cref_F=row.Cref_F))
    write_csv('target_refit_scores.csv',extended);write_csv('target_refit_membership.csv',refit_points);write_csv('target_refit_exclusions.csv',exclusions or [dict(reason='none')])
    write_csv('target_refit_summary.csv',summarize(extended,['condition','model'],['mape_valid_C','y_relative_rmse_all','y_rmse_all','y_mae_all','inclusion','nrmse_Cref','invalid_prediction_fraction']))
    # Identical paired low-loss cohort: both reference-window and reverse-test loss <= .1.
    reverse=scores[(scores.window=='low_04_08')&(scores.condition=='reverse_2e+06Hz')]
    low=set(reverse[(reverse.reference_loss_median>=0)&(reverse.reference_loss_median<=.1)&(reverse.target_loss_median>=0)&(reverse.target_loss_median<=.1)].device)
    lowrows=scores[(scores.window=='low_04_08')&scores.device.isin(low)&scores.condition.isin(['same_2MHz','reverse_2e+06Hz'])]
    write_csv('paired_low_loss_scores.csv',lowrows.to_dict('records'))
    write_csv('paired_low_loss_summary.csv',summarize(lowrows.to_dict('records'),['condition'],['mape','inclusion','nrmse_Cref']))
    windows=['low_04_08','low_05_10','low_05_15','low_10_30','primary']
    common=set.intersection(*[set(scores[(scores.window==w)&(scores.condition==c)].device) for w in windows for c in labels])
    commonrows=scores[scores.device.isin(common)&scores.window.isin(windows)&scores.condition.isin(labels)]
    write_csv('common_cohort_summary.csv',summarize(commonrows.to_dict('records'),['window','condition'],['mape','inclusion','nrmse_Cref']))
    strong=set(fits[(fits.window=='low_04_08')&(fits.r2>.99)&fits.expected_negative_slope].device)
    write_csv('high_linearity_summary.csv',summarize(scores[(scores.window=='low_04_08')&scores.device.isin(strong)].to_dict('records'),['condition'],['mape','inclusion']))
    # Independent numerical checks and source mappings.
    with ZipFile(ROOT/'insb_source_data.zip') as z: assert z.testzip() is None
    assert len(ds)==159 and sum(x['status']=='matched_hysteresis' for x in audit)==159
    assert all(set(d['reverse'])=={5000.,2000000.} for d in ds)
    assert not points.duplicated(['device','window','condition','V']).any()
    assert np.allclose(points.y,(points.Cref_F/points.C_F)**2,rtol=2e-13)
    assert np.array_equal(points.included,((points.y>=points.lower_y)&(points.y<=points.upper_y)).astype(int))
    for row in points.itertuples():
        d=lookup[row.device]
        if row.condition.startswith('reverse_'):v,c=d['reverse'][row.frequency_Hz]
        else:
            j=np.flatnonzero(d['freq']==row.frequency_Hz);assert len(j)==1;v,c=d['v'],d['c'][:,j[0]]
        i=np.flatnonzero(np.isclose(v,row.V,atol=1e-8,rtol=0));assert len(i)==1
        assert np.isclose(c[i[0]],row.C_F,rtol=2e-13,atol=0)
    maxdiff=0.
    for row in fits.itertuples():
        d=lookup[row.device];v=d['v'];ix=np.flatnonzero((v>=row.Vmin-1e-8)&(v<=row.Vmax+1e-8));assert len(ix)==row.n_window
        tr=ix[np.arange(len(ix))%3!=2];xx=v[tr];yy=(row.Cref_F/d['c'][tr,-1])**2
        b=np.sum((xx-xx.mean())*(yy-yy.mean()))/np.sum((xx-xx.mean())**2);a=yy.mean()-b*xx.mean()
        maxdiff=max(maxdiff,abs(a-row.a)/max(1,abs(row.a)),abs(b-row.b)/max(1,abs(row.b)))
    assert maxdiff<1e-9
    assert max(abs(t975(k)-v) for k,v in {3:3.182446305284263,10:2.2281388519649385,30:2.0422724563012373,100:1.9839715184496334}.items())<1e-8
    # Full CSV round-trip: every C/G value in all 159 primary devices, including invalids.
    raw_seen={};raw_total=0
    from collections import Counter
    for chunk in pd.read_csv(ROOT/'insb_raw_measurements.csv.gz',chunksize=25000,float_precision='round_trip'):
        for key,sub in chunk.groupby(['device','direction','source_value_column_1based'],sort=False):
            dev,direction,column=key;d=lookup[dev];j=int(column)-2;freq=d['freq'][j]
            start=raw_seen.get(key,0);acq=sub.acquisition_index_0based.to_numpy(int)
            assert np.array_equal(acq,np.arange(start,start+len(sub)))
            if direction=='forward':
                vv=d['v'][acq];cc=d['c'][acq,j];gg=d['g'][acq,j]
                amb=int(Counter(d['freq'])[freq]>1)
            else:
                rv,rc=d['reverse'][freq];ri=len(rv)-1-acq
                vv=rv[ri];cc=rc[ri];gg=d['reverse_g'][freq][1][ri];amb=0
            assert np.array_equal(sub.V_G_V.to_numpy(),vv,equal_nan=True)
            assert np.array_equal(sub.C_F.to_numpy(),cc,equal_nan=True)
            assert np.array_equal(sub.G_S.to_numpy(),gg,equal_nan=True)
            assert np.all(sub.frequency_label_Hz==freq) and np.all(sub.ambiguous_frequency_label==amb)
            assert np.array_equal(sub.C_valid.to_numpy(),valid(cc).astype(int))
            assert np.array_equal(sub.G_finite_nonnegative.to_numpy(),(np.isfinite(gg)&(gg>=0)&(gg<1e30)).astype(int))
            raw_seen[key]=start+len(sub);raw_total+=len(sub)
    expected={}
    for d in ds:
        for j,f in enumerate(d['freq']):expected[(d['device'],'forward',j+2)]=len(d['v'])
        for f,(rv,rc) in d['reverse'].items():
            j=int(np.flatnonzero(d['freq']==f)[0]);expected[(d['device'],'reverse',j+2)]=len(rv)
    assert raw_seen==expected
    # Recount target-refit error and prediction inclusion independently with centered OLS.
    member=pd.read_csv(ROOT/'insb_target_refit_membership.csv')
    target_rows=[x for x in extended if x['model']=='target_refit_same_voltage']
    for r in target_rows:
        block=member[(member.device==r['device'])&(member.condition==r['condition'])]
        tr=block[block.membership=='train'];te=block[block.membership=='test']
        assert not set(tr.V_G_V)&set(te.V_G_V)
        x=tr.V_G_V.to_numpy();y=(tr.Cref_F/tr.C_F).to_numpy()**2
        b=np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2);a=y.mean()-b*x.mean()
        xt=te.V_G_V.to_numpy();yt=(te.Cref_F/te.C_F).to_numpy()**2;mu=a+b*xt
        s2=np.sum((y-a-b*x)**2)/(len(x)-2)
        h=1/len(x)+(xt-x.mean())**2/np.sum((x-x.mean())**2)
        half=t975(len(x)-2)*np.sqrt(s2*(1+h));inc=np.mean((yt>=mu-half)&(yt<=mu+half))
        pos=mu>0;err=np.mean(np.abs(te.Cref_F.to_numpy()[pos]/np.sqrt(mu[pos])/te.C_F.to_numpy()[pos]-1))
        assert abs(inc-r['inclusion'])<1e-12 and np.isclose(err,r['mape_valid_C'],rtol=1e-10,atol=1e-12)
        yrmse=float(np.sqrt(np.mean((mu-yt)**2))); yrel=yrmse/np.mean(yt)
        assert np.isclose(yrel,r['y_relative_rmse_all'],rtol=1e-10,atol=1e-12)
        assert np.isclose(yrmse,r['y_rmse_all'],rtol=1e-10,atol=1e-12)
        assert int(np.sum(~pos))==r['invalid_C_prediction_count']
    verification=dict(raw_csv_roundtrip_rows=raw_total,raw_csv_all_C_and_G_values='exact binary float round-trip pass',target_refit_independent_recount=len(target_rows),target_refit_all_point_y_error_and_invalid_counts='independently verified',source_zip_crc='pass',hysteresis_exact_prefix='159/159',saved_observations_verified=len(points),
                      centered_regression_max_relative_difference=maxdiff,student_t_quantiles='pass',
                      common_cohort_devices=len(common),paired_low_loss_devices=len(low),high_linearity_devices=len(strong),
                      target_refit='same reference-defined voltage window, same train/test membership, no test labels used to refit',
                      exploratory_note='Whole reference curve defines window; target frequency and sweep change observable response; not iid external validation.')
    (ROOT/'insb_verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))
    print(pd.read_csv(ROOT/'insb_target_refit_summary.csv')[['condition','model','n_devices','mape_valid_C_median','y_relative_rmse_all_median','invalid_C_prediction_points','inclusion_median']].to_string(index=False))
    print(pd.read_csv(ROOT/'insb_paired_low_loss_summary.csv').to_string(index=False))

if __name__=='__main__':main()
