"""Reproducible exploratory pilot. Writes only inside this experiment directory."""
import sys
sys.dont_write_bytecode=True
import csv
import json
import hashlib
import platform
from collections import Counter
import numpy as np
from insb_io import ROOT, GROUPS, load_devices, valid, export_raw_csv
from insb_stats import fit_line, predict, score

SPEC = dict(reference_frequency_Hz=2000000, normalization_voltage_V=[4,5],
            windows={'primary':[.3,.7], 'lower':[.2,.6], 'upper':[.4,.8], 'wide':[.2,.8],
                     'low_04_08':[.04,.08], 'low_05_10':[.05,.10], 'low_05_15':[.05,.15],
                     'low_10_20':[.1,.2], 'low_10_30':[.1,.3], 'mid_20_40':[.2,.4]},
            phase2_note='Exploratory low-capacitance grid added after observing curvature in the initial 0.3-0.7 window. All inspected windows retained. Not a preregistered confirmatory test.',
            minimum_window_points=15, held_out='window positions i mod 3 == 2',
            reference_normalization='median forward 2MHz C for 4<=V<=5; fixed across paired conditions',
            target_selection='all unique frequencies in primary forward file; reverse 5kHz and 2MHz',
            excluded_repeats='additional HR and CVT temperature-series records excluded from filename-defined primary cohort',
            invalid_C='nonfinite, nonpositive, or >=1e30 (instrument sentinel)',
            window_selection='longest contiguous reference C/Cref threshold run; no R2 filter',
            low_loss_sensitivity='median reference G/(2*pi*f*C) <= 0.1; not a primary exclusion',
            metric_definitions={'y_relative_rmse_all':'sqrt(mean((mu_y-y_obs)^2))/mean(y_obs), all valid observed test points including mu_y<=0', 'mape_valid_C':'mean(abs(Cpred/Cobs-1)) only where mu_y>0; mape legacy alias identical; undefined C predictions counted separately', 'invalid_C_prediction_count':'number of test-point regression centers mu_y<=0; no clipping'},
            inference_scope='conditional descriptive diagnostics, not physical N/Vbi extraction or calibrated coverage')


def write_csv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/('insb_'+name)).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def window_indices(v, c, cref, low, high):
    mask=valid(c)&(c/cref>=low)&(c/cref<=high)
    ix=np.flatnonzero(mask)
    if not len(ix):
        return ix
    runs=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)
    return max(runs,key=len)


def crossing(v,c,target):
    # First upward crossing on the sorted voltage axis; counts are audited.
    i=np.flatnonzero(valid(c[:-1])&valid(c[1:])&(c[:-1]<=target)&(c[1:]>target))
    if not len(i):
        return float('nan'),0
    k=i[0]
    return float(v[k]+(v[k+1]-v[k])*(target-c[k])/(c[k+1]-c[k])),len(i)


def parameter_record(fit):
    a,b=fit['coef']
    vstar=-a/b if b else float('nan')
    grad=np.array([-1/b,a/(b*b)]) if b else np.array([np.nan,np.nan])
    se=float(np.sqrt(grad @ (fit['s2']*fit['inv']) @ grad))
    return dict(r2=fit['r2'], a=a, b=b, Vstar_V=vstar, Vstar_delta_SE_V=se,
                expected_negative_slope=bool(b<0))


def summarize(rows, keys, metrics):
    out=[]
    groups={tuple(r[k] for k in keys) for r in rows}
    for g in sorted(groups):
        subset=[r for r in rows if all(r[k]==val for k,val in zip(keys,g))]
        d=dict(zip(keys,g));d['n_devices']=len({r['device'] for r in subset})
        if all('n' in r for r in subset):
            d['n_test_points']=int(sum(r['n'] for r in subset))
            d['invalid_C_prediction_points']=int(sum(r.get('invalid_C_prediction_count',round(r.get('invalid_prediction_fraction',0)*r['n'])) for r in subset))
            d['devices_with_invalid_C_predictions']=sum(r.get('invalid_C_prediction_count',round(r.get('invalid_prediction_fraction',0)*r['n']))>0 for r in subset)
            d['invalid_C_prediction_point_fraction']=d['invalid_C_prediction_points']/d['n_test_points']
        for metric in metrics:
            a=np.array([r.get(metric,np.nan) for r in subset],float);a=a[np.isfinite(a)]
            if len(a):
                d.update({metric+'_median':float(np.median(a)),metric+'_q25':float(np.quantile(a,.25)),metric+'_q75':float(np.quantile(a,.75))})
        out.append(d)
    return out


def main():
    (ROOT/'insb_analysis_spec.json').write_text(json.dumps(SPEC,indent=2))
    ds,audit=load_devices()
    raw_counts=export_raw_csv(ds)
    rows=[];fits=[];points=[];excluded=[];hysteresis=[];inventory=[];refits=[]
    for d in ds:
        dev=d['device'];v=d['v'];fs=d['freq'];j=int(np.flatnonzero(fs==2e6)[0]);c=d['c'][:,j]
        dup=[float(f) for f,n in Counter(fs).items() if n>1]
        top=(v>=4)&(v<=5)&valid(c)
        inventory.append(dict(device=dev,group=d['group'],n_voltages=len(v),n_frequency_columns=len(fs),
                              duplicate_frequency_labels=','.join(map(str,dup)),invalid_primary_C=int((~valid(c)).sum()),
                              reverse_frequencies=','.join(map(str,d['reverse'])),source=d['file']))
        if top.sum()<10:
            excluded.append(dict(device=dev,window='all',reason='insufficient valid normalization points'));continue
        cref=float(np.median(c[top]));v50,n50=crossing(v,c,.5*cref)
        for f,(rv,rc) in d['reverse'].items():
            if f not in [5e3,2e6]:
                continue
            jf=int(np.flatnonzero(fs==f)[0]);vf,nf=crossing(v,d['c'][:,jf],.5*cref);vr,nr=crossing(rv,rc,.5*cref)
            hysteresis.append(dict(device=dev,group=d['group'],frequency_Hz=f,V50_forward_V=vf,V50_reverse_V=vr,
                                   shift_V=vr-vf,abs_shift_V=abs(vr-vf),forward_crossings=nf,reverse_crossings=nr))
        for window,(low,high) in SPEC['windows'].items():
            ix=window_indices(v,c,cref,low,high)
            if len(ix)<SPEC['minimum_window_points']:
                excluded.append(dict(device=dev,window=window,reason='fewer than 15 contiguous candidate-window points',n=len(ix)));continue
            test=ix[np.arange(len(ix))%3==2];train=ix[np.arange(len(ix))%3!=2]
            fit=fit_line(v[train],(cref/c[train])**2)
            params=parameter_record(fit)
            loss=float(np.median(d['g'][ix,j]/(2*np.pi*2e6*c[ix]))) if d['g'] is not None else float('nan')
            fitrow=dict(device=dev,group=d['group'],window=window,n_window=len(ix),n_train=len(train),n_test=len(test),
                        Vmin=float(v[ix[0]]),Vmax=float(v[ix[-1]]),Cref_F=cref,reference_loss_median=loss,
                        **params)
            fits.append(fitrow)
            conditions=[('same_2MHz',2e6,v,c)]
            for k,f in enumerate(fs):
                if f!=2e6 and f not in dup:
                    conditions.append((f'forward_{f:g}Hz',float(f),v,d['c'][:,k]))
            for f,(rv,rc) in d['reverse'].items():
                if f in [5e3,2e6]:
                    conditions.append((f'reverse_{f:g}Hz',float(f),rv,rc))
            for label,f,cv,cc in conditions:
                # Match original sample voltage; no interpolation or out-of-range extrapolation.
                mapping={round(float(x),8):i for i,x in enumerate(cv)}
                mapped=np.array([mapping.get(round(float(v[t]),8),-1) for t in test])
                ok=mapped>=0
                raw=np.full(len(test),np.nan);raw[ok]=cc[mapped[ok]]
                ok=ok&valid(raw)
                if ok.sum()<5:
                    excluded.append(dict(device=dev,window=window,condition=label,reason='fewer than 5 valid paired test points',n=int(ok.sum())));continue
                vv=v[test[ok]];ct=raw[ok]
                metrics=score(fit,vv,ct,cref)
                base_c=c[test[ok]]
                metrics['paired_abs_change_Cref']=float(np.median(np.abs(ct-base_c)/cref))
                metrics['paired_relative_change']=float(np.median(np.abs(ct/base_c-1)))
                target_loss=float('nan')
                if label.startswith('reverse_') and f in d['reverse_g']:
                    gv,gg=d['reverse_g'][f]
                    if np.array_equal(gv,cv):
                        target_loss=float(np.median(gg[mapped[ok]]/(2*np.pi*f*ct)))
                elif not label.startswith('reverse_') and d['g'] is not None:
                    jf=int(np.flatnonzero(fs==f)[0])
                    target_loss=float(np.median(d['g'][test[ok],jf]/(2*np.pi*f*ct)))
                row=dict(device=dev,group=d['group'],window=window,condition=label,frequency_Hz=f,
                         omitted_test_points=int((~ok).sum()),reference_r2=fit['r2'],reference_loss_median=loss,
                         target_loss_median=target_loss,**metrics)
                rows.append(row)
                mu,pl,pu=predict(fit,vv);yy=(cref/ct)**2
                if window in ['primary','low_04_08','low_05_10']:
                    for vi,ci,yi,mi,li,ui in zip(vv,ct,yy,mu,pl,pu):
                        points.append(dict(device=dev,group=d['group'],window=window,condition=label,frequency_Hz=f,V=vi,C_F=ci,
                                           Cref_F=cref,y=yi,mu_y=mi,lower_y=li,upper_y=ui,included=int(li<=yi<=ui)))
                    # Separately refit each condition in its own threshold window, using the same
                    # absolute C reference. This is a parameter-stability diagnostic, not transfer.
                    own=window_indices(cv,cc,cref,low,high)
                    if len(own)>=15:
                        ownfit=fit_line(cv[own],(cref/cc[own])**2);op=parameter_record(ownfit)
                        refits.append(dict(device=dev,group=d['group'],window=window,condition=label,frequency_Hz=f,
                                           n=len(own),Vmin=float(cv[own[0]]),Vmax=float(cv[own[-1]]),
                                           **op))
            # Contiguous-block holdout: fit first 75% of reference window, predict remaining 25%.
            nsplit=int(np.floor(.75*len(ix)));bt=ix[:nsplit];be=ix[nsplit:]
            if len(be)>=4 and len(bt)>=8:
                bf=fit_line(v[bt],(cref/c[bt])**2)
                rows.append(dict(device=dev,group=d['group'],window=window,condition='blocked_same_2MHz',frequency_Hz=2e6,
                                 omitted_test_points=0,reference_r2=bf['r2'],reference_loss_median=loss,**score(bf,v[be],c[be],cref)))
    write_csv('file_audit.csv',audit);write_csv('device_inventory.csv',inventory)
    write_csv('exclusions.csv',excluded);write_csv('reference_fits.csv',fits)
    write_csv('device_scores.csv',rows);write_csv('prediction_points.csv',points)
    write_csv('condition_refits.csv',refits);write_csv('hysteresis_halfcap.csv',hysteresis)
    metrics=['mape','mape_valid_C','y_relative_rmse_all','y_mae_all','y_rmse_all','nrmse_Cref','inclusion','invalid_prediction_fraction','reference_r2','paired_abs_change_Cref','paired_relative_change']
    summaries=summarize(rows,['window','condition'],metrics)
    write_csv('summary.csv',summaries)
    write_csv('group_summary.csv',summarize(rows,['window','group','condition'],metrics))
    write_csv('low_loss_summary.csv',summarize([r for r in rows if 0<=r['reference_loss_median']<=.1],['window','condition'],metrics))
    write_csv('both_low_loss_summary.csv',summarize([r for r in rows if 0<=r['reference_loss_median']<=.1 and 0<=r.get('target_loss_median',np.nan)<=.1],['window','condition'],metrics))
    write_csv('hysteresis_summary.csv',summarize(hysteresis,['group','frequency_Hz'],['abs_shift_V']))
    report=dict(raw_export_counts=raw_counts,primary_device_count=len(ds),groups=dict(Counter(d['group'] for d in ds)),
                primary_eligible_devices=sum(f['window']=='primary' for f in fits),
                parser_audit=dict(Counter(r['status'] for r in audit)),
                duplicate_frequency_label_devices=sum(bool(d['duplicate_frequency_labels']) for d in inventory),
                archive_sha256=hashlib.sha256((ROOT/'insb_source_data.zip').read_bytes()).hexdigest(),
                python=platform.python_version(),numpy=np.__version__,
                selected_summary=[r for r in summaries if r['window'] in ['primary','low_04_08','low_05_10'] and r['condition'] in ['same_2MHz','forward_5000Hz','forward_100000Hz','reverse_2e+06Hz','blocked_same_2MHz']],
                excluded_counts=dict(Counter((r['window']+':'+r['reason']) for r in excluded)))
    (ROOT/'insb_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='selected_summary'},indent=2))
    for r in report['selected_summary']:
        print(r['window'], r['condition'], r['n_devices'], 'R2', round(r['reference_r2_median'],4),
              'MAPE%',round(100*r['mape_median'],2),'PI%',round(100*r['inclusion_median'],1),
              'invalid%',round(100*r['invalid_prediction_fraction_median'],1))


if __name__=='__main__':
    main()
