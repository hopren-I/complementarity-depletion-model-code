"""Check data integrity, posterior transforms, numerical results and interval constructions."""
from pathlib import Path
import csv,hashlib,json,re,sys,subprocess
import numpy as np
from provenance_checks import check_archived_hash, CSV_LINE_ENDING_MATCHES
P=Path(__file__).resolve().parents[1]
def read_rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_hashes(entries, record):
    for relative, expected in entries.items():
        check_archived_hash(P/relative,expected,P)
manifest=json.loads((P/'data/manifest.json').read_text())
raw=[]
for series,meta in manifest['series'].items():
    for key in ['full_csv','primary_csv','details_csv']:
        f=P/meta[key];check_archived_hash(f,meta[key+'_sha256'],P)
    full=np.genfromtxt(P/meta['full_csv'],delimiter=',',names=True);primary=np.genfromtxt(P/meta['primary_csv'],delimiter=',',names=True)
    assert len(full)==meta['full_observation_count'] and len(primary)==meta['primary_observation_count']
    np.testing.assert_array_equal(full[meta['primary_observation_ids']],primary)
    raw.append((series,len(full),len(primary)))
# Validate the coverage provenance chain, including metadata-only input changes.
coverage_rel='data/coverage_results/coverage_results.json'
coverage=json.loads((P/coverage_rel).read_text())
coverage_manifest=json.loads((P/'data/coverage_results/manifest.json').read_text())
coverage_pointer=json.loads((P/'data/CURRENT_COVERAGE.json').read_text())
for record, value in [(coverage_rel, coverage), ('data/coverage_results/manifest.json', coverage_manifest)]:
    assert 'data/CURRENT_TABLE3.json' in value['inputs_sha256'], record
    check_hashes(value['inputs_sha256'], record)
check_hashes(coverage_manifest['outputs_sha256'], 'data/coverage_results/manifest.json')
coverage_hash=sha(P/coverage_rel)
assert coverage_manifest['outputs_sha256'][coverage_rel]==coverage_hash
assert coverage_pointer['results_json']==coverage_rel
assert coverage_pointer['results_json_sha256']==coverage_hash, 'data/CURRENT_COVERAGE.json: stale results_json_sha256'
post=json.loads((P/'data/results/posteriors.json').read_text());draws=np.load(P/'data/results/posterior_draws.npz')
for series,s in post['series'].items():
    beta=draws[series+'__beta_scaled'];vbi=draws[series+'__Vbi_V']
    assert beta.shape==(200000,2)
    np.testing.assert_array_equal(-beta[:,0]/beta[:,1],vbi)
    np.testing.assert_allclose([vbi.mean(),*np.percentile(vbi,[2.5,97.5])],[s['posterior']['Vbi_V']['mc_sample_mean'],*s['posterior']['Vbi_V']['equal_tail_95pct_CI']],rtol=1e-14)
t2=read_rows(P/'data/results/table2_summary.csv')
areas=[]
for r in t2:
    d=np.genfromtxt(P/'data'/(r['series']+'.csv'),delimiter=',',names=True)
    capacitance=float(np.median(d['C_F'][d['V_app_V']<=-1.5]))
    area=capacitance*float(r['thickness_nm'])*1e-7/(10.6*8.854187817e-14)
    np.testing.assert_allclose([float(r['median_capacitance_F']),float(r['area_cm2'])],[capacitance,area],rtol=1e-14)
    areas.append(area)
area_model=json.loads((P/'data/baseline_calibration.json').read_text())['area_model']
np.testing.assert_allclose([np.mean(areas),np.std(areas,ddof=1)],[area_model['mean_cm2'],area_model['sd_cm2']],rtol=1e-14)
si=np.load(P/'data/results/post_Si_beta.npy',allow_pickle=False)
assert si.shape==(200000,2)
si_data=np.genfromtxt(P/'data/si_ms_inset_digitized.csv',delimiter=',',names=True)
n_si=int(np.count_nonzero(si_data['V_app_V']<=-1.0))
fig1=json.loads((P/'data/results/figure1_summary.json').read_text())
assert fig1['posterior_source']=='data/results/post_Si_beta.npy'
assert fig1['posterior_source_sha256']==sha(P/fig1['posterior_source'])
assert (fig1['draw_count'],fig1['n_fit'],fig1['degrees_of_freedom'])==(len(si),n_si,n_si-2)
si_vbi=-si[:,0]/si[:,1]
si_N=-2/(1.602176634e-19*11.7*8.854187817e-14*si[:,1])
si_expected=json.loads((P/'data/results/baseline_results.json').read_text())['Si_standard']
for key,values in [('Vbi_V',si_vbi),('N_cm3',si_N)]:
    summary=[values.mean(),*np.percentile(values,[2.5,97.5])]
    np.testing.assert_allclose(summary,si_expected[key],rtol=1e-14)
    np.testing.assert_allclose([np.median(values),*np.percentile(values,[2.5,97.5])],fig1[key],rtol=1e-14)
t3=read_rows(P/'data/results/table3_summary.csv');t4=read_rows(P/'data/window_results/table4_values.csv');t5=read_rows(P/'data/coverage_results/table5_summary.csv')
# The final paper evaluates every archived observation exactly once. Membership
# is the actual calibration input, not a second voltage-threshold calculation.
membership=read_rows(P/'data/coverage_results/point_membership.csv')
points=read_rows(P/'data/coverage_results/coverage_points.csv')
key=lambda r:(r['series'],int(r['observation_id']))
expected_keys={(s,i) for s,m in manifest['series'].items() for i in range(m['full_observation_count'])}
assert len(points)==len(membership)==len(expected_keys)==123
assert len({key(r) for r in points})==len(points)
assert {key(r) for r in points}=={key(r) for r in membership}==expected_keys
for r in membership:
    meta=manifest['series'][r['series']]
    assert int(r['included_in_coverage'])==1
    assert int(r['used_in_primary_fit'])==int(int(r['observation_id']) in meta['primary_observation_ids'])
assert sum(int(r['used_in_primary_fit']) for r in points)==43
member_by_key={key(r):r for r in membership}
for r in points:
    m=member_by_key[key(r)]
    assert int(r['used_in_primary_fit'])==int(m['used_in_primary_fit'])
    assert float(r['V_app_V'])==float(m['V_app_V'])
    assert float(r['C_observed_F'])==float(m['C_F'])
    assert int(r['invalid_model_draw_count'])==int(r['nonpositive_predictive_y_count'])==0
    for method in ['credible','predictive']:
        lo,hi=float(r[f'C_{method}_lower_F']),float(r[f'C_{method}_upper_F'])
        assert int(r[f'inside_{method}'])==int(lo<=float(r['C_observed_F'])<=hi)
        np.testing.assert_allclose(float(r[f'relative_halfwidth_{method}']),
                                   (hi-lo)/(2*float(r['C_median_F'])),rtol=1e-14)
for summary in t5:
    selected=[r for r in points if (summary['series']=='pooled' or r['series']==summary['series'])
              and bool(int(r['used_in_primary_fit']))==(summary['evaluation_set']=='fitting')]
    assert len(selected)==int(summary['n_test_points'])
    for method in ['credible','predictive']:
        count=sum(int(r[f'inside_{method}']) for r in selected)
        assert count==int(summary[f'inside_{method}'])
        np.testing.assert_allclose(float(summary[f'coverage_{method}']),count/len(selected),rtol=1e-14)
        np.testing.assert_allclose(float(summary[f'mean_relative_halfwidth_{method}']),
                                   np.mean([float(r[f'relative_halfwidth_{method}']) for r in selected]),rtol=1e-14)
# Check the fitted series and stored calculation summaries.
allowed={'n1_500nm','n1_750nm','n2_750nm'}
assert set(manifest['series'])==set(post['series'])==allowed
assert all(key=='area_cm2' or key.split('__')[0] in allowed for key in draws.files)
assert len(t3)==3 and len(t4)==4
for label,directory in [('windows','window_results'),('bridge','secant_bridge_results')]:
    m=json.loads((P/'data'/directory/'manifest.json').read_text())
    check_hashes(m['inputs_sha256'], label)
    if 'outputs_sha256' in m:check_hashes(m['outputs_sha256'],label)
    if 'files_sha256' in m:
        check_hashes({'data/'+directory+'/'+k:v for k,v in m['files_sha256'].items()},label)
cross=read_rows(P/'data/secant_bridge_results/cross_endpoint_summary.csv')
assert [r['series'] for r in cross]==['n1_500nm','n1_750nm','n2_750nm','pooled']

# Current parameter summaries use medians of the already archived draws.
order = ['n1_500nm','n1_750nm','n2_750nm']
round_values = lambda values,digits: [float(f'{float(v):.{digits}f}') for v in values]
validation = json.loads((P/'data/validation/source_manifest.json').read_text())
for rel,entry in validation['files'].items():
    check_hashes({rel:entry['sha256']},'validation/source_manifest.json')
window_draws=np.load(P/'data/window_results/posterior_draws.npz',allow_pickle=False)
reading=read_rows(P/'data/digitization/reading_sensitivity.csv')
sensitivity={r['series']:r for r in read_rows(P/'data/validation/sensitivity.csv')}
expected_parameter_rows=[]
for series in order:
    v=draws[series+'__Vbi_V'];k=draws[series+'__k_NA2']/1e12;n=draws[series+'__N_cm_minus3']/1e17
    np.testing.assert_allclose(n,draws[series+'__k_NA2']/draws['area_cm2']**2/1e17,rtol=1e-14)
    ranges=[]
    for key in window_draws.files:
        if key.split('__')[1]==series:
            beta=window_draws[key];ranges.append(float(np.median(-beta[:,0]/beta[:,1])))
    rr=[float(r['OLS_Vbi_V']) for r in reading if r['series']==series]
    ss=sensitivity[series]
    np.testing.assert_allclose([min(ranges),max(ranges),min(rr),max(rr)],
        [float(ss[x]) for x in ['window_median_min','window_median_max','reading_OLS_min','reading_OLS_max']],rtol=1e-12)
    expected_parameter_rows.append([
        round_values([np.median(v),*np.quantile(v,[.025,.975])],3),
        [float(next(r['published_Vbi_V'] for r in t3 if r['series']==series))],
        round_values([np.median(k)],2),
        round_values([np.median(n),*np.quantile(n,[.025,.975])],2),
        round_values([min(ranges),max(ranges)],3),
        round_values([min(rr),max(rr)],3)])
comparison=read_rows(P/'data/secant_bridge_results/comparison_summary.csv')
methods=['cross_endpoint','paired_secant','credible','predictive']
expected_coverage=[]
for method in methods:
    allrow=next(r for r in comparison if r['series']=='pooled' and r['evaluation_set']=='all' and r['method']==method)
    outrow=next(r for r in comparison if r['series']=='pooled' and r['evaluation_set']=='out-of-window' and r['method']==method)
    expected_coverage.append(round_values([100*float(allrow['coverage']),100*float(outrow['coverage'])],1)
                             +round_values([100*float(allrow['mean_relative_halfwidth'])],2))

# InSb raw-analysis outputs are independently frozen; figures and historical
# tables have since been restyled and are intentionally outside this old hash set.
insb=json.loads((P/'data/insb_artifact_manifest.json').read_text())
check_hashes({k:v for k,v in insb['sha256'].items() if k.startswith('data/')},'InSb scientific archive')
assert sha(P/'data/insb_source_data.zip')==insb['source_zip_sha256']
from generate_current_tables import table_text, rows as insb_rows
insb_data=insb_rows()
for language in ['en','zh']:
    assert (P/f'tables/insb_prediction_summary_{language}.tex').read_text()==table_text(language)

# Calculate synthetic-summary values directly from the 1000 saved repetitions.
correct=read_rows(P/'data/validation/bayes_repetitions.csv')
stress=read_rows(P/'data/validation/stress_repetitions.csv')
assert len(correct)==8000 and len(stress)==16000
recovery={}
for arm, expected_error, expected_recovery_coverage in [
        ('correct',[5.47,2.09],[94.5,94.5]),
        ('heteroscedastic',[5.21,2.09],[95.6,94.8]),
        ('correlated',[9.19,3.47],[69.1,69.7])]:
    recovery[arm]={}
    for index,target in enumerate(['Vbi','k']):
        selected=[r for r in (correct if arm=='correct' else stress)
                  if r['target']==target and (arm=='correct' or r['arm']==arm)]
        assert len(selected)==1000 and len({r['rep'] for r in selected})==1000
        error=np.mean([abs(float(r['estimate'])-float(r['truth']))/float(r['truth']) for r in selected])*100
        covered=[float(r['lower'])<=float(r['truth'])<=float(r['upper']) for r in selected]
        assert [int(r['covered']) for r in selected]==[int(v) for v in covered]
        actual_coverage=float(np.mean(covered))*100
        assert round(error,2)==expected_error[index],(arm,target,error)
        assert round(actual_coverage,1)==expected_recovery_coverage[index],(arm,target,actual_coverage)
        recovery[arm][target]={'mean_absolute_relative_error_percent':round(error,2),
                               'interval_coverage_percent':round(actual_coverage,1),'repetitions':1000}

# Recompute the restored six-row table from archived interval endpoints.
validation_manifest=json.loads((P/'data/validation/source_manifest.json').read_text())
for rel in ['data/validation/bounds_baseline.csv','data/validation/convergence.csv']:
    assert sha(P/rel)==validation_manifest['files'][rel]['sha256'],rel
baseline=read_rows(P/'data/validation/bounds_baseline.csv')
convergence=read_rows(P/'data/validation/convergence.csv')
numerical_bounds=[]
for mode in ['joint_refinement','fixed_steps']:
    candidates=[r for r in baseline if r['case']=='near_transition' and r['mode']==mode and int(r['n'])==256]
    assert len(candidates)==1
    r=candidates[0];assert float(r['b'])==0.49 and r['point_bound_applicable']=='True'
    scale=32/int(r['n'])
    expected_steps=[.003*(scale**.4 if mode=='joint_refinement' else 1),
                    .00075*(scale**.8 if mode=='joint_refinement' else 1)]
    np.testing.assert_allclose([float(r['h']),float(r['delta'])],expected_steps,rtol=1e-13)
    lo,hi,c,rem=[float(r[k]) for k in ['C_lower','C_upper','C_exact','remainder']]
    np.testing.assert_allclose([lo,hi],[float(r['mean_C_lower'])-rem,float(r['mean_C_upper'])+rem],rtol=1e-13)
    relative=(hi-lo)/(2*c)
    np.testing.assert_allclose(relative,float(r['point_relative_halfwidth']),rtol=1e-13)
    numerical_bounds.append({'method':mode,'elements':256,'relative_halfwidth_percent':100*relative})
for n in [256,1024,2048,4096]:
    candidates=[r for r in convergence if float(r['b'])==.49 and int(r['n'])==n]
    assert len(candidates)==1
    r=candidates[0];lo,hi,c,rem=[float(r[k]) for k in ['lower','upper','exact_C','remainder']]
    np.testing.assert_allclose([lo,hi],[float(r['average_lower'])-rem,float(r['average_upper'])+rem],rtol=1e-13)
    relative=(hi-lo)/(2*c)
    np.testing.assert_allclose(relative,float(r['relative_halfwidth']),rtol=1e-13)
    numerical_bounds.append({'method':'gap_based_search','elements':n,'relative_halfwidth_percent':100*relative})
expected_numerical_bounds=[[r['elements'],round(r['relative_halfwidth_percent'],2)] for r in numerical_bounds]
assert expected_numerical_bounds==[[256,343.04],[256,40.32],[256,12.45],[1024,3.11],[2048,1.56],[4096,.80]]

subprocess.run([sys.executable,str(P/'code/verify_secant_bridge.py')],cwd=P,check=True)
report={'passed':True,'raw_series_counts_full_primary':raw,'generated_InSb_tables_match_results':True,
        'parameter_summaries':'medians and equal-tail quantiles from archived arrays',
        'window_and_reading_ranges_independently_checked':True,'coverage_provenance_hashes_match':True,
        'evaluation_points':123,'primary_fit_points':43,'nonfitting_points':80,
        'posterior_draws_per_film':200000,'independent_secant_check':True,
        'synthetic_parameter_recovery':recovery,
        'numerical_bounds_table_recomputed':numerical_bounds,
        'historical_csv_line_endings':CSV_LINE_ENDING_MATCHES,
        'reference_comparison_report':'verification/reference_comparison.json'}
(P/'qa').mkdir(exist_ok=True)
(P/'verification/results_checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
