"""Chapter 5: cross-endpoint quantiles, paired secants, and energy benchmark.

All 123 digitized points are evaluated with the same saved joint posterior,
without refitting, posterior truncation, or voltage-grid selection. Membership
in the actual primary CSV defines the fitting and non-fitting groups.
Intervals are pointwise and conditional on the model and digitization.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / 'data'
OUT = DATA / 'secant_bridge_results'
ORDER = ['n1_500nm', 'n1_750nm', 'n2_750nm']
STEPS = [0.5, 0.25, 0.125]
Q = 1.602176634e-19
EPS = 10.6 * 8.854187817e-14
THICKNESS = {s: float(s.split('_')[1][:-2])*1e-7 for s in ORDER}
VERSION = 'secant_bridge_20261102_full123'
METHODS = ['cross_endpoint', 'paired_secant', 'credible', 'predictive']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def groups(rows):
    for s in ORDER+['pooled']:
        sr = rows if s == 'pooled' else [r for r in rows if r['series'] == s]
        for label, selected in [('fitting', [r for r in sr if r['used_in_primary_fit']]),
                                ('out-of-window', [r for r in sr if not r['used_in_primary_fit']]),
                                ('all', sr)]:
            yield s, label, selected


def cross_interval(v, h, vbi, k):
    """Cross-endpoint marginal charge quantiles with median-parameter remainder."""
    factor = np.sqrt(2*EPS*Q*k)
    lo_m, hi_m = np.quantile(factor*np.sqrt(vbi-v+h), [.025, .975])
    lo_p, hi_p = np.quantile(factor*np.sqrt(vbi-v-h), [.025, .975])
    lo0, hi0 = (lo_m-hi_p)/(2*h), (hi_m-lo_p)/(2*h)
    vm, km = np.median(vbi), np.median(k)
    fm = np.sqrt(2*EPS*Q*km)
    remainder = (3/8)*fm*(vm-v-h)**(-2.5)*h**2/6
    return dict(cross_lower_F=float(lo0-remainder), cross_upper_F=float(hi0+remainder),
                cross_lower_without_remainder_F=float(lo0), cross_upper_without_remainder_F=float(hi0),
                cross_remainder_F=float(remainder))


def paired_interval(v, h, vbi, k, area, thickness):
    b = vbi-v
    transition = Q*(k/area**2)*thickness**2/(2*EPS)
    valid = np.isfinite(b) & np.isfinite(k) & np.isfinite(area) & np.isfinite(transition) & (k > 0) & (area > 0) & (b-h > 0) & (b+h < transition)
    # This publication's fixed point set is valid for every saved draw. Fail on
    # any violation rather than changing its posterior or evaluation population.
    if not valid.all():
        raise ValueError(f'Invalid secant domain at V={v}, h={h}: {np.sum(~valid)} draws')
    factor = np.sqrt(2*EPS*Q*k)
    exact = np.sqrt(EPS*Q*k/(2*b))
    # Rationalization avoids subtracting close square roots as h -> 0.
    secant = factor/(np.sqrt(b+h)+np.sqrt(b-h))
    remainder = (3/8)*factor*(b-h)**(-2.5)*h**2/6
    lower, upper = secant-remainder, secant+remainder
    sample_violation = np.maximum(np.maximum(lower-exact, exact-upper), 0)/exact
    assert np.max(sample_violation) <= 5e-14
    low, high = np.quantile(lower, .025), np.quantile(upper, .975)
    ci_low, median, ci_high = np.quantile(exact, [.025, .5, .975])
    assert low <= ci_low and high >= ci_high
    sec_low, sec_high = np.quantile(secant, [.025, .975])
    return dict(paired_lower_F=float(low), paired_upper_F=float(high),
                paired_secant_lower_F=float(sec_low), paired_secant_upper_F=float(sec_high),
                paired_remainder_median_F=float(np.median(remainder)),
                paired_remainder_max_F=float(np.max(remainder)),
                recomputed_C_credible_lower_F=float(ci_low), recomputed_C_median_F=float(median),
                recomputed_C_credible_upper_F=float(ci_high),
                invalid_secant_draw_count=int(np.sum(~valid)),
                minimum_flatband_margin_V=float(np.min(b-h)),
                minimum_transition_margin_secant_V=float(np.min(transition-b-h)),
                max_samplewise_relative_violation=float(np.max(sample_violation)),
                lower_quantile_margin_F=float(ci_low-low), upper_quantile_margin_F=float(high-ci_high),
                paired_excess_relative_width=float(((high-low)-(ci_high-ci_low))/(2*median)))


def bounds(row, method):
    if method == 'cross_endpoint':
        return row['cross_lower_F'], row['cross_upper_F']
    if method == 'paired_secant':
        return row['paired_lower_F'], row['paired_upper_F']
    return row['C_'+method+'_lower_F'], row['C_'+method+'_upper_F']


def summary(s, label, rows, method):
    pairs = [bounds(r, method) for r in rows]
    inside = [lo <= r['C_observed_F'] <= hi for r, (lo, hi) in zip(rows, pairs)]
    widths = [(hi-lo)/(2*r['C_median_F']) for r, (lo, hi) in zip(rows, pairs)]
    return dict(series=s, evaluation_set=label, method=method, h_V=rows[0]['h_V'],
                n_test_points=len(rows), inside_count=int(sum(inside)), coverage=float(np.mean(inside)),
                mean_relative_halfwidth=float(np.mean(widths)),
                negative_lower_bound_count=int(sum(lo < 0 for lo, _ in pairs)))


def energy_charge_secant(x, delta, factor, direction):
    """Stable divided difference of Etotal=(2/3)*factor*b**(3/2)."""
    lo, hi = (x-delta, x) if direction == 'minus' else (x, x+delta)
    return (2/3)*factor*(hi+np.sqrt(hi*lo)+lo)/(np.sqrt(hi)+np.sqrt(lo))


def energy_benchmark(rows, arrays):
    results = []
    for r in rows:
        s, v = r['series'], r['V_app_V']
        vm, km, am = (float(np.median(arrays[s+'__Vbi_V'])),
                      float(np.median(arrays[s+'__k_NA2'])), float(np.median(arrays['area_cm2'])))
        b, factor = vm-v, np.sqrt(2*EPS*Q*km)
        transition = Q*(km/am**2)*THICKNESS[s]**2/(2*EPS)
        c = factor/(2*np.sqrt(b))
        q = factor*np.sqrt(b)
        for h in STEPS:
            for refinement in [1, 4]:
                delta = h*h/50/refinement  # V: delta=h^2/(50 V), then delta/4.
                assert b-h-delta > 0 and b+h+delta < transition
                qm = energy_charge_secant(b, delta, factor, 'minus')
                qp = energy_charge_secant(b, delta, factor, 'plus')
                mean_c = factor/(np.sqrt(b+h)+np.sqrt(b-h))
                cl0 = (energy_charge_secant(b+h, delta, factor, 'minus')-
                       energy_charge_secant(b-h, delta, factor, 'plus'))/(2*h)
                cu0 = (energy_charge_secant(b+h, delta, factor, 'plus')-
                       energy_charge_secant(b-h, delta, factor, 'minus'))/(2*h)
                remainder = (3/8)*factor*(b-h)**(-2.5)*h*h/6
                cl, cu = cl0-remainder, cu0+remainder
                violation = max((qm-q)/q, (q-qp)/q, (cl0-mean_c)/mean_c,
                                (mean_c-cu0)/mean_c, (cl-c)/c, (c-cu)/c, 0)
                assert violation <= 1e-12
                # Direct differences are a separately evaluated check of the
                # rationalized energy quotients used above.
                energy = lambda x: (2/3)*factor*x**1.5
                direct_qm = (energy(b)-energy(b-delta))/delta
                direct_qp = (energy(b+delta)-energy(b))/delta
                np.testing.assert_allclose([qm, qp], [direct_qm, direct_qp], rtol=2e-10, atol=0)
                results.append(dict(series=s, observation_id=r['observation_id'], V_app_V=v,
                    Vbi_V=vm, k_NA2=km, area_cm2=am, b_V=b, h_V=h, delta_V=delta,
                    delta_refinement=refinement, exact_total_energy_J=float(energy(b)),
                    energy_gap_J=0.0, exact_Q_C=float(q), Q_lower_C=float(qm), Q_upper_C=float(qp),
                    exact_C_F=float(c), exact_mean_C_F=float(mean_c),
                    mean_C_lower_F=float(cl0), mean_C_upper_F=float(cu0),
                    remainder_F=float(remainder), C_lower_F=float(cl), C_upper_F=float(cu),
                    relative_halfwidth=float((cu-cl)/(2*c)), max_relative_violation=float(violation),
                    minimum_domain_margin_V=float(min(b-h-delta, transition-b-h-delta))))
    for r in rows:
        sub = [x for x in results if x['series']==r['series'] and x['observation_id']==r['observation_id']]
        for h in STEPS:
            coarse, fine = [x for x in sub if x['h_V']==h]
            assert fine['C_lower_F'] >= coarse['C_lower_F']-1e-12*fine['exact_C_F']
            assert fine['C_upper_F'] <= coarse['C_upper_F']+1e-12*fine['exact_C_F']
        for refinement in [1,4]:
            widths = [x['relative_halfwidth'] for x in sub if x['delta_refinement']==refinement]
            assert np.all(np.diff(widths) < 0)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    current_path = DATA/'CURRENT_TABLE3.json'
    current = json.loads(current_path.read_text())
    coverage_pointer = json.loads((DATA/'CURRENT_COVERAGE.json').read_text())
    assert sha(PROJECT/coverage_pointer['results_json']) == coverage_pointer['results_json_sha256']
    assert sha(PROJECT/current['draws_npz']) == current['draws_npz_sha256']
    arrays = np.load(PROJECT/current['draws_npz'], allow_pickle=False)
    source_rows = read_csv(DATA/'coverage_results/coverage_points.csv')
    for r in source_rows:
        for key in r:
            if key == 'series': continue
            r[key] = int(r[key]) if key in {'observation_id','used_in_primary_fit','inside_credible','inside_predictive','invalid_model_draw_count','nonpositive_predictive_y_count'} else float(r[key])
    expected = {}
    for s in ORDER:
        full = np.genfromtxt(PROJECT/current['series'][s]['full_csv'], delimiter=',', names=True)
        primary = np.genfromtxt(PROJECT/current['series'][s]['primary_csv'], delimiter=',', names=True)
        fitted = np.isin(full['V_app_V'], primary['V_app_V'])
        assert int(fitted.sum()) == len(primary)
        for i, datum in enumerate(full):
            expected[s, i] = (float(datum['V_app_V']), float(datum['C_F']), int(fitted[i]))
    assert len(source_rows) == len(expected) == 123
    assert len({(r['series'], r['observation_id']) for r in source_rows}) == len(source_rows)
    for r in source_rows:
        assert (r['V_app_V'], r['C_observed_F'], r['used_in_primary_fit']) == expected[r['series'], r['observation_id']]
    fit_count = sum(r['used_in_primary_fit'] for r in source_rows)
    assert fit_count == 43
    draw_count = len(arrays['area_cm2'])
    points = []
    for s in ORDER:
        vbi, k, area = arrays[s+'__Vbi_V'], arrays[s+'__k_NA2'], arrays['area_cm2']
        assert len(vbi) == len(k) == len(area) == draw_count
        for h in STEPS:
            for source in [r for r in source_rows if r['series'] == s]:
                row = dict(source, h_V=h)
                # Check every paired endpoint domain before evaluating either method.
                row.update(paired_interval(row['V_app_V'],h,vbi,k,area,THICKNESS[s]))
                row.update(cross_interval(row['V_app_V'],h,vbi,k))
                np.testing.assert_array_equal([row['recomputed_C_'+suffix] for suffix in ['credible_lower_F','median_F','credible_upper_F']],
                                              [row['C_'+suffix] for suffix in ['credible_lower_F','median_F','credible_upper_F']])
                for method, key in [('cross_endpoint','cross'),('paired_secant','paired')]:
                    lo, hi = bounds(row, method)
                    row['relative_halfwidth_'+key] = (hi-lo)/(2*row['C_median_F'])
                row['relative_halfwidth_cross_without_remainder'] = (row['cross_upper_without_remainder_F']-row['cross_lower_without_remainder_F'])/(2*row['C_median_F'])
                points.append(row)
        print(f'{s}: all {draw_count} draws valid at all observed points and three h values',flush=True)
    base = [r for r in points if r['h_V'] == STEPS[0]]
    comparison = [summary(s,label,rows,method) for s,label,rows in groups(base) for method in METHODS]
    sensitivity = []
    for h in STEPS:
        for s,label,rows in groups([r for r in points if r['h_V']==h]):
            stats = {m:summary(s,label,rows,m) for m in METHODS}
            sensitivity.append(dict(series=s,evaluation_set=label,h_V=h,n_test_points=len(rows),
                cross_coverage=stats['cross_endpoint']['coverage'],paired_coverage=stats['paired_secant']['coverage'],
                cross_mean_relative_halfwidth=stats['cross_endpoint']['mean_relative_halfwidth'],
                paired_mean_relative_halfwidth=stats['paired_secant']['mean_relative_halfwidth'],
                credible_mean_relative_halfwidth=stats['credible']['mean_relative_halfwidth'],
                predictive_mean_relative_halfwidth=stats['predictive']['mean_relative_halfwidth'],
                max_paired_excess_relative_width=max(r['paired_excess_relative_width'] for r in rows),
                mean_paired_excess_relative_width=float(np.mean([r['paired_excess_relative_width'] for r in rows])),
                minimum_flatband_margin_V=min(r['minimum_flatband_margin_V'] for r in rows),
                minimum_transition_margin_V=min(r['minimum_transition_margin_secant_V'] for r in rows),
                invalid_secant_draw_count=sum(r['invalid_secant_draw_count'] for r in rows)))
    # Strict, samplewise nesting can also be established for this square-root
    # branch; the publication reports the more general quantile inclusion test.
    for original in source_rows:
        series_points=[r for r in points if r['series']==original['series'] and r['observation_id']==original['observation_id']]
        assert np.all(np.diff([r['paired_excess_relative_width'] for r in series_points]) < 0)
    cross=[]
    for s,label,rows in groups(base):
        if label != 'all':continue
        without=[r['cross_lower_without_remainder_F'] <= r['C_observed_F'] <= r['cross_upper_without_remainder_F'] for r in rows]
        cross.append(dict(series=s,n_test_points=len(rows),
            inside_with_remainder=sum(r['cross_lower_F'] <= r['C_observed_F'] <= r['cross_upper_F'] for r in rows),
            coverage_with_remainder=float(np.mean([r['cross_lower_F'] <= r['C_observed_F'] <= r['cross_upper_F'] for r in rows])),
            inside_without_remainder=int(sum(without)),coverage_without_remainder=float(np.mean(without)),
            mean_relative_halfwidth=float(np.mean([r['relative_halfwidth_cross'] for r in rows])),
            mean_relative_halfwidth_without_remainder=float(np.mean([r['relative_halfwidth_cross_without_remainder'] for r in rows])),
            mean_remainder_relative_to_C_median=float(np.mean([r['cross_remainder_F']/r['C_median_F'] for r in rows])),
            maximum_relative_halfwidth=max(r['relative_halfwidth_cross'] for r in rows),
            negative_lower_bound_count=sum(r['cross_lower_F'] < 0 for r in rows),
            n_primary_fit_points_in_test=sum(r['used_in_primary_fit'] for r in rows),
            n_outside_primary_fit=sum(not r['used_in_primary_fit'] for r in rows),
            inside_outside_primary_with_remainder=sum(r['cross_lower_F']<=r['C_observed_F']<=r['cross_upper_F'] for r in rows if not r['used_in_primary_fit']),
            inside_outside_primary_without_remainder=sum(r['cross_lower_without_remainder_F']<=r['C_observed_F']<=r['cross_upper_without_remainder_F'] for r in rows if not r['used_in_primary_fit'])))
    energy=energy_benchmark(source_rows,arrays)
    outputs={'interval_points.csv':points,'comparison_summary.csv':comparison,
             'table6_summary.csv':[r for r in comparison if r['series']=='pooled' and r['evaluation_set']!='all'],
             'sensitivity_summary.csv':sensitivity,'table7_summary.csv':[r for r in sensitivity if r['evaluation_set']=='all'],
             'cross_endpoint_summary.csv':cross,'energy_benchmark.csv':energy}
    for name,rows in outputs.items():write_csv(OUT/name,rows)
    input_paths=[current_path,DATA/'CURRENT_COVERAGE.json',PROJECT/current['draws_npz'],PROJECT/current['results_json'],
                 DATA/'coverage_results/coverage_points.csv',PROJECT/coverage_pointer['results_json'],
                 DATA/'coverage_results/point_membership.csv', DATA/'coverage_results/voltage_segments.csv']
    inputs={str(p.relative_to(PROJECT)):sha(p) for p in input_paths}
    results=dict(version=VERSION,status='current_calculation',inputs_sha256=inputs,script_sha256=sha(Path(__file__)),
        point_count=len(source_rows),fit_count=fit_count,out_of_window_count=len(source_rows)-fit_count,draw_count_per_film=draw_count,steps_V=STEPS,
        shared_evaluation_membership=True,samples_discarded=0,invalid_secant_draw_count=0,
        sample_point_step_checks=len(points)*draw_count,
        maximum_samplewise_relative_violation=max(r['max_samplewise_relative_violation'] for r in points),
        minimum_lower_quantile_margin_F=min(r['lower_quantile_margin_F'] for r in points),
        minimum_upper_quantile_margin_F=min(r['upper_quantile_margin_F'] for r in points),
        minimum_flatband_margin_V=min(r['minimum_flatband_margin_V'] for r in points),
        minimum_transition_margin_V=min(r['minimum_transition_margin_secant_V'] for r in points),
        paired_excess_width_decreases_at_every_point=True,
        direct_intervals_reused_from_current_coverage=True,
        RHW_definition='(upper-lower)/(2*pointwise posterior median model C), for every method and every reported summary',
        domain_scope='All finite saved posterior draws were checked. This does not establish physical admissibility over the full support of the untruncated posterior',
        evaluation_rule='All digitized observations; actual primary CSV membership; same points and joint posterior draws at every h and for every method',
        inclusion_rate_interpretation='Descriptive observed-point inclusion, not a nominal calibration test for credible or secant bands; fitting reuse and same-curve non-fitting points are not independent experimental validation',
        interval_interpretation='paired secant quantile envelope contains the equal-tail pointwise model credible interval on the same joint draws; not an observation prediction interval or simultaneous band',
        cross_interpretation='cross-endpoint marginal Q quantiles plus median-parameter remainder; no claimed nominal 95% capacitance coverage',
        energy_benchmark=dict(test_count=len(energy),fixed_parameters='marginal medians of Vbi,k,area for each film',
            energy='total E=A*mathcalE=(2/3)*sqrt(2*eps*q*k)*b^(3/2)',energy_gap=0,
            charge_bracket_has_nonzero_width=True,delta_rule='delta=h^2/(50 V), and delta/4',
            all_energy_charge_capacitance_bounds_pass=True,all_delta_refinements_narrow=True,
            all_h_delta_joint_refinements_narrow=True,maximum_relative_violation=max(r['max_relative_violation'] for r in energy),
            minimum_domain_margin_V=min(r['minimum_domain_margin_V'] for r in energy),
            general_primal_dual_solver_tested=False),
        pooled_comparison=outputs['table6_summary.csv'],
        pooled_sensitivity=[r for r in sensitivity if r['series']=='pooled' and r['evaluation_set']=='all'],
        environment=dict(python=platform.python_version(),numpy=np.__version__))
    write_json(OUT/'results.json',results)
    produced={str((OUT/name).relative_to(PROJECT)):sha(OUT/name) for name in list(outputs)+['results.json']}
    write_json(OUT/'manifest.json',dict(version=VERSION,inputs_sha256=inputs,script_sha256=sha(Path(__file__)),outputs_sha256=produced))
    write_json(DATA/'CURRENT_SECANT_BRIDGE.json',dict(version=VERSION,directory=str(OUT.relative_to(PROJECT)),
        results_json=str((OUT/'results.json').relative_to(PROJECT)),results_json_sha256=sha(OUT/'results.json'),
        manifest_json=str((OUT/'manifest.json').relative_to(PROJECT)),manifest_json_sha256=sha(OUT/'manifest.json'),
        table6_csv=str((OUT/'table6_summary.csv').relative_to(PROJECT)),table7_csv=str((OUT/'table7_summary.csv').relative_to(PROJECT)),
        code='code/recalculate_secant_bridge.py',figure_code='paper/make_paper_figs.py'))
    print(json.dumps({k:results[k] for k in ['sample_point_step_checks','maximum_samplewise_relative_violation','minimum_flatband_margin_V','minimum_transition_margin_V','energy_benchmark','pooled_comparison','pooled_sensitivity']},indent=2),flush=True)


if __name__=='__main__':
    main()
