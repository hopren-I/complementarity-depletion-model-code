"""Independent membership, mathematical, summary, and provenance checks.

This verifier imports no calculation module. It uses the original full/primary
CSV files, saved posterior draws, direct endpoint differences, an analytic
Student-t predictive check, and an independent energy integral.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from scipy.stats import t
from provenance_checks import check_archived_hash, CSV_LINE_ENDING_MATCHES

P = Path(__file__).resolve().parents[1]
D = P / 'data/secant_bridge_results'
Q = 1.602176634e-19
EPS = 10.6*8.854187817e-14
STEPS = (0.5, 0.25, 0.125)
METHODS = {'cross_endpoint': ('cross_lower_F', 'cross_upper_F'),
           'paired_secant': ('paired_lower_F', 'paired_upper_F'),
           'credible': ('C_credible_lower_F', 'C_credible_upper_F'),
           'predictive': ('C_predictive_lower_F', 'C_predictive_upper_F')}


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def rows(name):
    return read(D/name)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, tolerance=2e-13):
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0)


def select(points, series, group, h=.5):
    return [p for p in points if float(p['h_V']) == h
            and (series == 'pooled' or p['series'] == series)
            and (group == 'all' or bool(int(p['used_in_primary_fit'])) == (group == 'fitting'))]


def statistics(selected, method):
    lower, upper = METHODS[method]
    inside = sum(float(p[lower]) <= float(p['C_observed_F']) <= float(p[upper]) for p in selected)
    widths = [(float(p[upper])-float(p[lower]))/(2*float(p['C_median_F'])) for p in selected]
    return inside, inside/len(selected), float(np.mean(widths))


def main():
    for directory, script in [(D, 'recalculate_secant_bridge.py'),
                              (P/'data/coverage_results', 'recalculate_coverage.py')]:
        manifest = json.loads((directory/'manifest.json').read_text())
        for mapping in (manifest['inputs_sha256'], manifest['outputs_sha256']):
            for relative, expected in mapping.items():
                check_archived_hash(P/relative, expected, P)
        assert sha(P/'code'/script) == manifest['script_sha256']
    for name in ['CURRENT_SECANT_BRIDGE.json', 'CURRENT_COVERAGE.json']:
        pointer = json.loads((P/'data'/name).read_text())
        assert sha(P/pointer['results_json']) == pointer['results_json_sha256']
    current = json.loads((P/'data/CURRENT_TABLE3.json').read_text())
    assert sha(P/current['draws_npz']) == current['draws_npz_sha256']
    assert sha(P/current['results_json']) == current['results_json_sha256']
    post = json.loads((P/current['results_json']).read_text())
    expected, expected_segments = {}, []
    for series, meta in current['series'].items():
        full = np.genfromtxt(P/meta['full_csv'], delimiter=',', names=True)
        primary = np.genfromtxt(P/meta['primary_csv'], delimiter=',', names=True)
        check_archived_hash(P/meta['full_csv'], meta['full_csv_sha256'], P)
        check_archived_hash(P/meta['primary_csv'], meta['primary_csv_sha256'], P)
        used = np.isin(full['V_app_V'], primary['V_app_V'])
        assert int(used.sum()) == len(primary)
        np.testing.assert_array_equal(full['C_F'][used], primary['C_F'])
        for i, datum in enumerate(full):
            expected[series, str(i)] = (float(datum['V_app_V']), float(datum['C_F']), int(used[i]))
        ids = np.argsort(full['V_app_V'])
        groups = np.split(ids, np.where(used[ids][1:] != used[ids][:-1])[0]+1)
        for segment_id, g in enumerate(groups, 1):
            expected_segments.append((series, segment_id, int(used[g[0]]), len(g),
                                      float(full['V_app_V'][g[0]]), float(full['V_app_V'][g[-1]])))
    original_rows = read(P/'data/coverage_results/coverage_points.csv')
    original = {(r['series'], r['observation_id']): r for r in original_rows}
    membership = read(P/'data/coverage_results/point_membership.csv')
    assert len(original_rows) == len(original) == len(membership) == len(expected) == 123
    assert set(original) == set(expected)
    assert len({(r['series'], r['observation_id']) for r in membership}) == len(expected)
    for row in membership:
        key = row['series'], row['observation_id']
        assert int(row['included_in_coverage']) == 1
        assert (float(row['V_app_V']), float(row['C_F']), int(row['used_in_primary_fit'])) == expected[key]
        assert row['evaluation_set'] == ('fitting' if expected[key][2] else 'out-of-window')
    for key, r in original.items():
        assert (float(r['V_app_V']), float(r['C_observed_F']), int(r['used_in_primary_fit'])) == expected[key]
        assert int(r['invalid_model_draw_count']) == int(r['nonpositive_predictive_y_count']) == 0
    assert sum(x[2] for x in expected.values()) == 43
    actual_segments = [(r['series'], int(r['segment_id']), int(r['used_in_primary_fit']), int(r['n_points']),
                        float(r['V_min_V']), float(r['V_max_V']))
                       for r in read(P/'data/coverage_results/voltage_segments.csv')]
    assert sorted(actual_segments) == sorted(expected_segments)

    points = rows('interval_points.csv')
    assert len(points) == len(original)*len(STEPS)
    for h in STEPS:
        selected = [r for r in points if float(r['h_V']) == h]
        assert len(selected) == len(original)
        assert sum(int(r['used_in_primary_fit']) for r in selected) == 43
        assert {(r['series'], r['observation_id']) for r in selected} == set(original)
    for r in points:
        baseline = original[r['series'], r['observation_id']]
        for key, value in baseline.items():
            if key != 'series':
                assert float(r[key]) == float(value), (r['series'], key)
        assert int(r['invalid_secant_draw_count']) == 0
        assert float(r['minimum_flatband_margin_V']) > 0 and float(r['minimum_transition_margin_secant_V']) > 0
        assert float(r['paired_lower_F']) <= float(r['C_credible_lower_F'])
        assert float(r['paired_upper_F']) >= float(r['C_credible_upper_F'])

    # Direct Q endpoint subtraction independently verifies every point and h.
    independent_checks, sample_checks, predictive_difference = 0, 0, 0.0
    with np.load(P/current['draws_npz'], allow_pickle=False) as saved:
        area = saved['area_cm2']
        assert np.all(np.isfinite(area) & (area > 0))
        for series in sorted(current['series']):
            vbi, k, beta = saved[series+'__Vbi_V'], saved[series+'__k_NA2'], saved[series+'__beta_scaled']
            assert len(vbi) == len(k) == len(beta) == len(area)
            close(vbi, -beta[:, 0]/beta[:, 1])
            close(k, -2/(Q*EPS*beta[:, 1]*1e18))
            factor = np.sqrt(2*EPS*Q*k)
            thickness = float(series.split('_')[1][:-2])*1e-7
            transition = Q*(k/area**2)*thickness**2/(2*EPS)
            fit = post['series'][series]['fit']
            bhat = np.asarray(fit['OLS_beta_scaled'])
            s2 = fit['residual_variance_scaled']
            inv = np.asarray(fit['t_scale_matrix_scaled'])/s2
            for r in [x for x in original_rows if x['series'] == series]:
                v, observed = float(r['V_app_V']), float(r['C_observed_F'])
                c_direct = 1/np.sqrt(1e18*(beta[:, 0]+beta[:, 1]*v))
                close(np.quantile(c_direct, [.025, .5, .975]),
                      [float(r['C_credible_lower_F']), float(r['C_median_F']), float(r['C_credible_upper_F'])])
                design = np.array([1.0, v])
                scale = np.sqrt(s2*(1+design@inv@design))
                y_quantiles = t.ppf([.975, .025], fit['posterior_df'], loc=design@bhat, scale=scale)
                assert np.all(y_quantiles > 0)
                analytic_pi = 1/np.sqrt(1e18*y_quantiles)
                mc_pi = np.array([float(r['C_predictive_lower_F']), float(r['C_predictive_upper_F'])])
                predictive_difference = max(predictive_difference, float(np.max(np.abs(mc_pi/analytic_pi-1))))
                assert int(analytic_pi[0] <= observed <= analytic_pi[1]) == int(r['inside_predictive'])
            for r in [x for x in points if x['series'] == series]:
                h, b = float(r['h_V']), vbi-float(r['V_app_V'])
                assert np.all(np.isfinite(k) & (k > 0) & np.isfinite(b) & (b-h > 0) & (b+h < transition))
                qm, qp = factor*np.sqrt(b+h), factor*np.sqrt(b-h)
                cq = (qm-qp)/(2*h)
                rem = factor*h*h/(16*(b-h)**2.5)
                exact = factor/(2*np.sqrt(b))
                assert np.all(cq-rem <= exact) and np.all(exact <= cq+rem)
                lo, hi = np.quantile(cq-rem, .025), np.quantile(cq+rem, .975)
                close([lo, hi], [float(r['paired_lower_F']), float(r['paired_upper_F'])])
                median_b = np.median(vbi)-float(r['V_app_V'])
                median_f = np.sqrt(2*EPS*Q*np.median(k))
                cross_rem = median_f*h*h/(16*(median_b-h)**2.5)
                c_lo = (np.quantile(qm, .025)-np.quantile(qp, .975))/(2*h)
                c_hi = (np.quantile(qm, .975)-np.quantile(qp, .025))/(2*h)
                close([c_lo, c_hi, c_lo-cross_rem, c_hi+cross_rem],
                      [float(r[x]) for x in ['cross_lower_without_remainder_F', 'cross_upper_without_remainder_F', 'cross_lower_F', 'cross_upper_F']])
                close([np.min(b-h), np.min(transition-b-h)],
                      [float(r['minimum_flatband_margin_V']), float(r['minimum_transition_margin_secant_V'])])
                independent_checks += 1
                sample_checks += len(vbi)
    assert predictive_difference < 0.001
    comparison = rows('comparison_summary.csv')
    assert len(comparison) == 4*3*4
    for r in comparison:
        selected = select(points, r['series'], r['evaluation_set'])
        stats = statistics(selected, r['method'])
        assert len(selected) == int(r['n_test_points']) and stats[0] == int(r['inside_count'])
        close(stats[1:], [float(r['coverage']), float(r['mean_relative_halfwidth'])], 2e-14)
        assert sum(float(p[METHODS[r['method']][0]]) < 0 for p in selected) == int(r['negative_lower_bound_count'])
    assert rows('table6_summary.csv') == [r for r in comparison if r['series'] == 'pooled' and r['evaluation_set'] != 'all']
    sensitivity = rows('sensitivity_summary.csv')
    assert len(sensitivity) == 3*4*3
    for r in sensitivity:
        selected = select(points, r['series'], r['evaluation_set'], float(r['h_V']))
        assert int(r['n_test_points']) == len(selected)
        for method, prefix in [('cross_endpoint', 'cross'), ('paired_secant', 'paired'), ('credible', 'credible'), ('predictive', 'predictive')]:
            stats = statistics(selected, method)
            close(stats[2], float(r[prefix+'_mean_relative_halfwidth']), 2e-14)
            if prefix in ('cross', 'paired'):
                close(stats[1], float(r[prefix+'_coverage']), 2e-14)
        close(max(float(p['paired_excess_relative_width']) for p in selected), float(r['max_paired_excess_relative_width']))
    assert rows('table7_summary.csv') == [r for r in sensitivity if r['evaluation_set'] == 'all']
    for key in original:
        selected = sorted([p for p in points if (p['series'], p['observation_id']) == key], key=lambda p: -float(p['h_V']))
        assert np.all(np.diff([float(p['paired_excess_relative_width']) for p in selected]) < 0)
    for r in rows('cross_endpoint_summary.csv'):
        selected = select(points, r['series'], 'all')
        stats = statistics(selected, 'cross_endpoint')
        assert int(r['n_test_points']) == len(selected) and int(r['inside_with_remainder']) == stats[0]
        close([float(r['coverage_with_remainder']), float(r['mean_relative_halfwidth'])], stats[1:])
        without = sum(float(p['cross_lower_without_remainder_F']) <= float(p['C_observed_F']) <= float(p['cross_upper_without_remainder_F']) for p in selected)
        assert int(r['inside_without_remainder']) == without
        close(float(r['mean_relative_halfwidth_without_remainder']), np.mean([(float(p['cross_upper_without_remainder_F'])-float(p['cross_lower_without_remainder_F']))/(2*float(p['C_median_F'])) for p in selected]))

    energy = rows('energy_benchmark.csv')
    assert len(energy) == len(original)*len(STEPS)*2 == 738
    assert len({(r['series'], r['observation_id'], r['h_V'], r['delta_refinement']) for r in energy}) == len(energy)
    for r in energy:
        area, k, b = float(r['area_cm2']), float(r['k_NA2']), float(r['b_V'])
        h, delta = float(r['h_V']), float(r['delta_V'])
        density = k/area**2
        width = np.sqrt(2*EPS*b/(Q*density))
        # Integrate eps*(u')^2/2 + q*N*u with u=q*N*(W-x)^2/(2*eps).
        integral = area*(Q*density)**2*width**3/(3*EPS)
        close(integral, float(r['exact_total_energy_J']), 2e-14)
        assert float(r['energy_gap_J']) == 0
        assert (r['series'], r['observation_id']) in original
        assert float(r['Q_lower_C']) <= float(r['exact_Q_C']) <= float(r['Q_upper_C'])
        assert float(r['mean_C_lower_F']) <= float(r['exact_mean_C_F']) <= float(r['mean_C_upper_F'])
        assert float(r['C_lower_F']) <= float(r['exact_C_F']) <= float(r['C_upper_F'])
        total_energy = lambda x: (2/3)*np.sqrt(2*EPS*Q*k)*x**1.5
        close([(total_energy(b)-total_energy(b-delta))/delta, (total_energy(b+delta)-total_energy(b))/delta],
              [float(r['Q_lower_C']), float(r['Q_upper_C'])], 2e-10)
        close(delta, h*h/(50*int(r['delta_refinement'])))
        assert float(r['minimum_domain_margin_V']) > 0
    for key in original:
        selected = [r for r in energy if (r['series'], r['observation_id']) == key]
        for h in STEPS:
            coarse, fine = sorted([r for r in selected if float(r['h_V']) == h], key=lambda r: int(r['delta_refinement']))
            assert float(fine['C_lower_F']) >= float(coarse['C_lower_F'])
            assert float(fine['C_upper_F']) <= float(coarse['C_upper_F'])
        for refinement in (1, 4):
            ordered = sorted([r for r in selected if int(r['delta_refinement']) == refinement], key=lambda r: -float(r['h_V']))
            assert np.all(np.diff([float(r['relative_halfwidth']) for r in ordered]) < 0)
    result = json.loads((D/'results.json').read_text())
    assert result['sample_point_step_checks'] == sample_checks == 73800000
    assert result['point_count'] == len(original) and result['fit_count'] == 43 and result['out_of_window_count'] == 80
    assert result['energy_benchmark']['test_count'] == len(energy)
    print(json.dumps({'secant_bridge_passed': True, 'unique_evaluation_points': len(original),
                      'primary_fit_points': 43, 'nonfitting_points': 80,
                      'independent_full_draw_point_step_checks': independent_checks,
                      'sample_point_step_checks': sample_checks,
                      'analytic_predictive_inclusion_agreements': len(original),
                      'maximum_MC_predictive_endpoint_relative_difference': predictive_difference,
                      'energy_integral_and_enclosure_checks': len(energy)}, indent=2))


if __name__ == '__main__':
    main()
