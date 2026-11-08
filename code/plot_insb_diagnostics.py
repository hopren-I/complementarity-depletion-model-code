"""Reproduce bilingual Figure 8 without changing manuscript text or tables.

Read the existing archive and result CSVs; reproduce the original representative
device, regression, cohort box plots and 31-device paired comparison. Only the
plot style and explanatory labels change. Outputs are PNG/SVG/PDF plus provenance.
Run in the project's NumPy/Matplotlib environment. The default output directory
is paper/figs. Use --output-dir for previews and --language en or zh for one
language. All inputs are read from the project's archived data directory.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / 'data'
sys.path.insert(0, str(PROJECT / 'code'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullFormatter
from insb_io import load_devices
from insb_analyze import window_indices
from insb_stats import fit_line, predict

COLORS = ['#256D89', '#BA812B', '#BB5543', '#785C9D']
INK = '#273741'
MUTED = '#63717A'
CONDITIONS = ['same_2MHz', 'forward_100000Hz', 'forward_5000Hz', 'reverse_2e+06Hz']
INPUTS = ['insb_source_data.zip', 'insb_reference_fits.csv', 'insb_device_scores.csv',
          'insb_paired_low_loss_scores.csv', 'insb_prediction_points.csv', 'insb_table8_values.csv']


def rows(name):
    with (DATA / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def load_plot_data():
    fits = sorted((r for r in rows('insb_reference_fits.csv') if r['window'] == 'low_04_08'),
                  key=lambda r: (float(r['r2']), r['device']))
    assert len(fits) == 127
    representative = fits[len(fits) // 2]
    assert representative['device'] == 'HCl-Water 200, Die 57 LM4'
    devices, _ = load_devices()
    device = next(d for d in devices if d['device'] == representative['device'])
    v, cref = device['v'], float(representative['Cref_F'])
    jref = int(np.flatnonzero(device['freq'] == 2e6)[0])
    c = device['c'][:, jref]
    ix = window_indices(v, c, cref, .04, .08)
    train = ix[np.arange(len(ix)) % 3 != 2]
    test = ix[np.arange(len(ix)) % 3 == 2]
    fit = fit_line(v[train], (cref / c[train]) ** 2)
    np.testing.assert_allclose(fit['coef'], [float(representative['a']), float(representative['b'])], rtol=1e-12)
    np.testing.assert_allclose(fit['r2'], float(representative['r2']), rtol=1e-12)
    grid = np.linspace(v[ix[0]], v[ix[-1]], 200)
    mu, lo, hi = predict(fit, grid)
    curves = []
    for frequency in [2e6, 1e5, 5e3]:
        j = int(np.flatnonzero(device['freq'] == frequency)[0])
        curves.append((v, device['c'][:, j] / cref))
    rv, rc = device['reverse'][2e6]
    curves.append((rv, rc / cref))
    transformed_test = [(cref / c[test]) ** 2]
    for frequency in [1e5, 5e3]:
        j = int(np.flatnonzero(device['freq'] == frequency)[0])
        transformed_test.append((cref / device['c'][test, j]) ** 2)
    transformed_test.append((cref / np.interp(v[test], rv, rc)) ** 2)
    archived = rows('insb_prediction_points.csv')
    for condition, values in zip(CONDITIONS, transformed_test):
        selected = sorted((r for r in archived if r['window'] == 'low_04_08'
                           and r['device'] == representative['device'] and r['condition'] == condition),
                          key=lambda r: float(r['V']))
        np.testing.assert_allclose([float(r['V']) for r in selected], v[test], atol=1e-12)
        np.testing.assert_allclose([float(r['y']) for r in selected], values, rtol=1e-12)
        test_mu, test_lo, test_hi = predict(fit, v[test])
        for key, actual in [('mu_y', test_mu), ('lower_y', test_lo), ('upper_y', test_hi)]:
            np.testing.assert_allclose([float(r[key]) for r in selected], actual, rtol=1e-11)
    scores = rows('insb_device_scores.csv')
    values, cohorts = [], []
    for condition in CONDITIONS:
        selected = sorted((r for r in scores if r['window'] == 'low_04_08'
                           and r['condition'] == condition), key=lambda r: r['device'])
        assert len(selected) == 127
        cohorts.append([r['device'] for r in selected])
        values.append(np.array([float(r['mape']) * 100 for r in selected]))
    assert all(c == cohorts[0] for c in cohorts)
    low = rows('insb_paired_low_loss_scores.csv')
    paired = []
    for condition in [CONDITIONS[0], CONDITIONS[-1]]:
        selected = sorted((r for r in low if r['condition'] == condition), key=lambda r: r['device'])
        assert len(selected) == 31
        paired.append(selected)
    assert [r['device'] for r in paired[0]] == [r['device'] for r in paired[1]]
    a, b = [np.array([float(r['mape']) * 100 for r in selected]) for selected in paired]
    table = rows('insb_table8_values.csv')
    for condition, group in zip(CONDITIONS, values):
        expected = next(r for r in table if r['cohort'] == 'all eligible' and r['condition'] == condition)
        np.testing.assert_allclose(np.quantile(group, [.25, .5, .75]),
                                   [100 * float(expected[k]) for k in ['mape_q25', 'mape_median', 'mape_q75']], rtol=1e-11)
    np.testing.assert_allclose([np.median(a), np.median(b)], [1.98080464898085, 4.6184088584506], rtol=1e-11)
    return dict(representative=representative, cref=cref, curves=curves,
                train_v=v[train], train_y=(cref / c[train]) ** 2,
                test_v=v[test], test_y=transformed_test, window=[v[ix[0]], v[ix[-1]]],
                grid=grid, mu=mu, lower=lo, upper=hi, r2=fit['r2'],
                values=values, cohort=cohorts[0], paired_devices=[r['device'] for r in paired[0]],
                paired_forward=a, paired_reverse=b)


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', which='major', color='#E7EBEE', linewidth=.7)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#8B969D')
        ax.spines[side].set_linewidth(.8)
    ax.tick_params(length=3, width=.75, labelsize=9, color='#8B969D')


def draw(d, lang, output_dir):
    zh = lang == 'zh'
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': ['FandolHei', 'DejaVu Sans'] if zh else ['DejaVu Sans'],
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10.5,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'mathtext.fontset': 'dejavusans', 'svg.fonttype': 'path',
        'svg.hashsalt': 'figure8-style-preview', 'figure.facecolor': 'white',
    })
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.1))
    fig.subplots_adjust(left=.083, right=.985, bottom=.097, top=.865, wspace=.27, hspace=.46)
    for ax in axs.flat:
        style_axes(ax)
    titles = (['(a) 同一器件的四种测量条件', '(b) 参考拟合与跨条件预测',
               '(c) 全部127个器件的预测误差', '(d) 同一31个低损耗器件的配对比较'] if zh else
              ['(a) One device, four conditions', '(b) Reference fit and transferred predictions',
               '(c) Prediction errors: all 127 devices', '(d) Paired errors: 31 low-loss devices'])
    for ax, title in zip(axs.flat, titles):
        ax.set_title(title, loc='left', pad=12)
    global_labels = (['2 MHz前扫（参考）', '100 kHz前扫', '5 kHz前扫', '2 MHz反扫'] if zh else
                     ['2 MHz forward (reference)', '100 kHz forward', '5 kHz forward', '2 MHz reverse'])
    handles = [Line2D([], [], color=color, lw=1.8) for color in COLORS]
    fig.legend(handles, global_labels, ncol=4, frameon=False, loc='upper center',
               bbox_to_anchor=(.535, .993), fontsize=9.3, handlelength=2.0,
               handletextpad=.5, columnspacing=1.5)

    ax = axs[0, 0]
    ax.axvspan(*d['window'], facecolor='#E8EFF3', edgecolor='none', zorder=0)
    for (v, c), color in zip(d['curves'], COLORS):
        ax.plot(v, c, color=color, linewidth=1.7)
    ax.set(xlim=(-5, 3), ylim=(.015, 1.45), yscale='log')
    ax.set_xticks([-4, -2, 0, 2])
    ax.set_yticks([.02, .05, .1, .5, 1], ['0.02', '0.05', '0.1', '0.5', '1'])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel('栅压（V）' if zh else 'Gate voltage (V)')
    ax.set_ylabel(r'$C/C_{\mathrm{ref}}$' + ('（对数刻度）' if zh else ' (log scale)'))
    ax.text(np.mean(d['window']), 1.22, '参考窗口' if zh else 'Fit window',
            ha='center', va='center', fontsize=8.6, color=COLORS[0])
    ax.text(.975, .055, d['representative']['device'], transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.8, color=MUTED)

    ax = axs[0, 1]
    ax.fill_between(d['grid'], d['lower'], d['upper'], color=COLORS[0], alpha=.15,
                    edgecolor='none', zorder=1)
    ax.plot(d['grid'], d['mu'], color=COLORS[0], lw=1.65, zorder=2)
    ax.scatter(d['train_v'], d['train_y'], s=12, color=COLORS[0],
               edgecolor='white', linewidth=.3, zorder=3)
    ax.scatter(d['test_v'], d['test_y'][0], s=24, facecolor='white',
               edgecolor=COLORS[0], linewidth=1.0, zorder=4)
    for values, color, marker in zip(d['test_y'][1:], COLORS[1:], ['^', 'D', 's']):
        ax.scatter(d['test_v'], values, s=20, color=color, marker=marker,
                   edgecolor='white', linewidth=.3, zorder=3)
    ax.set(xlim=(d['window'][0]-.04, d['window'][1]+.04), ylim=(-28, 755))
    ax.set_xlabel('栅压（V）' if zh else 'Gate voltage (V)')
    ax.set_ylabel(r'$(C_{\mathrm{ref}}/C)^2$')
    ax.text(.975, .985, rf'$R^2={d["r2"]:.4f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=9.5)
    h = [Line2D([], [], marker='o', ms=3.8, linestyle='none', color=COLORS[0]),
         Line2D([], [], marker='o', ms=4.3, linestyle='none', markerfacecolor='white', color=COLORS[0]),
         Line2D([], [], color=COLORS[0], lw=1.65),
         Patch(facecolor=matplotlib.colors.to_rgba(COLORS[0], .15), edgecolor='none')]
    labels = (['训练点', '留出点', '拟合线', '95%预测区间'] if zh else
              ['Training', 'Holdout', 'Fit', '95% prediction band'])
    legend = ax.legend(h, labels, loc='upper right', bbox_to_anchor=(.995, .875),
                       ncol=2, fontsize=7.5, handlelength=1.5, handletextpad=.5,
                       frameon=True, borderpad=.5, labelspacing=.35, columnspacing=1.0)
    legend.get_frame().set(facecolor='white', edgecolor='#E3E8EC', linewidth=.6, alpha=.96)

    tick_labels = (['同次前扫\n2 MHz', '前扫\n100 kHz', '前扫\n5 kHz', '反扫\n2 MHz'] if zh else
                   ['Same forward\n2 MHz', 'Forward\n100 kHz', 'Forward\n5 kHz', 'Reverse\n2 MHz'])
    ax = axs[1, 0]
    bp = ax.boxplot(d['values'], tick_labels=tick_labels, showfliers=False,
                    patch_artist=True, widths=.45, medianprops=dict(color=INK, linewidth=1.7),
                    whiskerprops=dict(linewidth=1.05), capprops=dict(linewidth=1.05))
    for i, (box, color, values) in enumerate(zip(bp['boxes'], COLORS, d['values'])):
        box.set(facecolor=matplotlib.colors.to_rgba(color, .23), edgecolor=color, linewidth=1.3)
        for artist in bp['whiskers'][2*i:2*i+2] + bp['caps'][2*i:2*i+2]:
            artist.set_color(color)
        ax.text(i+1, .965, f'{np.median(values):.2f}%', transform=ax.get_xaxis_transform(),
                color=color, ha='center', va='top', weight='bold', fontsize=10)
    ax.text(.98, .045, '箱体：中间50%；横线：中位数' if zh else 'Box: middle 50%; line: median',
            transform=ax.transAxes, ha='right', va='bottom', color=MUTED, fontsize=7.8)

    ax = axs[1, 1]
    a, b = d['paired_forward'], d['paired_reverse']
    for aa, bb in zip(a, b):
        ax.plot([0, 1], [aa, bb], color='#ADB8BF', alpha=.45, linewidth=.65, zorder=1)
    for x, y, color in [(0, a, COLORS[0]), (1, b, COLORS[3])]:
        ax.scatter(np.full(len(y), x), y, s=20, color=color, alpha=.75,
                   edgecolor='white', linewidth=.35, zorder=3)
    med = [np.median(a), np.median(b)]
    ax.plot([0, 1], med, color=INK, linewidth=1.8, marker='D', markersize=6, zorder=5)
    for x, value, shift, align in [(0, med[0], -.08, 'right'), (1, med[1], .08, 'left')]:
        ax.text(x+shift, value, f'{value:.2f}%', ha=align, va='center', fontsize=10,
                weight='bold', color=INK, zorder=6,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.8, alpha=.95))
    h = [Line2D([], [], color='#ADB8BF', lw=.9),
         Line2D([], [], color=INK, lw=1.8, marker='D', ms=4)]
    labels = ['同一器件', '中位数'] if zh else ['Same device', 'Median']
    ax.legend(h, labels, loc='upper left', frameon=False, ncol=2,
              fontsize=8, handlelength=1.8, columnspacing=1.4)
    ax.set_xticks([0, 1], [tick_labels[0], tick_labels[-1]])
    ax.set_xlim(-.39, 1.39)
    for ax in axs[1]:
        ax.set_yscale('log')
        ax.set_ylim(.04, 400)
        ax.set_yticks([.1, 1, 10, 100], ['0.1', '1', '10', '100'])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='minor', length=0)
        ax.tick_params(axis='x', labelsize=8.8, pad=7)
        ax.set_ylabel('电容误差（%，对数刻度）' if zh else 'C error (%, log scale)')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bounds = legend.get_window_extent(renderer).expanded(1.015, 1.04)
    diagnostic_xy = [np.column_stack([d['train_v'], d['train_y']])]
    diagnostic_xy.extend(np.column_stack([d['test_v'], values]) for values in d['test_y'])
    diagnostic_xy.extend(np.column_stack([d['grid'], values])
                         for values in [d['mu'], d['lower'], d['upper']])
    screen_xy = axs[0, 1].transData.transform(np.concatenate(diagnostic_xy))
    hits = [xy for xy in screen_xy if legend_bounds.contains(*xy)]
    assert not hits, (lang, axs[0, 1].transData.inverted().transform(legend_bounds.get_points()).tolist(),
                      axs[0, 1].transData.inverted().transform(hits[:3]).tolist())
    for text in [*fig.texts, *fig.legends[0].get_texts(),
                 *[t for ax in axs.flat for t in [ax.title, ax._left_title, ax.xaxis.label, ax.yaxis.label, *ax.texts]]]:
        if not text.get_text():
            continue
        if not zh:
            assert not any('\u4e00' <= c <= '\u9fff' for c in text.get_text()), text.get_text()
        bbox = text.get_window_extent(renderer)
        assert bbox.x0 >= -1 and bbox.x1 <= fig.bbox.x1+1, text.get_text()
        assert bbox.y0 >= -1 and bbox.y1 <= fig.bbox.y1+1, text.get_text()
    for ext in ['png', 'svg', 'pdf']:
        metadata = ({'Date': None} if ext == 'svg' else
                    {'CreationDate': datetime(2026, 11, 5, tzinfo=timezone.utc),
                     'ModDate': datetime(2026, 11, 5, tzinfo=timezone.utc)} if ext == 'pdf' else None)
        fig.savefig(output_dir/f'fig_insb_r3_{lang}.{ext}', dpi=400,
                    metadata=metadata, facecolor='white')
    plt.close(fig)


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def render_figures(output_dir, languages=('zh', 'en')):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    font_manager.fontManager.addfont(str(PROJECT/'assets/fonts/FandolHei-Regular.otf'))
    d = load_plot_data()
    for lang in languages:
        draw(d, lang, output_dir)
    manifest = {
        'source_sha256': {name: hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in INPUTS},
        'generator': 'code/plot_insb_diagnostics.py',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'statistical_change': 'none',
        'box_definition': '25th-75th percentiles, median; 1.5 IQR whiskers; outlier markers omitted as in original',
        'paired_point_count': 31,
        'cohort_count_per_condition': 127,
        'data': serializable(d),
        'figures_sha256': {
            f'fig_insb_r3_{lang}.{ext}': hashlib.sha256(
                (output_dir/f'fig_insb_r3_{lang}.{ext}').read_bytes()).hexdigest()
            for lang in languages for ext in ['png', 'svg', 'pdf']
        },
    }
    (output_dir/'figure8_plot_data.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(f'Generated Figure 8 ({", ".join(languages)}) in {output_dir}; numerical values match the archived results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'paper/figs')
    parser.add_argument('--language', choices=['en', 'zh', 'both'], default='both')
    args = parser.parse_args()
    render_figures(args.output_dir, ['zh', 'en'] if args.language == 'both' else [args.language])
