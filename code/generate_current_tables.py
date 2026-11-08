"""Generate Table V fragments from the InSb calculation summaries."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

P = Path(__file__).resolve().parents[1]
CONDITIONS = ['same_2MHz', 'blocked_same_2MHz', 'forward_100000Hz',
              'forward_5000Hz', 'reverse_2e+06Hz', 'same_2MHz', 'reverse_2e+06Hz']
LABELS = {
    'en': ['2 MHz, interleaved holdout', '2 MHz, contiguous holdout', '100 kHz, forward',
           '5 kHz, forward', '2 MHz, reverse, forward calibration',
           '2 MHz, reverse, recalibrated', '2 MHz, forward', '2 MHz, reverse'],
    'zh': ['2 MHz前扫，交错留出', '2 MHz前扫，连续留出', '100 kHz前扫', '5 kHz前扫',
           '2 MHz反扫，沿用前扫标定', '2 MHz反扫，重新标定', '2 MHz前扫', '2 MHz反扫'],
}


def rows():
    with (P / 'data/insb_table8_values.csv').open(newline='') as stream:
        result = list(csv.DictReader(stream))
    assert len(result) == 7 and [r['condition'] for r in result] == CONDITIONS
    assert [int(float(r['n_devices'])) for r in result] == [127] * 5 + [31] * 2
    assert all(float(r['invalid_C_prediction_points']) == 0 for r in result)
    with (P / 'data/insb_target_refit_summary.csv').open(newline='') as stream:
        refits = [r for r in csv.DictReader(stream)
                  if r['condition'] == 'reverse_2e+06Hz'
                  and r['model'] == 'target_refit_same_voltage']
    assert len(refits) == 1
    refit = refits[0]
    assert int(refit['n_devices']) == int(float(result[4]['n_devices'])) == 127
    assert int(refit['n_test_points']) == int(float(result[4]['n_test_points'])) == 3051
    assert int(refit['invalid_C_prediction_points']) == 0
    for key in ['median', 'q25', 'q75']:
        refit['mape_' + key] = refit['mape_valid_C_' + key]
    result.insert(5, refit)
    return result


def table_text(language):
    data = rows()
    zh = language == 'zh'
    header = ['测量条件', '器件数', r'\shortstack{电容误差/\%\\中位数[四分位区间]}'] if zh else [
        'Condition', 'Devices', r'\shortstack{C error (\%)\\median [IQR]}']
    lines = [r'\begingroup\color{black}\small\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{lrr}', r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    for index, row in enumerate(data):
        if index == 6:
            title = '同一双低损耗配对队列' if zh else 'Same paired low-loss cohort'
            lines.extend([r'\midrule', r'\multicolumn{3}{l}{' + title + r'} \\'])
        v = [100 * float(row['mape_' + key]) for key in ['median', 'q25', 'q75']]
        lines.append(f"{LABELS[language][index]} & {int(float(row['n_devices']))} & "
                     f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" + r' \\')
    counts = [str(int(float(r['n_test_points']))) for r in data]
    notes = [('前六行测试点数：' if zh else 'Test points (first six rows): ') + ', '.join(counts[:6]) + '.',
             ('配对两行测试点数：' if zh else 'Paired rows: ') + ', '.join(counts[6:]) + '.',
             '各行未定义电容预测点数均为0；未作截断。' if zh else 'All rows: zero undefined C predictions; no clipping.']
    lines += [r'\midrule'] + [r'\multicolumn{3}{l}{' + note + r'} \\' for note in notes]
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--language', choices=['en', 'zh', 'both'], default='both')
    args = ap.parse_args()
    languages = ['en', 'zh'] if args.language == 'both' else [args.language]
    generated = P / 'tables'
    generated.mkdir(exist_ok=True)
    outputs = {}
    for language in languages:
        path = generated / f'insb_prediction_summary_{language}.tex'
        path.write_text(table_text(language))
        outputs[str(path.relative_to(P))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'generator': 'code/generate_current_tables.py', 'table': 'InSb prediction summary',
              'source': 'data/insb_table8_values.csv', 'source_sha256': hashlib.sha256(
                  (P / 'data/insb_table8_values.csv').read_bytes()).hexdigest(),
              'refit_source': 'data/insb_target_refit_summary.csv', 'refit_source_sha256': hashlib.sha256(
                  (P / 'data/insb_target_refit_summary.csv').read_bytes()).hexdigest(),
              'outputs_sha256': outputs, 'scientific_data_modified': False}
    (P / 'qa').mkdir(exist_ok=True)
    (P / 'qa/current_tables.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
