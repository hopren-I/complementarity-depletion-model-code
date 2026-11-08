#!/usr/bin/env python3
"""Run the numerical experiments and regenerate their tables and figures."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CALCULATIONS = [
    ('verify_digitization_source.py', []),
    ('check_digitization.py', []),
    ('audit_n1_sampling.py', []),
    ('calibrate_baseline.py', []),
    ('export_table2.py', []),
    ('recalculate_table3.py', []),
    ('refresh_provenance.py', ['primary']),
    ('recalculate_window_results.py', []),
    ('recalculate_coverage.py', []),
    ('recalculate_secant_bridge.py', []),
    ('run_validation.py', []),
    ('insb_analyze.py', []),
    ('insb_extended.py', []),
    ('build_insb_table.py', []),
    ('refresh_provenance.py', ['completed']),
    ('compare_reference.py', []),
    ('generate_current_tables.py', []),
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_inputs():
    records = json.loads((ROOT / 'verification/raw_inputs.json').read_text())
    for relative, expected in records.items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError('Input checksum mismatch: ' + relative)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'figures', 'verify'], default='full',
                        help='full: recalculate; figures/verify: use a completed run')
    parser.add_argument('--skip-figures', action='store_true',
                        help='recalculate and verify without rendering figures')
    args = parser.parse_args()
    if sys.version_info < (3, 12):
        parser.error('Python 3.12 or newer is required.')
    if args.skip_figures and args.mode != 'full':
        parser.error('--skip-figures applies only to --mode full.')
    check_inputs()
    if args.mode != 'full' and not (ROOT / 'data/insb_artifact_manifest.json').exists():
        parser.error('Run python reproduce.py once before using figures or verify mode.')
    for relative in ['.work/tmp', '.work/mpl', '.work/cache', 'qa', 'tables',
                     'figure/en', 'figure/zh', 'data/results', 'data/validation/baseline']:
        (ROOT / relative).mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
                      MPLCONFIGDIR=str(ROOT / '.work/mpl'),
                      TMPDIR=str(ROOT / '.work/tmp'),
                      XDG_CACHE_HOME=str(ROOT / '.work/cache'))
    steps = list(CALCULATIONS) if args.mode == 'full' else []
    if args.mode in ['full', 'figures'] and not args.skip_figures:
        steps.append(('build_figures.py', []))
    if args.mode == 'verify':
        steps.append(('compare_reference.py', []))
    steps.append(('verify_results.py', []))
    logs = ROOT / 'verification/logs' / args.mode
    logs.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'mode': args.mode, 'python': platform.python_version(),
              'platform': platform.platform(), 'packages': {}, 'steps': []}
    for name in ['numpy', 'scipy', 'matplotlib', 'pandas', 'Pillow', 'pypdf', 'pypdfium2']:
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][name] = 'not installed'
    status = ROOT / 'verification' / (args.mode + '_run.json')
    for index, (script, arguments) in enumerate(steps, 1):
        print(f'[{index}/{len(steps)}] {script}', flush=True)
        started = time.monotonic()
        log = logs / f'{index:02d}_{Path(script).stem}.log'
        with log.open('w') as output:
            result = subprocess.run([sys.executable, str(ROOT / 'code' / script), *arguments],
                                    cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        report['steps'].append({'script': script, 'arguments': arguments,
                                'exit_code': result.returncode,
                                'seconds': round(time.monotonic() - started, 3),
                                'log': str(log.relative_to(ROOT))})
        check_inputs()
        report['passed'] = result.returncode == 0 and index == len(steps)
        report['inputs_unchanged'] = True
        status.write_text(json.dumps(report, indent=2) + '\n')
        if result.returncode:
            print(log.read_text()[-12000:])
            raise SystemExit('Failed: ' + script)
    print('Completed: ' + args.mode)

if __name__ == '__main__':
    main()
