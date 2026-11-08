"""Read archived text directly from ZIP; preserve raw archive unmodified."""
from pathlib import Path
from zipfile import ZipFile
import re
import io
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data"
GROUPS = ['Control', 'HCl-IPA 200', 'HCl-IPA 250', 'HCl-Water 200', 'HCl-Water 250']


def matrix(z, name):
    lines = z.read(name).decode('cp1252').splitlines()
    freqs = [float(s) for s in re.findall(r'[CG] at ([\d.E+\-]+)', lines[0])]
    arr = np.loadtxt(io.StringIO('\n'.join(lines[1:])))
    return np.asarray(freqs), arr


def load_devices():
    devices, audit = [], []
    with ZipFile(ROOT / 'insb_source_data.zip') as z:
        names = set(z.namelist())
        for name in sorted(names):
            if '/Raw Data/' not in name or 'Capacitance' not in name or not name.endswith('.txt'):
                continue
            stem = Path(name).name
            group = stem.split(',')[0]
            if group not in GROUPS or not stem.endswith(' Capacitance.txt'):
                audit.append(dict(file=name, status='out_of_primary_scope', detail='additional HR or CVT temperature-series record; outside filename-defined primary cohort'))
                continue
            freq, arr = matrix(z, name)
            assert arr.shape[1] == len(freq) + 1, name
            assert np.all(np.diff(arr[:, 0]) > 0), name
            device = stem.removesuffix(' Capacitance.txt')
            d = dict(device=device, group=group, file=name, freq=freq, v=arr[:,0], c=arr[:,1:],
                     reverse={}, g=None, reverse_g={})
            gname = name.replace('Capacitance.txt', 'Conductance.txt')
            if gname in names:
                gf, ga = matrix(z, gname)
                assert np.array_equal(gf, freq) and np.array_equal(ga[:,0], arr[:,0])
                d['g'] = ga[:,1:]
            hname = name.replace('Capacitance.txt', 'Hysteresis C.txt')
            if hname in names:
                hf, ha = matrix(z, hname)
                if ha.shape[1] != arr.shape[1] or len(ha) < len(arr) or not np.array_equal(ha[:len(arr)], arr):
                    audit.append(dict(file=hname, status='unmatched_hysteresis', detail='forward prefix differs; reverse excluded'))
                else:
                    rev = ha[len(arr):]
                    assert np.all(np.diff(rev[:,0]) < 0), hname
                    for j, f in enumerate(freq):
                        if np.any(rev[:, j+1] != 0):
                            d['reverse'][f] = (rev[::-1,0],rev[::-1,j+1])
                    audit.append(dict(file=hname, status='matched_hysteresis', detail='exact forward prefix; active reverse Hz='+','.join(str(x) for x in d['reverse'])))
                    hgname = hname.replace('Hysteresis C', 'Hysteresis G')
                    if hgname in names and d['g'] is not None:
                        _, hg = matrix(z, hgname)
                        if hg.shape == ha.shape and np.array_equal(hg[:len(arr),1:], d['g']):
                            for j, f in enumerate(freq):
                                if f in d['reverse']:
                                    d['reverse_g'][f] = (hg[len(arr):][::-1,0],hg[len(arr):][::-1,j+1])
            audit.append(dict(file=name, status='primary_device', detail=f'{len(arr)} voltages; {len(freq)} frequencies'))
            devices.append(d)
    return devices, audit


def valid(c):
    # Instrument overflow sentinel is 9.9e37. No arbitrary upper physical C cut.
    return np.isfinite(c) & (c > 0) & (c < 1e30)


def export_raw_csv(devices):
    """Export every primary measured C/G value; invalid values remain explicit."""
    import csv,gzip
    from collections import Counter
    counts = {'forward': 0, 'reverse': 0, 'invalid_C': 0, 'ambiguous_frequency_rows': 0}
    fields = ['device','group','direction','source_value_column_1based','frequency_label_Hz',
              'ambiguous_frequency_label','acquisition_index_0based','V_G_V','C_F','G_S','C_valid','G_finite_nonnegative']
    with gzip.open(ROOT/'insb_raw_measurements.csv.gz','wt',newline='') as out:
        writer=csv.writer(out);writer.writerow(fields)
        for d in devices:
            counts_f=Counter(d['freq'])
            for j,f in enumerate(d['freq']):
                ambiguous=counts_f[f]>1
                for i,v in enumerate(d['v']):
                    c=d['c'][i,j];g=d['g'][i,j] if d['g'] is not None else float('nan')
                    ok=bool(valid(c));gok=bool(np.isfinite(g) and 0<=g<1e30)
                    writer.writerow([d['device'],d['group'],'forward',j+2,f,int(ambiguous),i,v,c,g,int(ok),int(gok)])
                    counts['forward']+=1;counts['invalid_C']+=not ok;counts['ambiguous_frequency_rows']+=ambiguous
            for f,(v,c) in d['reverse'].items():
                j=int(np.flatnonzero(d['freq']==f)[0]);gg=d['reverse_g'].get(f)
                for i in range(len(v)-1,-1,-1):
                    g=gg[1][i] if gg is not None else float('nan');ok=bool(valid(c[i]));gok=bool(np.isfinite(g) and 0<=g<1e30)
                    writer.writerow([d['device'],d['group'],'reverse',j+2,f,0,len(v)-1-i,v[i],c[i],g,int(ok),int(gok)])
                    counts['reverse']+=1;counts['invalid_C']+=not ok
    return counts
