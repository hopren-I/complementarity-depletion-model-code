"""Compare newly calculated outputs with immutable baseline CSVs."""
from pathlib import Path
import csv,hashlib,json,math
import numpy as np
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    sources=json.loads((P/'reference_results/source_hashes.json').read_text())
    results=[];ignored={'search_seconds','reproduces_previous'}
    for rel,expected in sources.items():
        ref=P/'reference_results'/rel;out=P/'data'/rel
        assert sha(ref)==expected, 'Reference altered: '+rel
        a=list(csv.DictReader(ref.open()));b=list(csv.DictReader(out.open()))
        assert len(a)==len(b),(rel,len(a),len(b))
        assert set(a[0])-ignored==set(b[0])-ignored,rel
        count=0;max_relative=0.
        for index,(x,y) in enumerate(zip(a,b)):
            for key in set(x)-ignored:
                try:av,bv=float(x[key]),float(y[key])
                except ValueError:
                    assert x[key]==y[key],(rel,index,key,x[key],y[key]);continue
                if math.isnan(av) and math.isnan(bv):continue
                np.testing.assert_allclose(bv,av,rtol=1e-8,atol=1e-20,err_msg=f'{rel}:{index}:{key}')
                if math.isfinite(av) and abs(av)>1e-20:max_relative=max(max_relative,abs(bv-av)/abs(av))
                count+=1
        results.append({'file':rel,'rows':len(a),'numeric_cells':count,'max_relative_difference':max_relative})
    report={'passed':True,'relative_tolerance':1e-8,'absolute_tolerance':1e-20,'excluded_non_scientific_columns':sorted(ignored),'files':results}
    (P/'verification/reference_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
