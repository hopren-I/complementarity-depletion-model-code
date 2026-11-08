"""Re-extract GaN digitization in a temporary project; preserve input bytes."""
from pathlib import Path
import csv,json,shutil,subprocess,sys,tempfile
import numpy as np
P=Path(__file__).resolve().parents[1]

def main():
    with tempfile.TemporaryDirectory(prefix='ecs-digitization-') as temp:
        root=Path(temp);(root/'code').mkdir()
        raw=json.loads((P/'verification/raw_inputs.json').read_text())
        for rel in raw:
            if rel=='data/insb_source_data.zip':continue
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/rel,target)
        shutil.copy2(P/'code/prepare_table3_data.py',root/'code/prepare_table3_data.py')
        subprocess.run([sys.executable,str(root/'code/prepare_table3_data.py')],cwd=root,check=True)
        compared=[]
        for rel in ['data/all_marker_details.csv','data/digitization/reading_sensitivity.csv']+[
            f'data/gan_{s}_{suffix}.csv' for s in ['n1_500nm','n1_750nm','n2_750nm']
            for suffix in ['cv','primary_cv','marker_details']]:
            a=list(csv.DictReader((P/rel).open()));b=list(csv.DictReader((root/rel).open()))
            assert len(a)==len(b),rel
            for x,y in zip(a,b):
                assert x.keys()==y.keys(),rel
                for key in x:
                    try:av,bv=float(x[key]),float(y[key])
                    except ValueError:assert x[key]==y[key],(rel,key);continue
                    np.testing.assert_allclose(av,bv,rtol=1e-12,atol=1e-24,err_msg=rel+':'+key)
            compared.append({'file':rel,'rows':len(a)})
    report={'passed':True,'scope':'independent extraction from packaged Figure 4 raster; fixed axis calibration; numeric CSV equality','files':compared}
    (P/'verification/digitization.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
