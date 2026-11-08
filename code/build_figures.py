"""Render all five bilingual figures using calculated data and packaged fonts."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
from PIL import Image,ImageChops
P=Path(__file__).resolve().parents[1]
FIGURES=[('plot_numerical_bounds','fig_numerical_bounds_r2'),('plot_parameter_recovery','fig_parameter_recovery'),('plot_calibration_merged','fig_calibration_merged'),('plot_gan_merged','fig_gan_merged'),('plot_insb_diagnostics','fig_insb_r3')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=P/'.work/figures';out.mkdir(parents=True,exist_ok=True)
    original=json.loads((P/'reference_results/figure_hashes.json').read_text())
    records=[]
    before={str(p.relative_to(P)):sha(p) for p in (P/'data').rglob('*') if p.is_file()}
    for number,(script,stem) in enumerate(FIGURES,1):
        subprocess.run([sys.executable,str(P/'code'/f'{script}.py'),'--language','both','--output-dir',str(out)],cwd=P,check=True)
        for lang in ['en','zh']:
            source=out/f'{stem}_{lang}.png';target=P/f'figure/{lang}/figure{number}.png'
            image=Image.open(source)
            old=Image.open(target) if target.exists() else None
            same_pixels=(image.size==old.size and ImageChops.difference(image.convert('RGB'),old.convert('RGB')).getbbox() is None) if old else None
            # Original layout uses the same scientific data; pixel differences
            # are explicitly recorded rather than hidden by reusing old figures.
            shutil.copy2(source,target)
            records.append({'figure':number,'language':lang,'generator':'code/'+script+'.py','file':str(target.relative_to(P)),'sha256':sha(target),'reference_sha256':original[lang]['figures'][target.name],'pixels_identical_to_previous':same_pixels,'size':image.size})
    after={str(p.relative_to(P)):sha(p) for p in (P/'data').rglob('*') if p.is_file()}
    assert before==after, 'Plotting changed scientific data'
    (P/'verification/figure_build.json').write_text(json.dumps({'passed':True,'scientific_inputs_unchanged':True,'figures':records,'figure1_panel_a':'Regenerated from recalculated numerical bounds using code/bounds_panel.py. No archived figure raster is used.'},indent=2)+'\n')
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
