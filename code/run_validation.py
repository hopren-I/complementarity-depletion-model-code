"""Full deterministic bound tests and 3 x 1000 synthetic replications."""
from pathlib import Path
import json,shutil,platform
import numpy as np
import minimal_validation as mv
import numerical_bounds as nb
import validation_analysis as va
P=Path(__file__).resolve().parents[1]

def main():
    bounds,profiles,bmeta=mv.experiment_bounds()
    bayes,smeta=mv.experiment_bayes(1000,10000)
    for old,new in [('bounds_convergence.csv','bounds_baseline.csv'),('bayes_repetitions.csv','bayes_repetitions.csv'),('bayes_summary.csv','bayes_summary.csv')]:
        shutil.copy2(mv.OUT/old,va.OUT/new)
    # A separate Thomas tridiagonal solve checks the faster recurrence solver.
    cross=[]
    for b in [.2,.49,.499,.5,.501,.8]:
        for n in [32,64,128,256]:
            a=mv.solve_bounds(b,n);c=nb.solve(b,n)
            np.testing.assert_allclose(a[4],c[5],rtol=1e-9,atol=1e-11)
            cross.append({'b':b,'n':n,'max_u_difference':float(np.max(np.abs(a[4]-c[5])))})
    report={'bounds':bmeta,'synthetic_iid':smeta,'independent_solver_comparisons':cross,
            'numerical':va.numerical(),'film':va.film(),'stress':va.stress(),
            'python':platform.python_version(),'numpy':np.__version__}
    (va.OUT/'analysis_report.json').write_text(json.dumps(report,indent=2,default=lambda x:bool(x))+'\n')
    print(json.dumps(report,indent=2,default=lambda x:bool(x)))

if __name__=='__main__':main()
