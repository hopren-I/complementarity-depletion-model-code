"""Small, dependency-light regression diagnostics; no physical parameter claims."""
from functools import lru_cache
import math
import numpy as np


@lru_cache(None)
def t975(df):
    """Student-t 97.5% quantile by quadrature and bisection (df >= 3)."""
    assert df >= 3
    nodes, weights = np.polynomial.legendre.leggauss(96)
    lognorm = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - .5 * math.log(df * math.pi)
    lo, hi = 0., 20.
    for _ in range(60):
        mid = (lo + hi) / 2
        x = (nodes + 1) * mid / 2
        mass = mid / 2 * np.sum(weights * np.exp(lognorm - (df + 1) / 2 * np.log1p(x*x/df)))
        if mass < .475:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_line(v, y):
    v, y = np.asarray(v), np.asarray(y)
    X = np.column_stack([np.ones(len(v)), v])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ coef
    df = len(y) - 2
    s2 = float(residual @ residual / df)
    inv = np.linalg.inv(X.T @ X)
    sst = float(np.sum((y - y.mean())**2))
    return dict(coef=coef, inv=inv, s2=s2, df=df, n=len(y),
                r2=1-float(residual@residual)/sst if sst > 0 else float('nan'),
                residual=residual, v=v, y=y)


def predict(fit, v):
    X = np.column_stack([np.ones(len(v)), v])
    mu = X @ fit['coef']
    h = np.einsum('ij,jk,ik->i', X, fit['inv'], X)
    # Exact conditional Student-t predictive interval under iid Gaussian y errors
    # and p(beta,sigma^2) proportional to 1/sigma^2. Empirical inclusion only.
    half = t975(fit['df']) * np.sqrt(fit['s2'] * (1 + h))
    return mu, mu-half, mu+half


def score(fit, v, c, cref):
    y = (cref / c)**2
    mu, lo, hi = predict(fit, v)
    positive = mu > 0
    errors = np.full(len(v), np.nan)
    errors[positive] = np.abs(cref / np.sqrt(mu[positive]) / c[positive] - 1)
    normerr=(1/np.sqrt(mu[positive])-c[positive]/cref)
    return dict(n=len(v),
                valid_C_prediction_count=int(positive.sum()),
                invalid_C_prediction_count=int((~positive).sum()),
                y_rmse_all=float(np.sqrt(np.mean((mu-y)**2))),
                y_relative_rmse_all=float(np.sqrt(np.mean((mu-y)**2))/np.mean(y)),
                y_mae_all=float(np.mean(np.abs(mu-y))),
                mape_valid_C=float(np.nanmean(errors)) if positive.any() else float('nan'),
                interval_lower_nonpositive_count=int(np.sum(lo<=0)),
                inclusion=float(np.mean((y >= lo) & (y <= hi))),
                mape=float(np.nanmean(errors)) if positive.any() else float('nan'),
                nrmse_Cref=float(np.sqrt(np.mean(normerr**2))) if positive.any() else float('nan'),
                invalid_prediction_fraction=float(np.mean(~positive)))
