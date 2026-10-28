"""High-precision halo oracle and branch derivative receipt (run on science host)."""
import argparse
import json
from pathlib import Path
import numpy as np
import mpmath as mp
from hwoslaps.scene.halo_numerics import spherical_deflections


def reference(x, tau):
    with mp.workdps(100):
        x = mp.mpf(str(x))
        f = (mp.acosh(1/x)/mp.sqrt(1-x*x) if x < 1 else
             mp.acos(1/x)/mp.sqrt(x*x-1) if x > 1 else mp.mpf(1))
        if tau is None:
            projected = mp.log(x/2)+f
        else:
            t = mp.mpf(str(tau))
            root = mp.sqrt(x*x+t*t)
            projected = t*t/(t*t+1)**2*((t*t+2*x*x-1)*f+mp.pi*t+(t*t-1)*mp.log(t)
                +root*((t*t-1)/t*mp.log(x/(root+t))-mp.pi))
        return float(mp.mpf('.2')*projected/x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows=[]
    for tau in (None, .001, .01, .1, 1., 3., 40., 1e3):
        kind = 'NFW' if tau is None else 'TNFW'
        p = {'kappa_s':.05, 'scale_radius':1.}
        if tau is not None:
            p['truncation_radius'] = tau
        switch = .1*min(1., tau or 1.)
        radii = np.unique(np.r_[np.logspace(-16, 5, 100), switch*np.array([1-1e-8, 1, 1+1e-8]),
                                 [.98, 1., 1.02]])
        values = spherical_deflections(kind, np.column_stack([np.zeros_like(radii), radii]), p)[:, 1]
        for x, value in zip(radii, values):
            oracle = reference(x, tau)
            rows.append(dict(kind=kind, tau=tau, radius=float(x), value=float(value), oracle=oracle,
                             relative_error=float(value/oracle-1)))
    payload = {'method':'100-digit NFW / BMO n=1 closed forms', 'rows':rows,
               'max_relative_error':max(abs(r['relative_error']) for r in rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cases':len(rows),'max_relative_error':payload['max_relative_error']}))


if __name__ == '__main__':
    main()
