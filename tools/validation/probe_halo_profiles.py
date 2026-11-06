"""Direct baseline/corrected halo profile receipt, including true zero radius."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True,exist_ok=True)
os.chdir(args.output.parent)
from autoconf import conf
conf.instance.push(str(Path(importlib.util.find_spec('autoarray').origin).resolve().parent/'config'),keep_first=True)
import autolens as al
import numpy as np
import hwoslaps.scene.halo_profiles as profiles
rows=[]
for tau in (.1,1.,3.,40.):
    profile = profiles.TruncatedNFWSph(kappa_s=.05,scale_radius=1.,truncation_radius=tau)
    for r in (0.,1e-12,1e-8):
        vector = np.asarray(profile.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=[[0.,r]])))[0]
        rows.append({'kind':'TNFW','tau':tau,'radius':r,
                     'deflection':[float(x) if np.isfinite(x) else str(x) for x in vector]})
profile = getattr(profiles,'NFWSph',al.mp.NFWSph)(kappa_s=.05,scale_radius=1.)
for r in (0.,1e-12,1e-8):
    vector = np.asarray(profile.deflections_yx_2d_from(grid=al.Grid2DIrregular(values=[[0.,r]])))[0]
    rows.append({'kind':'NFW','radius':r,
                 'deflection':[float(x) if np.isfinite(x) else str(x) for x in vector]})
payload={'source':profiles.__file__,'python':sys.version,'numpy':np.__version__,'rows':rows}
args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
print(json.dumps(payload,indent=2,allow_nan=False))
