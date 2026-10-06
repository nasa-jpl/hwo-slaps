# HWO Science Engineering Interface v0.1.9

These files preserve the SEI v0.1.9 bytes used by the HWO reference. The upstream package
is `hwo_sci_eng`, by Breann Sitarski (NASA/GSFC) and Jason Tumlinson (STScI), from the
[SEI repository](https://github.com/HWO-GOMAP-Working-Groups/Sci-Eng-Interface).
HWO ETC `syotools` consumes the interface through its camera `set_from_sei` method.
Upstream `METADATA` records `License: CC` and accompanies the data.

| File | Use |
|---|---|
| `EAC1.yaml` | Telescope diameter, segmentation, gaps and focal length |
| `HRI.yaml` | UVIS plate scale, detector read noise, dark current, coatings and surface count |
| `XeLiF_refl.yaml` | Coating curve for 13 reflective surfaces |
| `Teledyne_COSMOS_CMOS_QE.yaml` | UVIS detector quantum efficiency |
| `METADATA` | Upstream package identity and metadata |
| `SHA256SUMS` | Original file and wheel hashes |

The original manifest also names `ProtectedAg_refl.yaml` and
`hwo_sci_eng-0.1.9-py3-none-any.whl`; they are not shipped here. The driver reports
these absent entries and verifies all shipped entries. Any mismatch raises.

SEI v0.1.9 is pre-formulation and its architectures are exploratory. The UVIS filter curves
referenced by HRI are absent from the wheel. The chromatic example assumes a flat filter
transmission of 0.832. Its pinned 500 nm product target is 0.20997331010823617:
XeLiF reflectivity to power 13, interpolated detector QE and the filter. Its dense-integral
flat-fnu band-mean target is 0.206800227875. Example-run measurements remain pending.
