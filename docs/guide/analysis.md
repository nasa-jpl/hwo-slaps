# Summaries and mass reach

A forecast gives *q* for every mass and position. This page turns those maps into
detection maps, fractions, areas and the smallest detectable mass. None of these steps
needs the scientific stack, so they work on saved results on any machine.

```python
from hwoslaps import load_forecast, mass_reach, summarize

result = load_forecast("out/quickstart/forecast.npz")
summary = summarize(result, q_threshold=10.0)
```

## Detections

A position is a **detection** at a given mass when *q* is at or above your threshold.
For the mismatch statistics, the fitted subhalo amplitude must also be positive.
hwoslaps has no default threshold; you always pass one.

```python
detected = result.detections(q_threshold=10.0)   # boolean, shape (masses, positions)
```

`detections` uses `result.detection_metric`: `q_asimov` for a matched PSF and
`q_mismatch` when the model PSF differs from the truth. Pass `metric=` to choose
another, for example `metric="q_spurious"` to count false positives caused by a PSF
error.

## Summaries

`summarize` reduces each mass to a few numbers:

| Field | Meaning |
|---|---|
| `q_max` | The largest *q* over the selected positions |
| `detectable_count` | The number of selected positions that are detections |
| `detectable_fraction` | That number divided by the number of selected positions |
| `detectable_area_arcsec2` | Detections times the grid cell area (grid positions only) |
| `boundary_detectable` | Whether any detection lies on the edge of the position grid |
| `selected_count`, `q_threshold`, `metric` | What the summary was computed from |

If `boundary_detectable` is true, the detectable region reaches the edge of your grid,
and a larger grid could find a larger area. Widen `half_width_arcsec` until it is false
at the masses you care about.

For the mismatch statistics, `q_max` counts a position with a negative fitted
amplitude as zero.

## Apertures

To summarize only the positions near the arc, pass a selection. `aperture_selection`
picks the positions inside a circle:

```python
from hwoslaps.analysis import aperture_selection

near_ring = aperture_selection(result, centre_yx=(0.0, 0.0), radius_arcsec=1.0)
summary = summarize(result, q_threshold=10.0, selection=near_ring)
```

Any boolean array with one entry per position works as a selection.

## Mass reach

The mass reach is the mass at which a summary quantity crosses a target value:

```python
reach = mass_reach(summary, quantity="q_max", target=10.0, interpolation="log")
print(reach.status, reach.mass_msun)
```

`quantity` is `q_max`, `detectable_fraction` or `detectable_area_arcsec2`. hwoslaps
finds the two evaluated masses on either side of the target and interpolates between
them in log mass. `interpolation` sets how the quantity itself is interpolated:

`"log"`
: Interpolate the logarithm of the quantity. Use it for `q_max`, which grows roughly
  as a power of mass. It needs positive values on both sides of the crossing.

`"linear"`
: Interpolate the quantity directly. Use it for fractions and areas, which can be zero.

The result's `status` says what was found:

| `status` | Meaning | `mass_msun` |
|---|---|---|
| `sampled` | One of your masses hits the target exactly | That mass |
| `bracketed` | The target lies between two of your masses | The interpolated mass |
| `below_range` | Even your smallest mass exceeds the target | `None`; the reach is below `upper_mass_msun` |
| `above_range` | No mass reaches the target | `None`; the reach is above `lower_mass_msun` |
| `non_monotonic` | The curve falls somewhere, so the crossing is not unique | `None` |

hwoslaps never extrapolates beyond the masses you evaluated. If you get `below_range` or
`above_range`, add masses on that side and forecast again.

### Refining the reach

`adaptive_mass_reach` finds the crossing by bisection, evaluating new masses until the
bracket is narrower than a tolerance in dex. You supply a function that returns the
quantity at one mass:

```python
from hwoslaps import forecast, load_config, prepare_forecast, summarize
from hwoslaps.analysis import adaptive_mass_reach

with prepare_forecast(load_config("configs/minimal.yaml")) as prepared:
    def q_max(mass):
        result = forecast(prepared, masses_msun=[mass])
        return float(summarize(result, q_threshold=10.0).q_max[0])

    found = adaptive_mass_reach(q_max, lower_mass_msun=1e6, upper_mass_msun=1e8,
                                target=10.0, interpolation="log", tolerance_dex=0.02)

print(found.reach.status, found.reach.mass_msun, len(found.masses_msun))
```

## Plotting

The plotting functions take a result or summary and return Matplotlib axes. Pass
`ax=` to draw into an existing figure.

| Function | Draws |
|---|---|
| `plot_statistic_map(result, statistic, mass_index=...)` | A map of one statistic, such as `"q_asimov"` or `"degradation"`, at one mass |
| `plot_detection_map(result, q_threshold=..., mass_index=...)` | The detections at one mass |
| `plot_mass_curve(summary, quantity, reach=...)` | A summary quantity against mass, with the target and reach |
| `plot_knowledge_error(areas)` | The PSF knowledge-error area ratios against mass |
| `plot_observation(observation, quantity)` | An observation: `"data"`, `"expected"`, `"noise"` or `"snr"` |
| `plot_kernel(psf, log=...)`, `plot_pupil(pupil)` | A PSF kernel or a telescope pupil |

```python
import matplotlib.pyplot as plt
from hwoslaps.plotting import plot_detection_map, plot_statistic_map

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
plot_statistic_map(result, "q_asimov", mass_index=2, ax=left)
fig.colorbar(left.images[0], ax=left, label="q")
plot_detection_map(result, q_threshold=10.0, mass_index=2, ax=right)
fig.savefig("maps.png")
```

Map plots need grid positions. They use the same `(y, x)` arcsecond coordinates as the
result, with *y* increasing upward.
