# Saving and loading

HWO-SLAPS saves three kinds of product. Each file holds the arrays together with the
configuration and provenance that produced them, so a saved result can be read and
checked years later without the original scripts.

| Product | Save | Load | Format |
|---|---|---|---|
| Forecast | `save_forecast(result, path)` | `load_forecast(path)` | NumPy `.npz` |
| Observation | `save_observation(observation, path)` | `load_observation(path)` | NumPy `.npz` |
| Nonlinear case | `save_case(case, path)` | `load_case(path)` | JSON |

```python
from hwoslaps import load_forecast, save_forecast

save_forecast(result, "out/run1/forecast.npz")
again = load_forecast("out/run1/forecast.npz")
```

The save functions never overwrite an existing file; choose a new path for each product.
The load functions check the file's structure and contents, and read no pickled
objects, so loading a file cannot run code.

Loading needs only the core package. You can copy results from a GPU node and analyze
them on a laptop.

## What a forecast file records

`result.config` is the complete configuration, with every default filled in.
`result.provenance` records how the forecast was made:

| Key | Contents |
|---|---|
| `config_digest`, `comparison_digest` | The configuration digests |
| `file_digests` | The SHA-256 of every file the configuration referenced |
| `truth_kernels`, `model_kernels`, `psf_relation` | The PSF kernels and how the model relates to the truth |
| `photometry`, `spectral` | How magnitudes became detected rates, and the chromatic weights |
| `nuisance_names`, `nuisance_rank`, `gram_condition_number` | The profiled parameters and the conditioning of their fit |
| `mask`, `positions`, `noise_covariance` | The pixels and trial positions used |
| `halo_model`, `mass_definition`, `subhalo_redshift`, `cosmology` | The subhalo model |
| `sampling` | The pixel sampling diagnostic |
| `engine` | The engine, its device and its worker settings |
| `hwoslaps_version` | The package version |

A `nuisance_rank` smaller than the number of nuisance names means some nuisance
directions were indistinguishable and were dropped from the fit.

## Command-line runs

`hwoslaps forecast` and `hwoslaps simulate` write their product with three companions:

`effective_config.yaml`
: The complete configuration. Pass it back to `hwoslaps forecast` to repeat the run
  exactly.

`provenance.json`
: The command, the start time, the configuration digest, the Python and package
  versions, the git revision of the source and the thread settings.

`run.log`
: The full log, including debug messages.

## Image sources

An `Image` light component uses a pixelized galaxy image as the source. Prepare the
image once and save it as an asset file:

```python
from hwoslaps.artifacts import save_image_asset
from hwoslaps.scene import prepare_image_asset

asset = prepare_image_asset(galaxy_image, half_light_radius_arcsec=0.1)
save_image_asset(asset, "source_asset.npz")
```

`prepare_image_asset` subtracts the background measured at the image border, keeps the
galaxy's main footprint, centres it on its flux centroid, sets its angular scale and
normalizes it to unit integral. Row 0 of the input array is its bottom row.

The configuration then refers to the file:

```yaml
scene:
  source:
    light:
      galaxy: {type: Image, asset_path: source_asset.npz, centre: [0.0, 0.05],
               flux: {ab_mag: 25.0}, sed: {kind: flat_fnu}}
```

The image's position, brightness (`flux_scale`), size (`size_scale`) and rotation
(`rotation_deg`) are profiled as nuisance parameters; its pixel values are fixed.
