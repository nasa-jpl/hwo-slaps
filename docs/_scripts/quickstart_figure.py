"""Make docs/_static/quickstart.png from the quickstart forecast.

Run from the repository root in the scientific environment:

    python docs/_scripts/quickstart_figure.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hwoslaps import forecast, load_config, mass_reach, prepare_forecast, summarize
from hwoslaps.plotting import plot_mass_curve, plot_statistic_map

config = load_config("configs/minimal.yaml")
with prepare_forecast(config) as prepared:
    result = forecast(prepared, masses_msun=[1e6, 3e6, 1e7, 3e7, 1e8])
summary = summarize(result, q_threshold=10.0)
reach = mass_reach(summary, quantity="q_max", target=10.0, interpolation="log")

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
plot_statistic_map(result, "q_asimov", mass_index=2, ax=left)
fig.colorbar(left.images[0], ax=left, label="q")
left.set_title("q for a 10$^7$ M$_\\odot$ NFW subhalo")
plot_mass_curve(summary, "q_max", reach=reach, ax=right)
right.set_yscale("log")
right.set_title("Largest q at each mass")
fig.savefig("docs/_static/quickstart.png", dpi=150)
