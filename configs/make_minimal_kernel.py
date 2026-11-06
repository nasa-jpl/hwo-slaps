"""Generate the pixel-integrated illustrative 7 x 7 Gaussian kernel (sigma 0.75 pixel)."""
from pathlib import Path
import hashlib

import numpy as np
from scipy.special import erf

bins = np.diff(0.5 * erf(np.arange(-3.5, 4.0) / (np.sqrt(2.0) * 0.75)))
kernel = np.outer(bins, bins)
kernel /= kernel.sum()
path = Path(__file__).with_name("minimal_kernel.npy")
np.save(path, kernel)
print(hashlib.sha256(path.read_bytes()).hexdigest())
