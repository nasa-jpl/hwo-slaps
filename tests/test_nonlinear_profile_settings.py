"""Public nonlinear settings round-trip contract."""
import pytest
from hwoslaps.modeling.nonlinear.profile_settings import FreshProfileSettings


@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_profile_settings_round_trip_preserves_posterior_start_separation(sigma):
    """Reloading saved settings must not change the admitted start population."""
    settings = FreshProfileSettings(start_separation_posterior_sigma=sigma)
    restored = FreshProfileSettings.from_mapping(settings.to_dict())
    assert restored == settings
