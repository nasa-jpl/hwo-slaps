"""Historical serialization support remains an exact revision allow-list."""
import pytest
from hwoslaps.modeling.nonlinear.profile_replay import validate_historical_identity_schema

@pytest.mark.parametrize('revision', [
    'a155b2a6b519a28e99cb7c6df3736f16b415cc78',
    'fe3819cc03386630cdd7cb3350bafc52de7e2108',
])
def test_verified_source_revisions_admitted(revision):
    validate_historical_identity_schema('a155b2a6-clumpy-null', revision)

@pytest.mark.parametrize('schema,revision', [
    ('a155b2a6-clumpy-null', 'fe3819cc'),
    ('a155b2a6-clumpy-null', 'unknown'),
    ('unverified-schema', 'fe3819cc03386630cdd7cb3350bafc52de7e2108'),
    ('consistent_sampling_v2', 'fe3819cc03386630cdd7cb3350bafc52de7e2108'),
])
def test_unverified_or_new_objective_schema_rejected(schema, revision):
    with pytest.raises(ValueError, match='Unsupported historical'):
        validate_historical_identity_schema(schema, revision)
