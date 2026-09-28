"""A prepared family embeds the detector register it was prepared under, and refuses another."""
from __future__ import annotations

import json

import pytest

from test_preparation import _family
from pilot_proxy.characterization import stability
from pilot_proxy.config.project import default_project


def test_prepared_family_embeds_the_register():
    register = default_project().register
    family = _family()
    assert family.policy_json == register.canonical_json()
    assert family.policy_sha256 == register.sha256()
    assert json.loads(family.policy_json)["schema"] == register.schema


def test_a_family_under_another_register_is_refused():
    other = json.loads(default_project().register.canonical_json())
    other["schema"] = "another-register"
    with pytest.raises(ValueError):
        _family(policy_json=json.dumps(other))


def test_the_operational_label_needs_every_register_decision():
    assert stability.OPERATIONAL_REQUIRED_IDS
    assert all(i.startswith(stability.OPERATIONAL_PREFIXES) for i in stability.OPERATIONAL_REQUIRED_IDS)
