import argparse
import shutil

import pytest

from banditgpt._resources import get_priors_path
from banditgpt.core.cli import cmd_verify_priors
from banditgpt.core.prior_manifest import (
    PriorIntegrityError,
    load_priors_manifest,
    verify_bundled_prior,
)


def test_verify_bundled_prior_ok():
    manifest = load_priors_manifest()
    path = get_priors_path("shippable_priors.npz")
    verify_bundled_prior(path, manifest=manifest)


def test_verify_bundled_prior_checksum_mismatch(tmp_path):
    manifest = load_priors_manifest()
    original = get_priors_path("shippable_priors.npz")
    target = tmp_path / "shippable_priors.npz"
    shutil.copyfile(original, target)
    # Mutate a value to force checksum failure
    with open(target, "r+b") as fh:
        fh.write(b"corrupt")
    with pytest.raises(PriorIntegrityError):
        verify_bundled_prior(target, manifest=manifest)


def test_cli_verify_priors():
    args = argparse.Namespace(check_user=False, user_priors="")
    assert cmd_verify_priors(args) == 0
