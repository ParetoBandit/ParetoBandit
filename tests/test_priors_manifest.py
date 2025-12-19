from banditgpt._resources import get_priors_path
from banditgpt.core.prior_downloader import ensure_priors
from banditgpt.core.prior_manifest import load_priors_manifest, validate_file


def test_manifest_matches_bundled_priors():
    manifest = load_priors_manifest()
    assert manifest.schema_version == "1.0"
    names = {f.name for f in manifest.files}
    assert {"shippable_priors.npz", "expert_priors.npz"}.issubset(names)

    for entry in manifest.files:
        path = get_priors_path(entry.name)
        assert path.exists()
        assert path.stat().st_size == entry.size_bytes
        assert validate_file(path, entry.sha256)


def test_ensure_priors_uses_bundled_when_present():
    resolved = ensure_priors()
    assert {"shippable_priors.npz", "expert_priors.npz"}.issubset(set(resolved))
    for path in resolved.values():
        assert path.exists()
