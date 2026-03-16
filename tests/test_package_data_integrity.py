from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import venv
from pathlib import Path

import pytest


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


@pytest.mark.slow
@pytest.mark.stress
def test_wheel_and_sdist_include_runtime_data(tmp_path: Path):
    """
    Validate pip-install reality:
    - wheel/sdist build succeeds
    - config/data assets are physically present in built artifacts
    - installed package can read required files at runtime
    """
    repo_root = Path(__file__).resolve().parent.parent
    src_pkg_root = repo_root / "src" / "pareto_bandit"
    pytest.importorskip("build")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    # Build fresh wheel + sdist into tmp output dir.
    _run(
        [sys.executable, "-m", "build", "--wheel", "--sdist", "--outdir", str(dist_dir)],
        cwd=repo_root,
    )

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    assert wheels, "Expected a built wheel artifact."
    assert sdists, "Expected a built sdist artifact."

    # Source-of-truth files that must be available post-install.
    required_relpaths = [
        "config/models.json",
    ]

    # Warmup priors (.joblib) are intentionally NOT shipped in the wheel.
    # Users generate their own via generate_warmup_priors().
    source_joblibs: list[str] = []

    # Check sdist content includes required files.
    with tarfile.open(sdists[0], "r:gz") as tf:
        sdist_names = tf.getnames()
    for rel in required_relpaths:
        assert any(name.endswith(f"src/pareto_bandit/{rel}") for name in sdist_names), rel

    # Install wheel into isolated venv and assert runtime availability.
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python_bin = venv_dir / "bin" / "python"
    pip_bin = venv_dir / "bin" / "pip"
    _run([str(pip_bin), "install", str(wheels[0])], cwd=repo_root)

    check_script = r"""
import json
from pathlib import Path
import pareto_bandit

pkg_root = Path(pareto_bandit.__file__).resolve().parent
required = json.loads(Path.cwd().joinpath("_required_files.json").read_text(encoding="utf-8"))
source_joblibs = json.loads(Path.cwd().joinpath("_source_joblibs.json").read_text(encoding="utf-8"))

for rel in required:
    path = pkg_root / rel
    if not path.exists():
        raise AssertionError(f"Missing required packaged file: {rel}")

for rel in source_joblibs:
    path = pkg_root / rel
    if not path.exists():
        raise AssertionError(f"Source joblib artifact missing in wheel: {rel}")
"""

    (tmp_path / "_required_files.json").write_text(
        json.dumps(required_relpaths), encoding="utf-8"
    )
    (tmp_path / "_source_joblibs.json").write_text(
        json.dumps(source_joblibs), encoding="utf-8"
    )
    _run([str(python_bin), "-c", check_script], cwd=tmp_path)
