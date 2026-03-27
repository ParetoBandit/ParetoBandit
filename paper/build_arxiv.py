#!/usr/bin/env python3
"""Build a self-contained arXiv submission archive from the paper sources.

The paper's LaTeX sources reference experiment files via ``../experiments/``
relative paths (since main.tex lives in ``paper/``).  arXiv requires a
self-contained upload, so this script:

1. Copies main.tex (and all auxiliary .tex, .sty, .bst, .bbl) to a flat root.
2. Mirrors the subset of ``experiments/`` that the paper actually uses.
3. Rewrites every ``../experiments/`` path to ``experiments/`` so that LaTeX
   resolves them relative to the new root.
4. Packages everything into a ``.tar.gz`` ready for upload.

Usage:
    python build_arxiv.py          # writes  arxiv_submission.tar.gz
    python build_arxiv.py --check  # dry-run: prints what would be included
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tarfile
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
BUILD_DIR = PAPER_DIR / "_arxiv_build"

PAPER_ROOT_FILES: list[str] = [
    "main.tex",
    "mlsys2025.sty",
    "mlsys2025.bst",
    "algorithmic.sty",
    "algorithm.sty",
    "fancyhdr.sty",
    "main.bbl",
    "paper_macros_autogen.tex",
]

PAPER_SUBDIRS: list[str] = [
    "sections",
    "figures",
]


def _collect_experiment_tex_files() -> list[Path]:
    """Return every .tex file under experiments/ that the paper inputs."""
    targets: list[Path] = []

    autogen_files = sorted(REPO_ROOT.glob("experiments/**/_autogen.tex"))
    targets.extend(autogen_files)

    discussion_files = sorted(REPO_ROOT.glob("experiments/**/results_discussion.tex"))
    targets.extend(discussion_files)

    sensitivity_files = sorted(
        REPO_ROOT.glob("experiments/**/t_adapt_sensitivity_discussion.tex")
    )
    targets.extend(sensitivity_files)

    caption_files = sorted(REPO_ROOT.glob("experiments/**/*caption*.tex"))
    targets.extend(caption_files)

    table_files = sorted(REPO_ROOT.glob("experiments/**/table_*.tex"))
    targets.extend(table_files)

    return list(dict.fromkeys(targets))


def _collect_experiment_image_files() -> list[Path]:
    """Return every image file under experiments/ referenced by the paper."""
    return sorted(REPO_ROOT.glob("experiments/**/results/*.png"))


def _rewrite_paths(text: str) -> str:
    r"""Replace ``../experiments/`` with ``experiments/`` for arXiv root layout."""
    return text.replace("../experiments/", "experiments/")


def build(*, check_only: bool = False) -> Path:
    """Build the arXiv submission directory and tarball.

    Args:
        check_only: If True, only print what would be included without writing.

    Returns:
        Path to the generated ``.tar.gz`` archive.
    """
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    tex_files = _collect_experiment_tex_files()
    image_files = _collect_experiment_image_files()

    if check_only:
        print("=== Paper root files ===")
        for f in PAPER_ROOT_FILES:
            p = PAPER_DIR / f
            status = "OK" if p.exists() else "MISSING"
            print(f"  [{status}] {f}")

        print("\n=== Paper subdirectories ===")
        for d in PAPER_SUBDIRS:
            p = PAPER_DIR / d
            if p.is_dir():
                for child in sorted(p.rglob("*")):
                    if child.is_file():
                        print(f"  {child.relative_to(PAPER_DIR)}")

        print("\n=== Experiment .tex files ===")
        for f in tex_files:
            print(f"  {f.relative_to(REPO_ROOT)}")

        print("\n=== Experiment image files ===")
        for f in image_files:
            print(f"  {f.relative_to(REPO_ROOT)}")

        print(f"\nTotal files: {len(PAPER_ROOT_FILES) + len(tex_files) + len(image_files)}")
        return BUILD_DIR

    BUILD_DIR.mkdir(parents=True)

    for fname in PAPER_ROOT_FILES:
        src = PAPER_DIR / fname
        if not src.exists():
            raise FileNotFoundError(f"Required paper file missing: {src}")
        dest = BUILD_DIR / fname
        if fname.endswith(".tex"):
            dest.write_text(_rewrite_paths(src.read_text()))
        else:
            shutil.copy2(src, dest)

    for subdir in PAPER_SUBDIRS:
        src_dir = PAPER_DIR / subdir
        dest_dir = BUILD_DIR / subdir
        if src_dir.is_dir():
            shutil.copytree(src_dir, dest_dir)
            for tex_file in dest_dir.rglob("*.tex"):
                tex_file.write_text(_rewrite_paths(tex_file.read_text()))

    for tex_path in tex_files:
        rel = tex_path.relative_to(REPO_ROOT)
        dest = BUILD_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_rewrite_paths(tex_path.read_text()))

    for img_path in image_files:
        rel = img_path.relative_to(REPO_ROOT)
        dest = BUILD_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dest)

    tarball = PAPER_DIR / "arxiv_submission.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for item in sorted(BUILD_DIR.rglob("*")):
            if item.is_file():
                arcname = item.relative_to(BUILD_DIR)
                tar.add(item, arcname=arcname)

    file_count = sum(1 for _ in BUILD_DIR.rglob("*") if _.is_file())
    size_mb = tarball.stat().st_size / (1024 * 1024)
    print(f"Created {tarball.name}  ({file_count} files, {size_mb:.1f} MB)")

    shutil.rmtree(BUILD_DIR)
    return tarball


def main() -> None:
    parser = argparse.ArgumentParser(description="Build arXiv submission archive.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Dry-run: list files that would be included without writing anything.",
    )
    args = parser.parse_args()
    build(check_only=args.check)


if __name__ == "__main__":
    main()
