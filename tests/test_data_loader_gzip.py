"""
Tests for JSONL data loading and gzip decompression.

Validates that the canonical per-prompt reward format can be loaded from
both plain ``.jsonl`` and compressed ``.jsonl.gz`` files, and that the
shipped train/val/test splits conform to the expected schema.
"""

import gzip
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from bandit_gpt.config import (
    K3_ARM_ORDER,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    HOLDOUT_DATA_PATH,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_RECORD_KEYS = {"prompt", "arms"}
REQUIRED_ARM_KEYS = {"reward", "cost"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file, auto-detecting gzip by extension."""
    open_fn = gzip.open if path.suffix == ".gz" else open
    mode = "rt" if path.suffix == ".gz" else "r"
    records: List[Dict[str, Any]] = []
    with open_fn(path, mode) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _make_sample_records(n: int = 5) -> List[Dict[str, Any]]:
    """Return *n* synthetic records in the canonical per-prompt format."""
    arms = {
        "model-a": {"reward": 0.9, "cost": 1e-5, "near_best": True},
        "model-b": {"reward": 0.7, "cost": 5e-4, "near_best": False},
    }
    return [
        {
            "prompt": f"Synthetic prompt {i}",
            "difficulty": "easy",
            "best_arm": "model-a",
            "best_reward": 0.9,
            "worst_reward": 0.7,
            "reward_spread": 0.2,
            "arms": arms,
            "source": "test",
        }
        for i in range(n)
    ]


def _write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records as newline-delimited JSON."""
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_jsonl_gz(records: List[Dict[str, Any]], path: Path) -> None:
    """Write records as gzip-compressed newline-delimited JSON."""
    with gzip.open(path, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------------------
# Gzip round-trip tests (self-contained, no real data needed)
# ---------------------------------------------------------------------------


class TestGzipRoundTrip:
    """Verify plain and gzip JSONL produce identical records."""

    @pytest.fixture(autouse=True)
    def _tmpdir(self, tmp_path: Path) -> None:
        self.tmp = tmp_path

    def test_plain_jsonl_loads_correctly(self) -> None:
        records = _make_sample_records(10)
        path = self.tmp / "data.jsonl"
        _write_jsonl(records, path)

        loaded = _load_jsonl(path)
        assert len(loaded) == 10
        assert loaded[0]["prompt"] == "Synthetic prompt 0"
        assert loaded[0]["arms"]["model-a"]["reward"] == pytest.approx(0.9)

    def test_gzip_jsonl_loads_correctly(self) -> None:
        records = _make_sample_records(10)
        path = self.tmp / "data.jsonl.gz"
        _write_jsonl_gz(records, path)

        loaded = _load_jsonl(path)
        assert len(loaded) == 10
        assert loaded[0]["arms"]["model-b"]["cost"] == pytest.approx(5e-4)

    def test_plain_and_gzip_are_identical(self) -> None:
        records = _make_sample_records(25)
        plain_path = self.tmp / "data.jsonl"
        gz_path = self.tmp / "data.jsonl.gz"
        _write_jsonl(records, plain_path)
        _write_jsonl_gz(records, gz_path)

        assert _load_jsonl(plain_path) == _load_jsonl(gz_path)

    def test_large_file_streaming(self) -> None:
        records = _make_sample_records(5_000)
        path = self.tmp / "large.jsonl.gz"
        _write_jsonl_gz(records, path)

        loaded = _load_jsonl(path)
        assert len(loaded) == 5_000


# ---------------------------------------------------------------------------
# Schema validation on shipped canonical splits
# ---------------------------------------------------------------------------


def _validate_record_schema(record: Dict[str, Any]) -> None:
    """Assert a single record conforms to the canonical per-prompt schema."""
    missing = REQUIRED_RECORD_KEYS - record.keys()
    assert not missing, f"Missing top-level keys: {missing}"

    assert isinstance(record["prompt"], str) and record["prompt"].strip()
    assert isinstance(record["arms"], dict) and len(record["arms"]) > 0

    for arm_id, arm_info in record["arms"].items():
        arm_missing = REQUIRED_ARM_KEYS - arm_info.keys()
        assert not arm_missing, f"Arm {arm_id!r} missing keys: {arm_missing}"
        assert isinstance(arm_info["reward"], (int, float))
        assert isinstance(arm_info["cost"], (int, float))
        assert 0.0 <= arm_info["reward"] <= 1.0, (
            f"reward out of [0,1]: {arm_info['reward']}"
        )
        assert arm_info["cost"] >= 0.0


@pytest.mark.skipif(
    not TRAIN_DATA_PATH.exists(), reason="Canonical train.jsonl not found"
)
class TestCanonicalSplits:
    """Validate schema and basic integrity of the shipped K=3 splits."""

    SPLITS = {
        "train": (TRAIN_DATA_PATH, 8_374),
        "val": (VAL_DATA_PATH, 1_785),
        "test": (HOLDOUT_DATA_PATH, 1_824),
    }

    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_split_row_count(self, split_name: str) -> None:
        path, expected_count = self.SPLITS[split_name]
        records = _load_jsonl(path)
        assert len(records) == expected_count, (
            f"{split_name}: expected {expected_count} rows, got {len(records)}"
        )

    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_split_schema(self, split_name: str) -> None:
        path, _ = self.SPLITS[split_name]
        records = _load_jsonl(path)
        for i, r in enumerate(records):
            _validate_record_schema(r)

    @pytest.mark.parametrize("split_name", ["train", "val", "test"])
    def test_split_contains_k3_arms(self, split_name: str) -> None:
        path, _ = self.SPLITS[split_name]
        records = _load_jsonl(path)
        for r in records[:50]:
            for arm_id in K3_ARM_ORDER:
                assert arm_id in r["arms"], (
                    f"Missing arm {arm_id!r} in record: {r['prompt'][:60]}"
                )

    def test_no_prompt_overlap_across_splits(self) -> None:
        prompts_by_split = {}
        for split_name, (path, _) in self.SPLITS.items():
            records = _load_jsonl(path)
            prompts_by_split[split_name] = {r["prompt"] for r in records}

        for a in self.SPLITS:
            for b in self.SPLITS:
                if a >= b:
                    continue
                overlap = prompts_by_split[a] & prompts_by_split[b]
                assert not overlap, (
                    f"{a}/{b} share {len(overlap)} prompts"
                )
