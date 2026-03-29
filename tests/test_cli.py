"""Tests for the CLI interface.

Validates argument parsing, default propagation, and that user-facing
flags (especially ``--cost-penalty``) are forwarded correctly to
``BanditRouter.create()``.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pareto_bandit.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(args: list[str]) -> None:
    """Invoke the CLI ``main()`` with the given argv list."""
    with patch("sys.argv", ["pareto-bandit", *args]):
        main()


def _mock_route_log() -> MagicMock:
    log = MagicMock()
    log.selected_model = "mock/model"
    log.predicted_utility = 0.75
    log.cost_usd = 0.00012
    return log


# ---------------------------------------------------------------------------
# --cost-penalty flag
# ---------------------------------------------------------------------------

class TestCostPenaltyCLI:
    """``--cost-penalty`` must be parsed and forwarded to the router."""

    @patch("pareto_bandit.cli.BanditRouter")
    def test_default_cost_penalty_forwarded(self, mock_cls, capsys):
        """Without ``--cost-penalty``, the default 0.3 is passed."""
        mock_router = MagicMock()
        mock_router.route.return_value = ("mock/model", _mock_route_log())
        mock_cls.create.return_value = mock_router

        _run_cli(["Hello world"])

        mock_cls.create.assert_called_once()
        kwargs = mock_cls.create.call_args
        assert kwargs.kwargs.get("cost_penalty") == pytest.approx(0.3)

    @patch("pareto_bandit.cli.BanditRouter")
    def test_custom_cost_penalty_forwarded(self, mock_cls, capsys):
        """An explicit ``--cost-penalty 0.6`` must reach BanditRouter.create()."""
        mock_router = MagicMock()
        mock_router.route.return_value = ("mock/model", _mock_route_log())
        mock_cls.create.return_value = mock_router

        _run_cli(["--cost-penalty", "0.6", "Hello world"])

        mock_cls.create.assert_called_once()
        kwargs = mock_cls.create.call_args
        assert kwargs.kwargs.get("cost_penalty") == pytest.approx(0.6)

    @patch("pareto_bandit.cli.BanditRouter")
    def test_zero_cost_penalty_disables_cost(self, mock_cls, capsys):
        """``--cost-penalty 0`` should forward 0.0 (quality-only routing)."""
        mock_router = MagicMock()
        mock_router.route.return_value = ("mock/model", _mock_route_log())
        mock_cls.create.return_value = mock_router

        _run_cli(["--cost-penalty", "0", "Explain gravity"])

        kwargs = mock_cls.create.call_args
        assert kwargs.kwargs.get("cost_penalty") == pytest.approx(0.0)

    def test_negative_cost_penalty_accepted_by_parser(self):
        """argparse should accept negative floats (validation is the router's job)."""
        with patch("pareto_bandit.cli.BanditRouter") as mock_cls:
            mock_router = MagicMock()
            mock_router.route.return_value = ("m", _mock_route_log())
            mock_cls.create.return_value = mock_router

            _run_cli(["--cost-penalty", "-0.1", "test"])

            kwargs = mock_cls.create.call_args
            assert kwargs.kwargs.get("cost_penalty") == pytest.approx(-0.1)

    @patch("pareto_bandit.cli.BanditRouter")
    def test_cost_penalty_coexists_with_max_cost(self, mock_cls, capsys):
        """Both ``--cost-penalty`` and ``--max-cost`` should be forwarded."""
        mock_router = MagicMock()
        mock_router.route.return_value = ("mock/model", _mock_route_log())
        mock_cls.create.return_value = mock_router

        _run_cli(["--cost-penalty", "0.5", "--max-cost", "0.01", "Prompt"])

        mock_cls.create.assert_called_once()
        assert mock_cls.create.call_args.kwargs["cost_penalty"] == pytest.approx(0.5)

        mock_router.route.assert_called_once()
        route_kwargs = mock_router.route.call_args
        assert route_kwargs.kwargs.get("max_cost") == pytest.approx(0.01)
