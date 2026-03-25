"""Shared utilities for auto-generating LaTeX from JSON results.

Each experiment has a ``generate_latex.py`` script that:

1. Reads the JSON results file.
2. Extracts all numbers referenced in the paper.
3. Writes ``_autogen.tex`` — a file of ``\\newcommand`` definitions.
4. Optionally regenerates table and caption ``.tex`` files.

The narrative ``.tex`` files reference these commands instead of
hardcoded numbers, eliminating staleness bugs.

Naming convention
-----------------
Experiment prefixes (to avoid command-name collisions):

- Exp 01 (stationary budget pacing):  ``\\bp``
- Exp 02 (budget + reward-shift):     ``\\bd``
- Exp 03 (catastrophic failure):      ``\\cf``
- Exp 04 (model onboarding):          ``\\mo``
- Exp 05 (hparam optimisation):       ``\\hp``
- Appendix warmup ablation:           ``\\wa``
- Appendix val burn-in ablation:      ``\\vb``
- Appendix prior mismatch:            ``\\prm``
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


# ======================================================================
# Shared I/O
# ======================================================================


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON results file."""
    with open(path, "r") as f:
        return json.load(f)


# ======================================================================
# Shared table constants (Exp 02, 03, and appendix phase tables)
# ======================================================================

BUDGET_LABEL_TO_SHORT: Dict[str, str] = {
    "tight": "Tight",
    "moderate": "Mod",
    "loose": "Loose",
}

BUDGET_TABLE_DISPLAY: Dict[str, str] = {
    "Tight": "Tight",
    "Mod": "Moderate",
    "Loose": "Loose",
}

BINDING_RATIO_LOW: float = 0.95
BINDING_RATIO_HIGH: float = 1.05

PHASE_NAMES: Dict[int, str] = {1: "One", 2: "Two", 3: "Three"}


def format_ratio_cell(
    ratio: float,
    is_paretobandit: bool,
    is_non_binding: bool = False,
) -> str:
    """Format a budget-utilisation ratio cell with optional bold and dagger.

    Args:
        ratio: Budget utilisation ratio (target 1.0).
        is_paretobandit: Whether the row is for ParetoBandit (always bold).
        is_non_binding: If ``True``, appends a dagger superscript.

    Returns:
        LaTeX-ready cell string.
    """
    within_5pct = BINDING_RATIO_LOW <= ratio <= BINDING_RATIO_HIGH
    should_bold = is_paretobandit or within_5pct

    ratio_str = fmt_ratio(ratio)
    inner = f"\\mathbf{{{ratio_str}}}" if should_bold else ratio_str

    if is_non_binding:
        return f"${inner}^{{\\dagger}}$"
    return f"${inner}$"


# ======================================================================
# Number formatting
# ======================================================================


def fmt_reward(val: float, se: Optional[float] = None, digits: int = 3) -> str:
    """Format reward: ``0.908`` or ``0.908 \\pm 0.0007``."""
    s = f"{val:.{digits}f}"
    if se is not None:
        se_digits = max(1, -int(math.floor(math.log10(max(abs(se), 1e-15)))))
        s += f" \\pm {se:.{se_digits}f}"
    return s


def fmt_cost_sci(val: float) -> str:
    r"""Format cost in scientific notation: ``\$2.3{\times}10^{-4}``."""
    if val == 0:
        return "\\$0"
    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / (10 ** exp)
    if exp == 0:
        return f"\\${mantissa:.2f}"
    return f"\\${mantissa:.1f}{{\\times}}10^{{{exp}}}"


def fmt_cost_dollar(val: float) -> str:
    """Format as plain dollar amount: ``\\$0.00182``."""
    if val == 0.0:
        return "\\$0"
    digits = max(2, -int(math.floor(math.log10(abs(val)))) + 2)
    return f"\\${val:.{digits}f}"


def fmt_cost_eng(val: float) -> str:
    """Format cost in engineering notation for tables: ``\\$2.3e-4``."""
    if val == 0.0:
        return "\\$0"
    return f"\\${val:.1e}"


def fmt_ratio(val: float, digits: int = 2) -> str:
    """Format utilisation ratio: ``1.02\\times``."""
    return f"{val:.{digits}f}\\times"


def fmt_num(val: float, digits: int = 1) -> str:
    """Format a plain number."""
    return f"{val:.{digits}f}"


def fmt_pct(val: float, digits: int = 0) -> str:
    """Format a fraction as a whole-number percentage: ``29``."""
    return f"{val * 100:.{digits}f}"


def fmt_int(val: float) -> str:
    """Format as integer."""
    return str(int(round(val)))


# ======================================================================
# LaTeX command accumulator
# ======================================================================


class CommandSet:
    """Accumulates ``\\newcommand`` definitions for one experiment.

    Parameters
    ----------
    prefix : str
        Short prefix prepended to every command name (e.g. ``bp``).
    """

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._cmds: Dict[str, str] = {}

    # -- convenience adders --

    def add(self, name: str, value: str) -> None:
        """``\\newcommand{\\<prefix><name>}{<value>}``."""
        self._cmds[f"{self.prefix}{name}"] = value

    def num(self, name: str, val: float, digits: int = 1) -> None:
        self.add(name, fmt_num(val, digits))

    def reward(self, name: str, val: float, se: Optional[float] = None,
               digits: int = 3) -> None:
        self.add(name, fmt_reward(val, se, digits))

    def cost_sci(self, name: str, val: float) -> None:
        self.add(name, fmt_cost_sci(val))

    def cost_dollar(self, name: str, val: float) -> None:
        self.add(name, fmt_cost_dollar(val))

    def ratio(self, name: str, val: float) -> None:
        self.add(name, fmt_ratio(val))

    def pct(self, name: str, val: float, digits: int = 0) -> None:
        self.add(name, fmt_pct(val, digits))

    def raw(self, name: str, val: str) -> None:
        self.add(name, val)

    # -- emitters --

    def emit(self, header: str = "") -> str:
        """Return the full ``\\newcommand`` block as a string."""
        lines = [
            "% " + "=" * 68,
            f"% AUTO-GENERATED — do not edit by hand.",
            f"% Regenerate: python generate_latex.py",
            "% " + "=" * 68,
        ]
        if header:
            lines.insert(1, f"% {header}")
        lines.append("")
        for cmd_name in self._cmds:
            val = self._cmds[cmd_name]
            lines.append(f"\\newcommand{{\\{cmd_name}}}{{{val}}}")
        lines.append("")
        return "\n".join(lines)

    def write(self, path: Path, header: str = "") -> None:
        path.write_text(self.emit(header))
        print(f"  Wrote {len(self._cmds)} commands → {path}")

    def __len__(self) -> int:
        return len(self._cmds)
