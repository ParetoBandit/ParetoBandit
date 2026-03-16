#!/usr/bin/env python3
"""Synthetic rubric test: verify the new discriminative rubric produces
expected score separation across quality tiers.

Sends 10 prompts x 3 quality tiers (weak / mid / strong) = 30 tasks to
DeepSeek-R1 with both the OLD and NEW rubric, then compares discrimination.

Usage
-----
    python data_collection/scripts/synthetic_rubric_test.py

Requirements
------------
    export OPENROUTER_API_KEY=...
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        API_KEY = os.getenv("OPENROUTER_API_KEY")
    except ImportError:
        pass
if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set")

BASE_URL = "https://openrouter.ai/api/v1"

# ── Synthetic test data ─────────────────────────────────────────────────

SYNTHETIC_DATA: List[Dict[str, Any]] = [
    {
        "prompt": "What is the time complexity of binary search and why?",
        "weak": (
            "Binary search has a time complexity of O(n) because it looks "
            "through each element one by one until it finds the target. "
            "It's a very efficient algorithm that works well on any list "
            "of data. The algorithm compares each element sequentially "
            "and returns the index when found."
        ),
        "mid": (
            "Binary search has O(log n) time complexity. It works by "
            "repeatedly dividing the sorted array in half. At each step, "
            "it compares the middle element to the target and eliminates "
            "half the remaining elements. So for n elements, it takes at "
            "most log₂(n) comparisons."
        ),
        "strong": (
            "Binary search achieves O(log n) time complexity because each "
            "comparison eliminates exactly half of the remaining search "
            "space. Formally, if T(n) is the number of comparisons for an "
            "array of size n, then T(n) = T(n/2) + 1, with T(1) = 1. By "
            "the Master Theorem (case 2), this solves to T(n) = Θ(log n). "
            "Importantly, this requires the input to be sorted — on "
            "unsorted data you'd need O(n log n) preprocessing. The space "
            "complexity is O(1) for iterative and O(log n) for recursive "
            "implementations due to stack frames. In practice, binary "
            "search is cache-friendly since it accesses contiguous memory, "
            "though for very small n (< ~64), linear search can be faster "
            "due to branch prediction and simpler control flow."
        ),
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome.",
        "weak": (
            "```python\ndef is_palindrome(s):\n    for i in range(len(s)):\n"
            "        if s[i] != s[len(s) - i]:\n            return False\n"
            "    return True\n```\n"
            "This function checks each character against its mirror position. "
            "It's efficient and handles all cases."
        ),
        "mid": (
            "```python\ndef is_palindrome(s: str) -> bool:\n"
            "    s = s.lower().replace(' ', '')\n"
            "    return s == s[::-1]\n```\n"
            "This normalizes the string to lowercase and removes spaces, "
            "then checks if it equals its reverse. For example, "
            "'Race Car' returns True."
        ),
        "strong": (
            "```python\nimport re\n\n\ndef is_palindrome(s: str) -> bool:\n"
            '    """Check whether *s* is a palindrome, ignoring case and\n'
            "    non-alphanumeric characters.\n\n"
            "    >>> is_palindrome('A man, a plan, a canal: Panama')\n"
            "    True\n"
            '    >>> is_palindrome("hello")\n'
            "    False\n"
            '    """\n'
            "    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()\n"
            "    return cleaned == cleaned[::-1]\n```\n"
            "Key design choices: (1) strip all non-alphanumeric characters, "
            "not just spaces, so punctuation like commas and colons are "
            "handled; (2) case-insensitive via `.lower()`; (3) O(n) time "
            "and O(n) space for the cleaned copy. For O(1) space you could "
            "use two pointers converging from both ends, skipping non-alnum "
            "characters in-place."
        ),
    },
    {
        "prompt": "Explain how vaccines work to protect against diseases.",
        "weak": (
            "Vaccines work by injecting a small amount of the disease into "
            "your body so you get a mild version of the illness. This makes "
            "your body stronger and able to fight it off next time. The "
            "vaccine basically gives you the disease but in a controlled way. "
            "After getting vaccinated, you're immune forever because your "
            "body remembers the disease."
        ),
        "mid": (
            "Vaccines introduce a weakened or inactivated form of a pathogen "
            "(or parts of it) into the body. This triggers the immune system "
            "to produce antibodies without causing the actual disease. The "
            "immune system also creates memory cells, so if you encounter "
            "the real pathogen later, your body can mount a rapid response. "
            "This is why booster shots are sometimes needed — to refresh "
            "that immune memory."
        ),
        "strong": (
            "Vaccines exploit the adaptive immune system's ability to form "
            "immunological memory. They present antigens — proteins or "
            "fragments from a pathogen — to antigen-presenting cells "
            "(dendritic cells, macrophages), which activate both arms of "
            "adaptive immunity:\n\n"
            "1. **Humoral response**: B cells differentiate into plasma "
            "cells producing pathogen-specific antibodies, plus long-lived "
            "memory B cells.\n"
            "2. **Cell-mediated response**: CD8+ cytotoxic T cells learn to "
            "kill infected cells; CD4+ helper T cells coordinate the "
            "response.\n\n"
            "Modern vaccine platforms differ in how they deliver antigens: "
            "live-attenuated (MMR), inactivated (polio IPV), subunit "
            "(hepatitis B), mRNA (COVID-19 Pfizer/Moderna — delivers "
            "instructions for cells to produce the spike protein), and "
            "viral vector (AstraZeneca). Immunity isn't always lifelong — "
            "antibody titers wane, which is why boosters are needed. "
            "Herd immunity thresholds depend on R₀; for measles (R₀ ≈ 15), "
            "~93% coverage is required."
        ),
    },
    {
        "prompt": "What causes the seasons on Earth?",
        "weak": (
            "The seasons are caused by the Earth's distance from the Sun. "
            "In summer, the Earth is closer to the Sun so it's hotter. In "
            "winter, the Earth is farther away so it's colder. This is why "
            "the Southern Hemisphere has summer when the Northern Hemisphere "
            "has winter — they're on different sides of the Earth relative "
            "to the Sun."
        ),
        "mid": (
            "Seasons are caused by the Earth's axial tilt of about 23.5°. "
            "As Earth orbits the Sun, different hemispheres are tilted "
            "toward or away from the Sun. When the Northern Hemisphere is "
            "tilted toward the Sun, it gets more direct sunlight and longer "
            "days, causing summer. Six months later, it's tilted away, "
            "resulting in winter. The distance from the Sun doesn't matter "
            "much — Earth is actually closest to the Sun in January."
        ),
        "strong": (
            "Earth's seasons arise from its 23.44° axial obliquity combined "
            "with its orbital motion around the Sun. Two mechanisms create "
            "seasonal temperature variation:\n\n"
            "1. **Solar angle**: When a hemisphere tilts sunward, incoming "
            "radiation strikes at a higher angle, concentrating energy per "
            "unit area (flux ∝ sin(elevation angle)).\n"
            "2. **Day length**: The tilted hemisphere also experiences "
            "longer daylight hours, increasing total insolation.\n\n"
            "Crucially, Earth's slightly elliptical orbit (e ≈ 0.017) plays "
            "a negligible role — perihelion occurs around January 3, during "
            "Northern Hemisphere winter. This is a common misconception. "
            "The seasonal lag (warmest month ~1 month after solstice) occurs "
            "because oceans and landmasses have thermal inertia. Near the "
            "equator, where the solar angle is always high, seasonal "
            "variation is minimal; instead, wet/dry cycles driven by the "
            "ITCZ dominate."
        ),
    },
    {
        "prompt": "Solve: A train leaves station A at 60 km/h. Another train leaves station B (300 km away) at 90 km/h toward A. When do they meet?",
        "weak": (
            "The trains are going toward each other so we add the speeds: "
            "60 + 90 = 150 km/h. The distance is 300 km. So they meet "
            "after 300 / 150 = 3 hours. They meet at a point 180 km from "
            "station A, which is 180 km from station B."
        ),
        "mid": (
            "Since both trains move toward each other, their combined "
            "approach speed is 60 + 90 = 150 km/h. Time to meet = "
            "300 / 150 = 2 hours. In that time, Train A travels "
            "60 × 2 = 120 km from A, and Train B travels 90 × 2 = 180 km "
            "from B. Check: 120 + 180 = 300 km ✓."
        ),
        "strong": (
            "Let t be the time in hours after departure. Setting up "
            "the position equations with A at origin:\n\n"
            "- Train A position: x_A(t) = 60t\n"
            "- Train B position: x_B(t) = 300 − 90t\n\n"
            "They meet when x_A(t) = x_B(t):\n"
            "60t = 300 − 90t → 150t = 300 → t = 2 hours.\n\n"
            "Meeting point: 60 × 2 = 120 km from A (180 km from B).\n\n"
            "Sanity check: 120 + 180 = 300 ✓. Equivalently, with "
            "combined closure rate 150 km/h over 300 km, t = 2 h, "
            "consistent with the algebraic solution. Note this assumes "
            "simultaneous departure and constant speeds (no acceleration "
            "phase)."
        ),
    },
    {
        "prompt": "What are the main differences between TCP and UDP?",
        "weak": (
            "TCP and UDP are both internet protocols. TCP is faster because "
            "it sends data in a continuous stream, while UDP sends individual "
            "packets that can arrive in any order. TCP is used for "
            "downloading files and UDP is used for streaming video. The main "
            "difference is that TCP is more reliable and UDP is more "
            "efficient."
        ),
        "mid": (
            "TCP (Transmission Control Protocol) is connection-oriented — "
            "it establishes a connection via a three-way handshake, "
            "guarantees delivery through acknowledgments and retransmissions, "
            "and ensures in-order delivery. UDP (User Datagram Protocol) is "
            "connectionless — it sends datagrams without guarantees of "
            "delivery or ordering, making it faster but less reliable. TCP "
            "is used for web browsing, email, and file transfers; UDP for "
            "real-time applications like video streaming, gaming, and DNS."
        ),
        "strong": (
            "TCP and UDP operate at the transport layer (Layer 4) and differ "
            "across several dimensions:\n\n"
            "| Aspect | TCP | UDP |\n"
            "|--------|-----|-----|\n"
            "| Connection | 3-way handshake (SYN/SYN-ACK/ACK) | Connectionless |\n"
            "| Reliability | ACKs, retransmission, checksums | Best-effort, checksum optional in IPv4 |\n"
            "| Ordering | Sequence numbers guarantee order | No ordering guarantees |\n"
            "| Flow control | Sliding window (receiver-driven) | None |\n"
            "| Congestion control | Slow start, AIMD, fast retransmit | None (application must handle) |\n"
            "| Header size | 20-60 bytes | 8 bytes |\n"
            "| Overhead | Higher (connection state, retransmissions) | Lower |\n\n"
            "The choice depends on the application's tolerance for packet loss "
            "vs. latency sensitivity. Real-time applications (VoIP, gaming) "
            "prefer UDP because a retransmitted packet arriving late is worse "
            "than a dropped one. QUIC (used by HTTP/3) builds reliability on "
            "top of UDP to get TCP-like guarantees with lower handshake "
            "latency (0-RTT resumption)."
        ),
    },
    {
        "prompt": "A farmer has 100 meters of fencing. What dimensions maximize the area of a rectangular enclosure?",
        "weak": (
            "To maximize area, you want the rectangle to be as big as "
            "possible. If the perimeter is 100 meters, then each side is "
            "25 meters (100 / 4 = 25). So the maximum area is "
            "25 × 25 = 625 square meters. A square is always the most "
            "efficient shape."
        ),
        "mid": (
            "Let the sides be x and y. Perimeter: 2x + 2y = 100, so "
            "y = 50 − x. Area A = x(50 − x) = 50x − x². To maximize, "
            "take dA/dx = 50 − 2x = 0, giving x = 25. So y = 25 as well. "
            "The maximum area is 25 × 25 = 625 m². A square maximizes "
            "area for a given perimeter."
        ),
        "strong": (
            "Let the sides be x and y with constraint 2x + 2y = 100 "
            "(i.e., y = 50 − x, where 0 < x < 50).\n\n"
            "Area: A(x) = x(50 − x) = 50x − x².\n\n"
            "First derivative: A'(x) = 50 − 2x = 0 → x = 25.\n"
            "Second derivative: A''(x) = −2 < 0, confirming a maximum.\n\n"
            "So x = y = 25 m, giving A_max = 625 m².\n\n"
            "This result generalizes: among all rectangles with fixed "
            "perimeter, the square maximizes area (by AM-GM: xy ≤ "
            "((x+y)/2)² with equality iff x = y). More broadly, among all "
            "closed curves with perimeter P, the circle maximizes area "
            "(isoperimetric inequality: A ≤ P²/4π). A circular enclosure "
            "with 100 m of fencing would yield A = 10000/4π ≈ 795.8 m², "
            "about 27% more than the square — relevant if the farmer isn't "
            "constrained to a rectangle."
        ),
    },
    {
        "prompt": "Explain the difference between correlation and causation with an example.",
        "weak": (
            "Correlation means two things happen together, and causation "
            "means one thing causes the other. For example, ice cream sales "
            "and drowning deaths are correlated because eating ice cream "
            "causes people to go swimming more, which leads to more "
            "drownings. So in this case, correlation does imply causation "
            "because there's a clear mechanism."
        ),
        "mid": (
            "Correlation means two variables tend to move together — when "
            "one increases, the other tends to increase (or decrease). "
            "Causation means one variable directly influences the other. "
            "A classic example: ice cream sales and drowning deaths are "
            "positively correlated, but ice cream doesn't cause drowning. "
            "The confounding variable is temperature — hot weather increases "
            "both ice cream consumption and swimming, leading to the "
            "observed correlation."
        ),
        "strong": (
            "**Correlation** measures the statistical association between "
            "two variables (e.g., Pearson's r). **Causation** means "
            "intervening on X changes the distribution of Y (the do-calculus "
            "notion: P(Y | do(X)) ≠ P(Y)).\n\n"
            "Classic example: ice cream sales and drowning deaths are "
            "positively correlated (r ≈ 0.7 in summer data). But banning "
            "ice cream wouldn't reduce drownings. The confounder is "
            "temperature, which drives both. The causal DAG:\n\n"
            "Temperature → Ice cream sales\n"
            "Temperature → Swimming → Drowning\n\n"
            "To establish causation, you need one of:\n"
            "1. **Randomized experiment** (gold standard): randomly assign "
            "treatment and measure outcome.\n"
            "2. **Natural experiment / instrumental variable**: exploit "
            "exogenous variation.\n"
            "3. **Causal inference from observational data**: use methods "
            "like propensity score matching, difference-in-differences, or "
            "regression discontinuity to control for confounders.\n\n"
            "Simpson's paradox illustrates how confounders can even reverse "
            "the direction of an association — UC Berkeley's 1973 admission "
            "data appeared to show gender bias that disappeared after "
            "conditioning on department."
        ),
    },
    {
        "prompt": "What is the output of this Python code?\n\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)",
        "weak": (
            "The output is [1, 2, 3]. The variable y is a copy of x, so "
            "when you append 4 to y, it only changes y. x remains unchanged "
            "because Python creates a new list when you assign y = x. This "
            "is how variables work in Python — each variable has its own "
            "independent copy of the data."
        ),
        "mid": (
            "The output is [1, 2, 3, 4]. In Python, `y = x` doesn't "
            "create a copy — both variables point to the same list object "
            "in memory. So `y.append(4)` modifies the shared list, and "
            "`print(x)` shows the updated list."
        ),
        "strong": (
            "Output: `[1, 2, 3, 4]`\n\n"
            "In Python, `y = x` creates an alias, not a copy. Both names "
            "reference the same `list` object on the heap (you can verify "
            "with `id(x) == id(y)` → `True`). Since lists are mutable, "
            "`y.append(4)` mutates the shared object in-place.\n\n"
            "This is a common source of bugs. To create an independent "
            "copy:\n"
            "- Shallow copy: `y = x.copy()` or `y = x[:]` or "
            "`y = list(x)`\n"
            "- Deep copy (for nested structures): "
            "`import copy; y = copy.deepcopy(x)`\n\n"
            "This behavior applies to all mutable types (lists, dicts, "
            "sets). Immutable types (int, str, tuple) don't have this "
            "issue because operations create new objects rather than "
            "mutating existing ones."
        ),
    },
    {
        "prompt": "Should governments regulate social media to combat misinformation?",
        "weak": (
            "Yes, the government should definitely regulate social media "
            "because there is too much fake news. Companies like Facebook "
            "and Twitter don't do enough to stop it. If the government "
            "makes strict rules, misinformation will be eliminated and "
            "people will only see true information. Other countries have "
            "already done this successfully, proving it works."
        ),
        "mid": (
            "This is a complex issue with valid arguments on both sides. "
            "Proponents of regulation argue that unchecked misinformation "
            "causes real harm — it can influence elections, discourage "
            "vaccination, and incite violence. Critics argue that "
            "government regulation risks censorship, is difficult to "
            "implement fairly (who defines 'misinformation'?), and could "
            "stifle free speech. A balanced approach might involve "
            "requiring transparency in content moderation algorithms and "
            "supporting media literacy education, rather than direct "
            "content censorship."
        ),
        "strong": (
            "This requires weighing competing values — public safety, "
            "free expression (First Amendment in the US / Article 10 ECHR "
            "in Europe), and practical enforceability.\n\n"
            "**Arguments for regulation**:\n"
            "- Documented harms: vaccine hesitancy correlated with "
            "misinformation exposure (Loomba et al., 2021); election "
            "interference (Mueller Report, 2019); real-world violence "
            "(Myanmar/Facebook, UN report 2018).\n"
            "- Market failure: platforms profit from engagement, which "
            "algorithmic amplification of outrage maximizes.\n"
            "- Precedent: the EU's Digital Services Act (2024) requires "
            "platforms to assess systemic risks and provide transparency.\n\n"
            "**Arguments against**:\n"
            "- Definitional problem: 'misinformation' is often contested "
            "(lab leak hypothesis went from 'misinformation' to plausible "
            "in 18 months).\n"
            "- Chilling effects: over-regulation suppresses legitimate "
            "speech, especially for marginalized voices.\n"
            "- Government capture: regulators in authoritarian contexts "
            "use 'anti-misinformation' laws to silence dissent (Russia's "
            "'fake news' law, Singapore's POFMA).\n\n"
            "**Middle-ground approaches**:\n"
            "1. Transparency mandates (algorithmic audits, ad archives) "
            "rather than content removal.\n"
            "2. Structural interventions: friction on resharing, "
            "prebunking/inoculation-based media literacy.\n"
            "3. Independent oversight bodies (like the EU's model) rather "
            "than direct government content moderation.\n\n"
            "The evidence suggests that structural and transparency-based "
            "regulation is more effective and less risky than direct "
            "content censorship."
        ),
    },
]

# ── Rubric prompts ──────────────────────────────────────────────────────

OLD_RUBRIC = (
    "You are a Discriminative Router Judge. Your goal is to evaluate "
    "how well an LLM response addresses the given prompt.\n\n"
    "Score on three continuous dimensions (0.0–1.0). Use the FULL "
    "range; do NOT default to 0 or 1.\n\n"
    "1. **Reasoning Quality (40 %)** — How sound is the reasoning?\n"
    "   0.9–1.0 Flawless; every step correct and clearly justified.\n"
    "   0.7–0.8 Sound overall; minor inefficiency or a trivial error "
    "that does not change the conclusion.\n"
    "   0.5–0.6 Partially correct; approach is reasonable but "
    "important steps are wrong or missing.\n"
    "   0.3–0.4 Weak; only fragments of correct logic.\n"
    "   0.0–0.2 No coherent reasoning, or completely wrong approach.\n"
    "   If the prompt needs no multi-step reasoning, score factual "
    "accuracy and depth of explanation.\n\n"
    "2. **Instruction Following (30 %)** — Were all explicit and "
    "implicit constraints satisfied?\n"
    "   0.9–1.0 Every constraint followed precisely.\n"
    "   0.7–0.8 All major constraints met; one minor instruction "
    "partially missed.\n"
    "   0.5–0.6 Some important instructions missed or only partially "
    "addressed.\n"
    "   0.3–0.4 Multiple instructions ignored or misinterpreted.\n"
    "   0.0–0.2 Response largely ignores the prompt's requirements.\n\n"
    "3. **Communication Quality (30 %)** — How clear, well-structured, "
    "and useful is the response?\n"
    "   0.9–1.0 Exceptionally clear, well-organized, appropriate "
    "detail.\n"
    "   0.7–0.8 Clear and competent; minor improvements possible.\n"
    "   0.5–0.6 Adequate but noticeably unclear, verbose, or poorly "
    "organized.\n"
    "   0.3–0.4 Hard to follow; significant clarity issues.\n"
    "   0.0–0.2 Unintelligible, unhelpful, or inappropriate tone.\n\n"
    "Format your response EXACTLY as follows:\n\n"
    "## Reasoning\n"
    "<Concise chain-of-thought analysis>\n\n"
    "## Reasoning Quality\n"
    "<0.0 to 1.0>\n\n"
    "## Instruction Following\n"
    "<0.0 to 1.0>\n\n"
    "## Communication Quality\n"
    "<0.0 to 1.0>"
)

NEW_RUBRIC = (
    "## Role\n"
    "You are an Expert Meta-Cognitive Verifier. Your task is to rigorously "
    "audit an LLM response. You must distinguish between \"surface-level "
    "fluency\" and \"structural correctness.\"\n\n"
    "## Internal Protocol (Pre-Scoring)\n"
    "Before providing scores, use your internal thinking space to:\n"
    "1. **Adversarial Deconstruction:** Actively try to find a scenario "
    "where the model's logic fails.\n"
    "2. **Template Detection:** Check if the model is just repeating a "
    "standard training-data pattern or actually reasoning through THIS "
    "specific prompt.\n"
    "3. **Complexity Check:** Does the response address the hardest 10% "
    "of the prompt, or just the easy 90%?\n\n"
    "---\n\n"
    "## Evaluation Criteria\n\n"
    "### 1. Factual Integrity & Grounding (50%)\n"
    "* **Frontier (0.9-1.0):** Zero hallucinations. Identifies if the "
    "prompt contains false premises.\n"
    "* **Mid (0.6-0.8):** Correct on main facts but might miss technical "
    "nuances or specific data points.\n"
    "* **Low (0.0-0.5):** Contains \"Confident Bullshit\"—authoritative "
    "tone but objectively false claims.\n\n"
    "### 2. Logic & Structural Depth (35%)\n"
    "* **Frontier (0.9-1.0):** **Counter-dependency check.** If Step A "
    "changes, does the model correctly update Step B? Exhibits \"System 2\" "
    "thinking.\n"
    "* **Mid (0.5-0.8):** Correct linear logic but \"fragile.\" Fails if "
    "the problem is slightly permuted.\n"
    "* **Low (0.0-0.4):** Circular reasoning, logical leaps, or "
    "\"stochastic parroting\" of the prompt.\n\n"
    "### 3. Edge Case & Nuance Recall (15%)\n"
    "* **Frontier (0.9-1.0):** Mentions at least one non-obvious "
    "limitation, edge case, or \"it depends\" factor.\n"
    "* **Mid (0.5-0.8):** Addresses all explicit parts of the prompt but "
    "ignores implicit complexity.\n"
    "* **Low (0.0-0.4):** Generic, one-size-fits-all response.\n\n"
    "---\n\n"
    "## Calibration for the Judge (Model Tiers)\n"
    "* **Detecting Low-Tier:** High verbosity, low info-density. Look for "
    "\"In conclusion,\" \"It is important to note,\" etc., used to pad a "
    "thin answer.\n"
    "* **Detecting Mid-Tier:** Accurate but \"Safe.\" It looks like a "
    "high-quality Wikipedia summary.\n"
    "* **Detecting Frontier:** Concise or deeply technical where needed. "
    "It might challenge the user's prompt or provide a \"Step 0\" "
    "(clarifying assumptions) that others missed.\n\n"
    "---\n\n"
    "## Output Format\n"
    "1. **Thought Trace:** [Briefly summarize your internal verification "
    "of their logic]\n"
    "2. **Correctness Score:** [0.0 - 1.0]\n"
    "3. **Reasoning Score:** [0.0 - 1.0]\n"
    "4. **Completeness Score:** [0.0 - 1.0]\n"
    "5. **Model Tier Classification:** [Low | Mid | Frontier] + "
    "1-sentence justification."
)

# Dimension weights and heading patterns for each rubric.
OLD_DIMS = {
    "reasoning_quality": (r"Reasoning\s+Quality", 0.40),
    "instruction_following": (r"Instruction\s+Following", 0.30),
    "communication_quality": (r"Communication\s+Quality", 0.30),
}
NEW_DIMS = {
    "correctness": (r"Correctness\s+Score", 0.50),
    "reasoning": (r"Reasoning\s+Score", 0.35),
    "completeness": (r"Completeness\s+Score", 0.15),
}

# ── API helpers ─────────────────────────────────────────────────────────


def _parse_score(content: str, heading: str, *, default: float = 0.5) -> float:
    """Extract a continuous 0.0-1.0 score from various markdown formats.

    Handles:
      - ``## Heading: 0.8``  or  ``## Heading\\n0.8``
      - ``### Heading: 0.8``
      - ``**Heading:** 0.8`` or ``**Heading:** [0.8]``
      - Numbered items: ``2. **Heading:** 0.8``
    """
    patterns = [
        r"#{1,3}\s*" + heading + r"\s*[:\-]?\s*(\d+\.?\d*)",
        r"\*\*" + heading + r"[:\*]*\s*\[?\s*(\d+\.?\d*)",
        r"\d+\.\s*\*\*" + heading + r"[:\*]*\s*\[?\s*(\d+\.?\d*)",
        heading + r"\s*[:\-]\s*\[?\s*(\d+\.?\d*)",
    ]
    for pat in patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if val > 1.0:
                val /= 100.0
            return max(0.0, min(1.0, val))
    return default


def judge_single(
    prompt: str,
    response: str,
    system_prompt: str,
    dims: Dict[str, Tuple[str, float]],
    *,
    judge_model: str = "deepseek/deepseek-r1",
    max_retries: int = 3,
) -> Optional[Dict[str, Any]]:
    """Send one (prompt, response) to R1 and parse dimension scores.

    Args:
        prompt: The user prompt.
        response: The LLM response to judge.
        system_prompt: The rubric system prompt.
        dims: Mapping of dimension name → (heading_regex, weight).
        judge_model: OpenRouter judge model identifier.
        max_retries: Retry count on transient failures.

    Returns:
        Dict with per-dimension scores and composite, or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/paretobandit/llm-jury",
    }

    effective_prompt = system_prompt
    if "deepseek" in judge_model.lower():
        effective_prompt += (
            "\n\nIMPORTANT: Keep analysis sections to 3-5 sentences. "
            "Be direct — identify errors or confirm correctness, "
            "then move to scoring."
        )

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": effective_prompt},
            {"role": "user", "content": f"PROMPT: {prompt}\n\nRESPONSE: {response}"},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()

            result: Dict[str, Any] = {"raw_content": content}
            composite = 0.0
            for dim_name, (heading, weight) in dims.items():
                score = _parse_score(content, heading)
                result[dim_name] = score
                composite += score * weight
            result["composite"] = round(composite, 4)

            tier_match = re.search(
                r"Model\s+Tier\s+Classification[:\*]*\s*\[?\s*(Low|Mid|Frontier)",
                content, re.IGNORECASE,
            )
            if tier_match:
                result["tier_classification"] = tier_match.group(1).capitalize()

            return result

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED after {max_retries} attempts: {e}")
                return None


# ── Main test ───────────────────────────────────────────────────────────

@dataclass
class TestResult:
    prompt_idx: int
    tier: str  # "weak", "mid", "strong"
    rubric: str  # "old", "new"
    scores: Dict[str, float] = field(default_factory=dict)
    composite: float = 0.0


def run_test() -> None:
    """Run the synthetic rubric comparison test."""
    tiers = ["weak", "mid", "strong"]
    rubrics = [
        ("old", OLD_RUBRIC, OLD_DIMS),
        ("new", NEW_RUBRIC, NEW_DIMS),
    ]

    tasks: List[Tuple[int, str, str, str, str, Dict]] = []
    for idx, item in enumerate(SYNTHETIC_DATA):
        for tier in tiers:
            for rubric_name, rubric_prompt, dims in rubrics:
                tasks.append((
                    idx, tier, rubric_name,
                    item["prompt"], item[tier], dims,
                ))

    print(f"Synthetic Rubric Test: {len(SYNTHETIC_DATA)} prompts × "
          f"{len(tiers)} tiers × {len(rubrics)} rubrics = {len(tasks)} tasks")
    print("=" * 70)

    results: List[TestResult] = []

    # Build (prompt_text, response_text, rubric_prompt, dims) for API calls.
    api_tasks = []
    for idx, tier, rubric_name, prompt, response, dims in tasks:
        rubric_prompt = OLD_RUBRIC if rubric_name == "old" else NEW_RUBRIC
        api_tasks.append((idx, tier, rubric_name, prompt, response, rubric_prompt, dims))

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for task_tuple in api_tasks:
            idx, tier, rubric_name, prompt, response, rubric_prompt, dims = task_tuple
            fut = executor.submit(
                judge_single, prompt, response, rubric_prompt, dims,
            )
            futures[fut] = (idx, tier, rubric_name)

        completed = 0
        for fut in as_completed(futures):
            idx, tier, rubric_name = futures[fut]
            res = fut.result()
            completed += 1
            if res is None:
                print(f"  [{completed}/{len(tasks)}] FAILED: prompt {idx}, "
                      f"{tier}, {rubric_name}")
                continue

            tr = TestResult(
                prompt_idx=idx,
                tier=tier,
                rubric=rubric_name,
                scores={k: v for k, v in res.items()
                        if k not in ("raw_content", "composite",
                                     "tier_classification")},
                composite=res["composite"],
            )
            results.append(tr)
            tier_cls = res.get("tier_classification", "")
            tier_str = f" [{tier_cls}]" if tier_cls else ""
            print(f"  [{completed}/{len(tasks)}] prompt {idx:2d} | "
                  f"{tier:6s} | {rubric_name:3s} | "
                  f"composite={res['composite']:.3f}{tier_str}")

    # ── Analysis ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for rubric_name in ["old", "new"]:
        print(f"\n{'─' * 35} {rubric_name.upper()} RUBRIC {'─' * 35}")
        tier_scores: Dict[str, List[float]] = {t: [] for t in tiers}
        tier_dims: Dict[str, Dict[str, List[float]]] = {
            t: {} for t in tiers
        }
        for r in results:
            if r.rubric != rubric_name:
                continue
            tier_scores[r.tier].append(r.composite)
            for dim_name, val in r.scores.items():
                tier_dims[r.tier].setdefault(dim_name, []).append(val)

        print(f"\n  {'Tier':<8s} {'N':>3s} {'Mean':>7s} {'Std':>7s} "
              f"{'Min':>7s} {'Max':>7s}")
        print(f"  {'─' * 42}")
        tier_means = {}
        for tier in tiers:
            arr = np.array(tier_scores[tier])
            if len(arr) == 0:
                continue
            tier_means[tier] = float(arr.mean())
            print(f"  {tier:<8s} {len(arr):3d} {arr.mean():7.3f} "
                  f"{arr.std():7.3f} {arr.min():7.3f} {arr.max():7.3f}")

        if "weak" in tier_means and "mid" in tier_means and "strong" in tier_means:
            gap_wm = tier_means["mid"] - tier_means["weak"]
            gap_ms = tier_means["strong"] - tier_means["mid"]
            total_gap = tier_means["strong"] - tier_means["weak"]
            print(f"\n  Gaps: weak→mid = {gap_wm:+.3f}, "
                  f"mid→strong = {gap_ms:+.3f}, total = {total_gap:.3f}")

        # Per-dimension breakdown.
        print(f"\n  Per-dimension means:")
        all_dims = set()
        for tier in tiers:
            all_dims.update(tier_dims[tier].keys())
        dim_list = sorted(all_dims)
        header = f"  {'Dim':<25s}" + "".join(f" {t:>8s}" for t in tiers)
        print(header)
        print(f"  {'─' * (25 + 9 * len(tiers))}")
        for dim in dim_list:
            row = f"  {dim:<25s}"
            for tier in tiers:
                vals = tier_dims[tier].get(dim, [])
                if vals:
                    row += f" {np.mean(vals):8.3f}"
                else:
                    row += f" {'N/A':>8s}"
            print(row)

    # ── Head-to-head comparison ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("RUBRIC COMPARISON")
    print(f"{'=' * 70}")
    for rubric_name in ["old", "new"]:
        tier_scores_r = {t: [] for t in tiers}
        for r in results:
            if r.rubric == rubric_name:
                tier_scores_r[r.tier].append(r.composite)
        means = {t: np.mean(tier_scores_r[t]) for t in tiers if tier_scores_r[t]}
        total = means.get("strong", 0) - means.get("weak", 0)
        print(f"  {rubric_name.upper():>4s}:  weak={means.get('weak',0):.3f}  "
              f"mid={means.get('mid',0):.3f}  strong={means.get('strong',0):.3f}  "
              f"| total_gap={total:.3f}")

    # Save raw results.
    output_path = PROJECT_ROOT / "data_collection" / "k3_calibrated" / "synthetic_rubric_test.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "prompt_idx": r.prompt_idx,
            "tier": r.tier,
            "rubric": r.rubric,
            "scores": r.scores,
            "composite": r.composite,
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    run_test()
