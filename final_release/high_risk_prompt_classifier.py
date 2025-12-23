#!/usr/bin/env python3
"""
high_risk_prompt_classifier.py

Binary classifier:
  - "high"      -> hallucinations unacceptable
  - "not_high"  -> standard prompts

No ML. No APIs. Fully deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Pattern, Tuple


# =========================
# Utilities
# =========================

def _c(pattern: str) -> Pattern:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


# =========================
# Result object
# =========================

@dataclass
class HighRiskResult:
    label: str                 # "high" | "not_high"
    score: float
    threshold: float
    matched_rules: List[str]
    matched_text: List[str]


# =========================
# Classifier
# =========================

class HighRiskPromptClassifier:
    """
    Binary risk classifier.
    Output is ONLY:
      - "high"
      - "not_high"
    """

    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold

        # -------------------------
        # Core risk domains
        # -------------------------
        self.rules: Dict[str, List[Tuple[str, Pattern, float]]] = {

            # ===== MEDICAL =====
            "medical": [
                ("diagnosis_or_treatment",
                 _c(r"\b(diagnos(e|is|ing)|what\s+do\s+i\s+have|treatment|cure|prognosis)\b"),
                 3.0),

                ("dosage_or_prescription",
                 _c(r"\b(dos(e|age)|how\s+much\s+should\s+i\s+take|mg|mcg|ml|tablet|pill|inject)\b"),
                 3.0),

                ("serious_symptoms",
                 _c(r"\b(chest\s+pain|stroke|seizure|shortness\s+of\s+breath|fainting|suicidal)\b"),
                 3.5),

                ("pediatric_or_pregnancy",
                 _c(r"\b(pregnan(t|cy)|breastfeed|newborn|infant|toddler|pediatric)\b"),
                 2.5),
            ],

            # ===== LEGAL =====
            "legal": [
                ("legal_advice",
                 _c(r"\b(legal\s+advice|should\s+i\s+sue|am\s+i\s+liable|my\s+rights)\b"),
                 3.0),

                ("contracts",
                 _c(r"\b(contract|nda|indemnif(y|ication)|termination\s+clause|arbitration)\b"),
                 2.5),

                ("criminal_family_immigration",
                 _c(r"\b(arrest(ed)?|charged|custody|divorce|restraining\s+order|visa|immigration)\b"),
                 3.0),
            ],

            # ===== TAX / FINANCE =====
            "tax_finance": [
                ("tax_filing",
                 _c(r"\b(IRS|tax\s+return|1040|1099|W-2|capital\s+gains|deduction|audit)\b"),
                 3.0),

                ("investment_decision",
                 _c(r"\b(should\s+i\s+buy|should\s+i\s+sell|portfolio|options\s+trade|margin)\b"),
                 2.5),
            ],

            # ===== SAFETY =====
            "physical_safety": [
                ("dangerous_instructions",
                 _c(r"\b(poison|asbestos|carbon\s+monoxide|chemical\s+spill|electrocution)\b"),
                 3.5),

                ("structural_electrical",
                 _c(r"\b(load[-\s]?bearing|electrical\s+wiring|gas\s+leak)\b"),
                 2.5),
            ],

            # ===== REGULATED / COMPLIANCE =====
            "regulated": [
                ("medical_regulation",
                 _c(r"\b(HIPAA|FDA|clinical\s+trial|medical\s+device)\b"),
                 2.5),

                ("data_privacy",
                 _c(r"\b(GDPR|CCPA|personal\s+data|PII|PHI)\b"),
                 2.5),
            ],
        }

        # -------------------------
        # Intent amplifiers
        # -------------------------
        self.intent_boosters: List[Tuple[str, Pattern, float]] = [
            ("instructional",
             _c(r"\b(tell\s+me\s+what\s+to\s+do|step[-\s]?by[-\s]?step|instructions)\b"),
             1.5),

            ("urgency",
             _c(r"\b(urgent|emergency|immediately|right\s+now)\b"),
             1.0),

            ("action_request",
             _c(r"\b(should\s+i|can\s+i|is\s+it\s+safe\s+to)\b"),
             1.5),
        ]

    # =========================
    # Main API
    # =========================

    def classify(self, prompt: str) -> HighRiskResult:
        prompt = prompt.strip()
        score = 0.0
        matched_rules: List[str] = []
        matched_text: List[str] = []

        # Domain rules
        for domain, rules in self.rules.items():
            domain_hit = False
            for name, pattern, weight in rules:
                for m in pattern.finditer(prompt):
                    domain_hit = True
                    score += weight
                    matched_rules.append(f"{domain}:{name}")
                    matched_text.append(m.group(0))

            # Base penalty if domain hit at all
            if domain_hit:
                score += 1.5

        # Intent boosters
        for name, pattern, weight in self.intent_boosters:
            for m in pattern.finditer(prompt):
                score += weight
                matched_rules.append(f"intent:{name}")
                matched_text.append(m.group(0))

        label = "high" if score >= self.threshold else "not_high"

        return HighRiskResult(
            label=label,
            score=round(score, 2),
            threshold=self.threshold,
            matched_rules=matched_rules,
            matched_text=matched_text,
        )


# =========================
# Example usage
# =========================

if __name__ == "__main__":
    clf = HighRiskPromptClassifier()

    examples = [
        "What dose of amoxicillin should I give my 2 year old?",
        "Draft an NDA clause for a contractor in California.",
        "Explain transformers in simple terms.",
        "Should I sell my Tesla stock today?",
        "Write a poem about the ocean.",
    ]

    for e in examples:
        r = clf.classify(e)
        print("=" * 70)
        print(e)
        print(f"Label: {r.label} | score={r.score}")
        print("Rules:", r.matched_rules)
