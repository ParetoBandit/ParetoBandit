import re


class HeuristicService:
    """
    Utility service for lightweight prompt analysis heuristics.

    Provides reusable logic for:
    - Difficulty detection (proxy for structural complexity)
    - Trap detection (language, adversarial, tool use)
    """

    @staticmethod
    def detect_difficulty(prompt: str) -> float:
        """
        Lightweight difficulty proxy based on prompt length and structural markers.

        Args:
            prompt: Task prompt

        Returns:
            Score in [0.0, 1.0]. Higher is harder.
        """
        if not prompt:
            return 0.0

        # 1. Base length proxy: 0-1000 chars -> 0.0-0.7
        # Long prompts usually imply complex context/constraints
        len_score = min(len(prompt) / 1000.0, 1.0) * 0.7

        # 2. Structural density markers: +0.1 for code/math markers
        structural_markers = [
            '```', 'def ', 'class ', '\\frac', '\\int',
            'integral', 'derivative', 'theorem', 'prove'
        ]
        marker_score = 0.0
        if any(marker in prompt for marker in structural_markers):
            marker_score = 0.3

        return min(len_score + marker_score, 1.0)

    @staticmethod
    def detect_trap(prompt: str) -> bool:
        """
        Detect if a prompt falls into known 'Arbitrage Traps' (Korean, Jailbreaks, Tools).

        Args:
            prompt: Task prompt

        Returns:
            True if prompt matches a trap signature
        """
        # 1. Non-English (Korean/Eastern) detector (Basic range check)
        # Hangul Syllables: U+AC00–U+D7A3
        if re.search(r'[\uac00-\ud7a3]', prompt):
            return True

        # 2. Adversarial / Jailbreak markers
        jailbreak_patterns = [
            r'ignore previous instructions',
            r'system override',
            r'developer mode',
            r'rusted confidant'
        ]
        if any(re.search(pattern, prompt, re.I) for pattern in jailbreak_patterns):
            return True

        # 3. Tool use / Real-time markers
        tool_keywords = ['search for', 'stock price', 'weather forecast', 'right now']
        if any(kw in prompt.lower() for kw in tool_keywords):
            return True

        return False
