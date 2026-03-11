"""EDA: Prompt-level features predictive of reward gaps across K=3 arms.

Investigates which *text-derived* features (beyond raw length) can help a
contextual router decide between small / medium / large.  We intentionally
exclude prompt character-length as a feature because it is a poor proxy for
difficulty and the PCA embedding already captures sequence-level semantics.

Feature families explored
-------------------------
1. **Lexical complexity** — type-token ratio, avg word length, rare-word
   ratio (proxy for vocabulary sophistication).
2. **Structural signals** — question-mark count, enumeration markers,
   code-fence presence, bullet/list density, presence of constraints
   (e.g. "must", "ensure", "exactly").
3. **Task-type indicators** — regex detectors for math/code/reasoning/
   creative/factoid tasks.
4. **Cognitive demand proxies** — number of sub-questions, conditional
   clauses ("if…then"), negation density, comparison requests.
5. **PCA embedding features** — the 32-dim PCA projection used by the
   production router (captures semantic structure).

For each feature we compute:
- Spearman correlation with per-arm reward.
- Spearman correlation with reward *gap* (large − small, medium − small).
- Mutual information with the categorical "best arm" label.
- Feature importance from a gradient-boosted classifier predicting which
  arm wins.

Outputs
-------
- Console tables of top features.
- ``experiments/results/eda_pareto_features/`` with PNG plots and a JSON
  summary.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from bandit_gpt.config import DATA_COLLECTION_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PARETO_DIR = DATA_COLLECTION_DIR / "pareto_dataset"
CLASSIFIED_PATH = PARETO_DIR / "pareto_classified.jsonl"
REWARDS_PATH = PARETO_DIR / "pareto_rewards.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "eda_pareto_features"

# Arm short names (ordered budget → premium)
ARM_ORDER = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-large-2512",
    "google/gemini-2.5-pro",
]
ARM_SHORT = {
    "meta-llama/llama-3.1-8b-instruct": "small",
    "mistralai/mistral-large-2512": "medium",
    "google/gemini-2.5-pro": "large",
}

# Constraint/instruction keywords (case-insensitive)
_CONSTRAINT_WORDS = re.compile(
    r"\b(must|ensure|exactly|at least|at most|no more than|strictly|required|"
    r"constraint|do not|don't|never|always|make sure|limit|restrict)\b",
    re.IGNORECASE,
)
_MATH_PATTERN = re.compile(
    r"(\b(solve|compute|calculate|evaluate|integral|derivative|equation|"
    r"sum|product|factorial|probability|variance|matrix|vector|eigenvalue|"
    r"prove|theorem|lemma)\b|[=+\-*/^].*[=+\-*/^]|\d+\s*[+\-*/^]\s*\d+)",
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(
    r"(```|def\s+\w+|class\s+\w+|function\s+\w+|import\s+\w+|"
    r"\b(write|implement|code|program|algorithm|function|debug|refactor|"
    r"compile|syntax|output of|print|return|loop|array|string|int|float|"
    r"python|java|javascript|c\+\+|sql|html|css)\b)",
    re.IGNORECASE,
)
_REASONING_PATTERN = re.compile(
    r"\b(explain|why|how does|reasoning|logic|deduce|infer|conclude|"
    r"step.by.step|think|analyze|compare|contrast|evaluate|justify|"
    r"argument|because|therefore|hence|implication)\b",
    re.IGNORECASE,
)
_CREATIVE_PATTERN = re.compile(
    r"\b(write a (story|poem|essay|letter|song|script|dialogue)|"
    r"creative|imagine|fiction|narrative|describe a scene|"
    r"compose|draft|rewrite|paraphrase|summarize)\b",
    re.IGNORECASE,
)
_FACTOID_PATTERN = re.compile(
    r"\b(what is|who is|when did|where is|which|name the|"
    r"capital of|population of|define|meaning of)\b",
    re.IGNORECASE,
)
_CONDITIONAL_PATTERN = re.compile(
    r"\b(if|assuming|suppose|given that|provided that|in case|"
    r"when .+ then|unless)\b",
    re.IGNORECASE,
)
_NEGATION_PATTERN = re.compile(
    r"\b(not|no|never|neither|nor|don't|doesn't|didn't|"
    r"won't|can't|cannot|isn't|aren't|wasn't|weren't)\b",
    re.IGNORECASE,
)
_COMPARISON_PATTERN = re.compile(
    r"\b(compare|difference between|versus|vs\.?|better|worse|"
    r"advantage|disadvantage|pros and cons|similarities)\b",
    re.IGNORECASE,
)

# Common English words (top ~200) for rare-word ratio
_COMMON_WORDS = set(
    "the be to of and a in that have i it for not on with he as you do at "
    "this but his by from they we say her she or an will my one all would "
    "there their what so up out if about who get which go me when make can "
    "like time no just him know take people into year your good some could "
    "them see other than then now look only come its over think also back "
    "after use two how our work first well way even new want because any "
    "these give day most us is are was were been has had did does do the a "
    "an and but or if then else when where how what which who whom that this "
    "those".split()
)


def extract_text_features(prompt: str) -> Dict[str, float]:
    """Extract a rich set of text-derived features from a prompt string.

    Returns a flat dictionary of numeric features.  All features are designed
    to capture *complexity*, *task type*, and *cognitive demand* without relying
    on raw character or token length.
    """
    words = prompt.split()
    n_words = max(len(words), 1)
    unique_words = set(w.lower() for w in words)

    # --- Lexical complexity ---
    type_token_ratio = len(unique_words) / n_words
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    rare_ratio = sum(1 for w in words if w.lower() not in _COMMON_WORDS) / n_words

    # --- Structural signals ---
    n_sentences = max(len(re.split(r'[.!?]+', prompt)), 1)
    n_questions = prompt.count("?")
    has_code_fence = int(bool(re.search(r"```", prompt)))
    has_enumeration = int(bool(re.search(r"(\n\s*\d+[.)]\s|\n\s*[-*]\s)", prompt)))
    n_constraints = len(_CONSTRAINT_WORDS.findall(prompt))
    n_newlines = prompt.count("\n")
    has_table = int(bool(re.search(r"\|.*\|.*\|", prompt)))

    # --- Task-type indicators ---
    is_math = int(bool(_MATH_PATTERN.search(prompt)))
    is_code = int(bool(_CODE_PATTERN.search(prompt)))
    is_reasoning = int(bool(_REASONING_PATTERN.search(prompt)))
    is_creative = int(bool(_CREATIVE_PATTERN.search(prompt)))
    is_factoid = int(bool(_FACTOID_PATTERN.search(prompt)))

    # --- Cognitive demand ---
    n_sub_questions = n_questions
    n_conditionals = len(_CONDITIONAL_PATTERN.findall(prompt))
    n_negations = len(_NEGATION_PATTERN.findall(prompt))
    n_comparisons = len(_COMPARISON_PATTERN.findall(prompt))
    has_multi_step = int(bool(re.search(
        r"(step\s*\d|first.*then|part\s*[a-d(]|\b[a-d]\))", prompt, re.IGNORECASE
    )))

    # --- Punctuation / formatting density ---
    n_parens = prompt.count("(") + prompt.count(")")
    n_quotes = prompt.count('"') + prompt.count("'")
    special_char_ratio = sum(1 for c in prompt if not c.isalnum() and not c.isspace()) / max(len(prompt), 1)

    return {
        "type_token_ratio": round(type_token_ratio, 4),
        "avg_word_len": round(avg_word_len, 4),
        "rare_word_ratio": round(rare_ratio, 4),
        "n_sentences": n_sentences,
        "n_questions": n_questions,
        "has_code_fence": has_code_fence,
        "has_enumeration": has_enumeration,
        "n_constraints": n_constraints,
        "n_newlines": n_newlines,
        "has_table": has_table,
        "is_math": is_math,
        "is_code": is_code,
        "is_reasoning": is_reasoning,
        "is_creative": is_creative,
        "is_factoid": is_factoid,
        "n_sub_questions": n_sub_questions,
        "n_conditionals": n_conditionals,
        "n_negations": n_negations,
        "n_comparisons": n_comparisons,
        "has_multi_step": has_multi_step,
        "n_parens": n_parens,
        "n_quotes": n_quotes,
        "special_char_ratio": round(special_char_ratio, 4),
    }


# ---------------------------------------------------------------------------
# PCA embedding features
# ---------------------------------------------------------------------------

def load_pca_embeddings(
    prompts: List[str],
    pca_components: int = 15,
) -> Optional[np.ndarray]:
    """Compute PCA-projected embeddings for a list of prompts.

    Uses the production FeatureService to guarantee the same feature space
    as the deployed router.

    Parameters
    ----------
    prompts:
        Raw prompt strings.
    pca_components:
        Expected PCA dimensionality (for validation only).

    Returns
    -------
    np.ndarray or None
        Shape ``(n_prompts, pca_components)`` (bias term excluded).
        ``None`` if the feature service is unavailable.
    """
    try:
        from bandit_gpt.feature_service import FeatureService
        fs = FeatureService()
        vectors = fs.extract_features_batch(prompts)
        return vectors[:, :-1]  # strip bias column
    except Exception as e:
        logger.warning("Could not load PCA embeddings: %s", e)
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_classified_data() -> pd.DataFrame:
    """Load pareto_classified.jsonl into a tidy DataFrame.

    Each row is one prompt with columns for per-arm rewards, costs,
    gaps, and the best-arm label.
    """
    records = []
    with open(CLASSIFIED_PATH) as f:
        for line in f:
            r = json.loads(line)
            row: Dict[str, Any] = {
                "prompt": r["prompt"],
                "source": r.get("source", "unknown"),
                "difficulty": r["difficulty"],
                "reward_spread": r["reward_spread"],
                "best_arm": ARM_SHORT.get(r.get("best_arm", ""), "?"),
            }
            for arm_id in ARM_ORDER:
                short = ARM_SHORT[arm_id]
                info = r["arms"].get(arm_id, {})
                row[f"reward_{short}"] = info.get("reward", np.nan)
                row[f"cost_{short}"] = info.get("cost", 0.0)

            records.append(row)

    df = pd.DataFrame(records)

    # Derived gap columns
    df["gap_large_small"] = df["reward_large"] - df["reward_small"]
    df["gap_medium_small"] = df["reward_medium"] - df["reward_small"]
    df["gap_large_medium"] = df["reward_large"] - df["reward_medium"]

    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_feature_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
) -> pd.DataFrame:
    """Spearman correlations between features and targets.

    Parameters
    ----------
    df:
        DataFrame with feature and target columns.
    feature_cols:
        Column names of features.
    target_cols:
        Column names of targets (rewards, gaps).

    Returns
    -------
    pd.DataFrame
        Shape ``(n_features, n_targets)`` with Spearman rho values.
    """
    results = {}
    for target in target_cols:
        y = df[target].values
        corrs = {}
        for feat in feature_cols:
            x = df[feat].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 30:
                corrs[feat] = np.nan
                continue
            rho, _ = stats.spearmanr(x[mask], y[mask])
            corrs[feat] = round(rho, 4)
        results[target] = corrs

    return pd.DataFrame(results, index=feature_cols)


def compute_mutual_information(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "best_arm",
    n_neighbors: int = 5,
) -> pd.Series:
    """Mutual information between features and a categorical target.

    Parameters
    ----------
    df:
        DataFrame with feature and target columns.
    feature_cols:
        Column names of features.
    target_col:
        Categorical target column.
    n_neighbors:
        KNN neighbors for MI estimation.

    Returns
    -------
    pd.Series
        MI values indexed by feature name.
    """
    X = df[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].values)

    mi = mutual_info_classif(X, y, n_neighbors=n_neighbors, random_state=42,
                              n_jobs=1)
    return pd.Series(mi, index=feature_cols).sort_values(ascending=False)


def train_gap_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "best_arm",
    n_cv: int = 5,
) -> Tuple[GradientBoostingClassifier, np.ndarray, float]:
    """Train a GBT classifier to predict the best arm from prompt features.

    Parameters
    ----------
    df:
        DataFrame with feature and target columns.
    feature_cols:
        Column names of features.
    target_col:
        Target column (categorical).
    n_cv:
        Number of cross-validation folds.

    Returns
    -------
    tuple[GradientBoostingClassifier, np.ndarray, float]
        (fitted model, feature importances, mean CV accuracy).
    """
    X = df[feature_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].values)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )

    cv_scores = cross_val_score(clf, X, y, cv=n_cv, scoring="accuracy", n_jobs=1)
    clf.fit(X, y)

    return clf, clf.feature_importances_, float(cv_scores.mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_cols: List[str],
    importances: np.ndarray,
    title: str,
    save_path: Path,
    top_k: int = 25,
) -> None:
    """Horizontal bar chart of top-K feature importances."""
    idx = np.argsort(importances)[-top_k:]
    fig, ax = plt.subplots(figsize=(8, 0.35 * top_k + 1))
    ax.barh(
        [feature_cols[i] for i in idx],
        importances[idx],
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Importance (GBT split gain)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_correlation_heatmap(
    corr_df: pd.DataFrame,
    title: str,
    save_path: Path,
    top_k: int = 20,
) -> None:
    """Heatmap of top-K features by max absolute correlation."""
    max_abs = corr_df.abs().max(axis=1)
    top_features = max_abs.nlargest(top_k).index
    sub = corr_df.loc[top_features]

    fig, ax = plt.subplots(figsize=(10, 0.45 * top_k + 1.5))
    im = ax.imshow(sub.values, aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(sub.columns)))
    ax.set_xticklabels(sub.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(sub.index)))
    ax.set_yticklabels(sub.index, fontsize=9)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            val = sub.values[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Spearman ρ")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_gap_distributions_by_task(
    df: pd.DataFrame,
    save_path: Path,
) -> None:
    """Box plots of gap(large−small) conditioned on task-type indicators."""
    task_features = ["is_math", "is_code", "is_reasoning", "is_creative", "is_factoid"]
    present = [f for f in task_features if f in df.columns]

    fig, axes = plt.subplots(1, len(present), figsize=(3.5 * len(present), 4), sharey=True)
    if len(present) == 1:
        axes = [axes]

    for ax, feat in zip(axes, present):
        groups = [
            df.loc[df[feat] == 0, "gap_large_small"].dropna().values,
            df.loc[df[feat] == 1, "gap_large_small"].dropna().values,
        ]
        bp = ax.boxplot(groups, labels=["No", "Yes"], widths=0.5, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#BBDEFB", "#E57373"]):
            patch.set_facecolor(color)
        ax.set_title(feat.replace("is_", "").title(), fontsize=11)
        ax.set_ylabel("Gap (large − small)" if ax == axes[0] else "")
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)

    fig.suptitle("Reward gap (large − small) by task type", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_best_arm_by_feature(
    df: pd.DataFrame,
    feature: str,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """Stacked bar chart showing best-arm distribution across feature quantiles."""
    finite = df[df[feature].notna()].copy()
    if finite[feature].nunique() <= 2:
        finite["bin"] = finite[feature].astype(str)
    else:
        finite["bin"] = pd.qcut(finite[feature], q=n_bins, duplicates="drop")
        finite["bin"] = finite["bin"].astype(str)

    pivot = finite.groupby(["bin", "best_arm"]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    arm_colors = {"small": "#66BB6A", "medium": "#42A5F5", "large": "#EF5350"}
    ordered = [c for c in ["small", "medium", "large"] if c in pivot_pct.columns]

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot_pct[ordered].plot.bar(stacked=True, ax=ax,
                                 color=[arm_colors[c] for c in ordered],
                                 edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Fraction best arm")
    ax.set_xlabel(feature)
    ax.set_title(f"Best arm distribution across {feature}")
    ax.legend(title="Best arm", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_reward_curves_by_feature(
    df: pd.DataFrame,
    feature: str,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """Line plot of mean reward per arm across feature quantiles."""
    finite = df[df[feature].notna()].copy()
    if finite[feature].nunique() <= 3:
        finite["bin"] = finite[feature]
        bin_order = sorted(finite["bin"].unique())
    else:
        finite["bin"] = pd.qcut(finite[feature], q=n_bins, duplicates="drop")
        bin_order = sorted(finite["bin"].unique())
        finite["bin_mid"] = finite["bin"].apply(lambda x: x.mid if hasattr(x, "mid") else x)

    arm_colors = {"small": "#66BB6A", "medium": "#42A5F5", "large": "#EF5350"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for arm_label, short in ARM_SHORT.items():
        col = f"reward_{short}"
        means = finite.groupby("bin")[col].mean()
        means = means.reindex(bin_order)
        ax.plot(range(len(means)), means.values, "o-",
                label=short, color=arm_colors[short], linewidth=2, markersize=5)

    ax.set_xticks(range(len(bin_order)))
    labels = [str(b)[:15] for b in bin_order]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean reward")
    ax.set_xlabel(feature)
    ax.set_title(f"Per-arm reward across {feature} quantiles")
    ax.legend(title="Model", loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading classified data from %s", CLASSIFIED_PATH)
    df = load_classified_data()
    logger.info("  Loaded %d prompts", len(df))

    # ------------------------------------------------------------------
    # Extract text features
    # ------------------------------------------------------------------
    logger.info("Extracting text features...")
    text_feats = df["prompt"].apply(extract_text_features)
    feat_df = pd.DataFrame(text_feats.tolist())
    text_feature_cols = list(feat_df.columns)
    df = pd.concat([df, feat_df], axis=1)

    # ------------------------------------------------------------------
    # PCA embedding features (cached to avoid repeated encoding)
    # ------------------------------------------------------------------
    pca_cache_path = RESULTS_DIR / "_pca_embeddings_cache.npy"
    if pca_cache_path.exists():
        logger.info("Loading cached PCA embeddings from %s", pca_cache_path)
        pca_matrix: Optional[np.ndarray] = np.load(pca_cache_path)
    else:
        logger.info("Computing PCA embeddings (this takes ~2 min)...")
        pca_matrix = load_pca_embeddings(df["prompt"].tolist())
        if pca_matrix is not None:
            np.save(pca_cache_path, pca_matrix)
            logger.info("  Cached to %s", pca_cache_path)

    pca_feature_cols: List[str] = []
    if pca_matrix is not None:
        n_pca = pca_matrix.shape[1]
        pca_feature_cols = [f"pca_{i}" for i in range(n_pca)]
        for i, col in enumerate(pca_feature_cols):
            df[col] = pca_matrix[:, i]
        logger.info("  Added %d PCA features", n_pca)
    else:
        logger.warning("  PCA embeddings unavailable; skipping PCA features")

    all_feature_cols = text_feature_cols + pca_feature_cols

    # ------------------------------------------------------------------
    # 1. Spearman correlations
    # ------------------------------------------------------------------
    logger.info("\n=== Spearman correlations ===")
    target_cols = [
        "reward_small", "reward_medium", "reward_large",
        "gap_large_small", "gap_medium_small", "gap_large_medium",
        "reward_spread",
    ]
    corr_df = compute_feature_correlations(df, all_feature_cols, target_cols)

    print("\n--- Top features by |correlation| with gap_large_small ---")
    gap_corr = corr_df["gap_large_small"].abs().sort_values(ascending=False)
    for feat, val in gap_corr.head(20).items():
        sign = "+" if corr_df.loc[feat, "gap_large_small"] > 0 else "-"
        print(f"  {feat:30s}  {sign}{val:.4f}")

    print("\n--- Top features by |correlation| with gap_medium_small ---")
    gap_corr2 = corr_df["gap_medium_small"].abs().sort_values(ascending=False)
    for feat, val in gap_corr2.head(20).items():
        sign = "+" if corr_df.loc[feat, "gap_medium_small"] > 0 else "-"
        print(f"  {feat:30s}  {sign}{val:.4f}")

    print("\n--- Top features by |correlation| with reward_spread ---")
    spread_corr = corr_df["reward_spread"].abs().sort_values(ascending=False)
    for feat, val in spread_corr.head(20).items():
        sign = "+" if corr_df.loc[feat, "reward_spread"] > 0 else "-"
        print(f"  {feat:30s}  {sign}{val:.4f}")

    # ------------------------------------------------------------------
    # 2. Mutual information with best arm
    # ------------------------------------------------------------------
    logger.info("\n=== Mutual information with best_arm ===")
    mi = compute_mutual_information(df, all_feature_cols, "best_arm")
    print("\n--- Top features by MI with best_arm ---")
    for feat, val in mi.head(20).items():
        print(f"  {feat:30s}  {val:.4f}")

    # ------------------------------------------------------------------
    # 3. GBT classifier: predict best arm
    # ------------------------------------------------------------------
    logger.info("\n=== GBT best-arm classifier ===")

    logger.info("  Training text-only classifier...")
    clf_text, imp_text, acc_text = train_gap_classifier(
        df, text_feature_cols, "best_arm",
    )
    print(f"\n  Text features only:  CV accuracy = {acc_text:.4f}  "
          f"(chance = {1/3:.4f})")

    acc_pca = None
    if pca_feature_cols:
        logger.info("  Training PCA-only classifier...")
        clf_pca, imp_pca, acc_pca = train_gap_classifier(
            df, pca_feature_cols, "best_arm",
        )
        print(f"  PCA features only:   CV accuracy = {acc_pca:.4f}")

    logger.info("  Training all-features classifier...")
    clf_all, imp_all, acc_all = train_gap_classifier(
        df, all_feature_cols, "best_arm",
    )
    print(f"  All features:        CV accuracy = {acc_all:.4f}")

    # ------------------------------------------------------------------
    # 4. GBT regressor: predict gap(large − small)
    # ------------------------------------------------------------------
    logger.info("\n=== GBT gap regressor (large − small) ===")
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score as cvs

    X_all = df[all_feature_cols].values.astype(np.float64)
    X_all = np.nan_to_num(X_all, nan=0.0)
    y_gap = df["gap_large_small"].values

    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42,
    )
    gap_cv = cvs(reg, X_all, y_gap, cv=5, scoring="r2", n_jobs=1)
    reg.fit(X_all, y_gap)
    print(f"\n  Gap(large−small) R² = {gap_cv.mean():.4f} ± {gap_cv.std():.4f}")

    print("\n--- Top features for gap prediction (regressor importance) ---")
    gap_imp = pd.Series(reg.feature_importances_, index=all_feature_cols)
    for feat, val in gap_imp.sort_values(ascending=False).head(20).items():
        print(f"  {feat:30s}  {val:.4f}")

    # ------------------------------------------------------------------
    # 5. Per-source analysis: which benchmarks favor which arm?
    # ------------------------------------------------------------------
    logger.info("\n=== Per-source arm preference ===")
    source_stats = df.groupby("source").agg(
        n=("prompt", "size"),
        mean_small=("reward_small", "mean"),
        mean_medium=("reward_medium", "mean"),
        mean_large=("reward_large", "mean"),
        mean_gap_ls=("gap_large_small", "mean"),
        best_is_small=("best_arm", lambda x: (x == "small").mean()),
        best_is_medium=("best_arm", lambda x: (x == "medium").mean()),
        best_is_large=("best_arm", lambda x: (x == "large").mean()),
    ).sort_values("mean_gap_ls", ascending=False)
    print("\n" + source_stats.head(20).to_string())

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    logger.info("\n=== Generating plots ===")

    plot_correlation_heatmap(
        corr_df, "Feature–reward/gap Spearman correlations",
        RESULTS_DIR / "correlation_heatmap.png",
    )

    plot_feature_importance(
        all_feature_cols, imp_all,
        "Feature importance: best-arm classifier (all features)",
        RESULTS_DIR / "feature_importance_best_arm.png",
    )

    plot_feature_importance(
        all_feature_cols, reg.feature_importances_,
        "Feature importance: gap(large−small) regressor",
        RESULTS_DIR / "feature_importance_gap.png",
    )

    plot_gap_distributions_by_task(df, RESULTS_DIR / "gap_by_task_type.png")

    # Reward curves for top discriminative text features
    top_text_feats = gap_imp[text_feature_cols].sort_values(ascending=False).head(5).index
    for feat in top_text_feats:
        plot_reward_curves_by_feature(
            df, feat, RESULTS_DIR / f"reward_curves_{feat}.png",
        )

    # Best-arm distribution for top features
    for feat in list(top_text_feats[:3]) + ["is_math", "is_code"]:
        if feat in df.columns:
            plot_best_arm_by_feature(
                df, feat, RESULTS_DIR / f"best_arm_dist_{feat}.png",
            )

    # ------------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------------
    summary = {
        "n_prompts": len(df),
        "n_text_features": len(text_feature_cols),
        "n_pca_features": len(pca_feature_cols),
        "best_arm_accuracy": {
            "text_only": round(acc_text, 4),
            "pca_only": round(acc_pca, 4) if pca_feature_cols else None,
            "all_features": round(acc_all, 4),
            "chance": round(1/3, 4),
        },
        "gap_regression_r2": round(gap_cv.mean(), 4),
        "top_features_by_gap_importance": {
            feat: round(val, 4)
            for feat, val in gap_imp.sort_values(ascending=False).head(15).items()
        },
        "top_features_by_mi_best_arm": {
            feat: round(val, 4)
            for feat, val in mi.head(15).items()
        },
        "top_correlations_gap_large_small": {
            feat: round(corr_df.loc[feat, "gap_large_small"], 4)
            for feat in gap_corr.head(15).index
        },
    }
    summary_path = RESULTS_DIR / "eda_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    logger.info("\nDone. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
