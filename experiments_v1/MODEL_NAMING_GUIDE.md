# Model Naming Style Guide

**Date**: February 13, 2026  
**Purpose**: Ensure consistent model naming across paper, code, and documentation

---

## Canonical Forms

### In Code/Technical Contexts (tables, configs, filenames)

Use full qualified names with hyphens:

- **Mixtral**: `mistralai/mixtral-8x7b-instruct`
- **GPT-4-Turbo**: `openai/gpt-4-turbo`
- **GPT-4o**: `openai/gpt-4o`
- **GPT-5.1**: `openai/gpt-5.1` (synthetic model for experiments)

### In Paper Text (prose, captions)

Use readable capitalized names:

- **Mixtral**: Mixtral-8x7B-Instruct (first mention), then Mixtral
- **GPT-4-Turbo**: GPT-4-Turbo (consistent throughout)
- **GPT-4o**: GPT-4o (consistent throughout)
- **GPT-5.1**: GPT-5.1 (synthetic, explain in footnote)

### In LaTeX Tables

Use abbreviated forms with proper formatting:

```latex
\texttt{mixtral-8x7b-instruct}  % Code font for model identifiers
Mixtral-8x7B                     % Regular text
GPT-4-Turbo                      % Hyphenated
GPT-4o                           % No hyphen after 4
```

---

## Common Mistakes to Avoid

❌ **Don't use**:
- "Mixtral" vs "mixtral" inconsistently
- "GPT-4 Turbo" (space instead of hyphen)
- "gpt4o" (missing hyphen)
- "GPT4-Turbo" (missing hyphen after GPT)
- "Mixtral-8x7b-instruct" (inconsistent capitalization)

✅ **Do use**:
- Consistent capitalization within each context
- Hyphens in "GPT-4-Turbo" (both hyphens)
- Lowercase in code contexts: `gpt-4-turbo`
- Title case in text: GPT-4-Turbo

---

## Context-Specific Guidelines

### In Abstract
Use: Mixtral, GPT-4-Turbo, GPT-4o (no technical details)

### In Introduction
First mention: "Mixtral-8x7B-Instruct (henceforth Mixtral)"  
Subsequent: Mixtral

### In Methods/Experiments
Use full technical names with model family prefix:
- `mistralai/mixtral-8x7b-instruct`
- `openai/gpt-4-turbo`

### In Results/Discussion
Use abbreviated readable forms: Mixtral, GPT-4-Turbo, GPT-4o

### In Figures/Tables
- **Axes labels**: Abbreviated (e.g., "Mixtral")
- **Legends**: Full names (e.g., "Mixtral-8x7B-Instruct")
- **Captions**: First mention full, then abbreviated

---

## Cost References

When mentioning costs, use this format:

- Mixtral: \$0.50/1M tokens
- GPT-4-Turbo: \$10/1M tokens  
- GPT-4o: \$2.50/1M tokens

**In LaTeX**: Use `\$` for dollar signs

---

## Verification Checklist

Before submission, verify:

- [ ] All model names in abstract match style guide
- [ ] Introduction establishes full names, then uses abbreviations
- [ ] Methods section uses technical qualified names
- [ ] Tables use consistent capitalization
- [ ] Figures use consistent naming in legends
- [ ] Code blocks use lowercase qualified names
- [ ] No "GPT4" without hyphen
- [ ] No "GPT-4 Turbo" with space

---

## Quick Reference Table

| Context | Mixtral | GPT-4-Turbo | GPT-4o |
|---------|---------|-------------|--------|
| **Code** | `mistralai/mixtral-8x7b-instruct` | `openai/gpt-4-turbo` | `openai/gpt-4o` |
| **Text (full)** | Mixtral-8x7B-Instruct | GPT-4-Turbo | GPT-4o |
| **Text (abbrev)** | Mixtral | GPT-4-Turbo | GPT-4o |
| **LaTeX Code** | `\texttt{mixtral-8x7b}` | `\texttt{gpt-4-turbo}` | `\texttt{gpt-4o}` |
| **Figure Label** | Mixtral | GPT-4-Turbo | GPT-4o |

---

## Status

- ✅ Guide created: February 13, 2026
- ⏳ To implement: Review and update all experiment READMEs
- ⏳ To verify: Check paper sections for consistency

**Priority**: Medium (consistency improves professionalism)  
**Effort**: 1-2 hours to implement across all files  
**Risk**: Low (cosmetic changes only)

---

**Note**: This guide prioritizes **consistency within each context** over uniform naming across all contexts. Code should look like code, prose should read naturally.
