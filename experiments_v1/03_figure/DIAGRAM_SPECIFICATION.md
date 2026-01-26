# Figure 3: Corralled Architecture Diagram Specification

## Visual Design Overview

**Goal**: Create a clear, publication-quality diagram showing the coordinator-expert hierarchy with information flows.

**Style**: Clean, professional, suitable for academic publication

**Format**: TikZ (LaTeX) or SVG for vector graphics

**Color Scheme**: 
- Coordinator: Blue (#3498db)
- Warmup Expert: Green (#2ecc71)
- Tabula Rasa Expert: Orange (#e67e22)
- Information Flow: Gray arrows
- Data Flow: Black bold arrows

## Layout Structure

```
┌─────────────────────────────────────────────────────────┐
│                    COORDINATOR LAYER                     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Trust Distribution: π = [0.72, 0.28]           │  │
│  │  Cumulative Losses: L = [45.2, 89.7]            │  │
│  │  Learning Rate: η = 0.1                          │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
            │                                    │
            │ Sample                             │ Sample
            │ (p=0.72)                          │ (p=0.28)
            ▼                                    ▼
┌─────────────────────────┐        ┌─────────────────────────┐
│   WARMUP EXPERT         │        │  TABULA RASA EXPERT     │
│  ┌───────────────────┐  │        │  ┌───────────────────┐  │
│  │ A₀ = λI + Prior   │  │        │  │ A₀ = λI           │  │
│  │ b₀ = Prior Data   │  │        │  │ b₀ = 0            │  │
│  │ θ = A⁻¹b          │  │        │  │ θ = A⁻¹b          │  │
│  └───────────────────┘  │        │  └───────────────────┘  │
│                         │        │                         │
│  Recommendation:        │        │  Recommendation:        │
│  model = "gpt-4"        │        │  model = "claude-opus"  │
│  UCB = 0.85             │        │  UCB = 0.78             │
└─────────────────────────┘        └─────────────────────────┘
            │                                    │
            └────────────┬───────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   Selected: "gpt-4"     │
            │   Execute & Observe     │
            │   Reward: r = 0.92      │
            └─────────────────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │   FEEDBACK PHASE        │
            │  Loss = (1-r)/p         │
            │  Update Weights π       │
            │  Update Expert          │
            └─────────────────────────┘
```

## Component Specifications

### Top Layer: Coordinator

**Box Dimensions**: 600px × 120px  
**Position**: Top center  
**Contents**:
- Title: "Coordinator Layer" (bold)
- State variables with current values:
  - π (trust weights vector)
  - L (cumulative losses vector)
  - η (learning rate)
- Sampling probabilities shown on arrows

**Annotations**:
- Left arrow: "Sample expert_0 with p=π₀"
- Right arrow: "Sample expert_1 with p=π₁"

### Middle Layer: Experts

**Layout**: Two boxes side-by-side, 250px × 200px each

#### Left Box: Warmup Expert
**Color**: Light green background (#e8f8f5)  
**Contents**:
- Title: "Warmup Expert" (bold)
- Initialization formulas:
  - A₀ = λI + Σ φ(xᵢ)φ(xᵢ)ᵀ
  - b₀ = Σ rᵢφ(xᵢ)
- Current state:
  - θ (weight vector)
  - Sample count: n = 720
- Recommendation box:
  - Chosen model
  - UCB score

#### Right Box: Tabula Rasa Expert
**Color**: Light orange background (#fef5e7)  
**Contents**:
- Title: "Tabula Rasa Expert" (bold)
- Initialization formulas:
  - A₀ = λI
  - b₀ = 0
- Current state:
  - θ (weight vector)
  - Sample count: n = 280
- Recommendation box:
  - Chosen model
  - UCB score

### Bottom Layer: Execution & Feedback

**Layout**: Two boxes vertically stacked

#### Execution Box
**Dimensions**: 300px × 80px  
**Position**: Center, below experts  
**Contents**:
- Selected model
- Execution indicator
- Observed reward

#### Feedback Box
**Dimensions**: 300px × 100px  
**Position**: Bottom center  
**Contents**:
- Loss calculation: ℓ = (1 - r) / π[i]
- Weight update: π ∝ exp(-ηL)
- Expert update: A, b matrices

## Arrow Specifications

### Arrow Types

1. **Selection Arrows** (Coordinator → Experts)
   - Style: Dashed blue
   - Width: 2pt
   - Label: Sampling probability
   - Direction: Top → Middle

2. **Recommendation Arrows** (Experts → Execution)
   - Style: Solid gray
   - Width: 1.5pt
   - Label: Model + UCB score
   - Direction: Middle → Bottom

3. **Feedback Arrows** (Feedback → Coordinator + Expert)
   - Style: Bold black
   - Width: 3pt
   - Label: Loss, Updated weights
   - Direction: Bottom → Top (bidirectional)

### Information Flow Legend

Include a legend box in the top-right corner:

```
┌─────────────────┐
│ LEGEND          │
├─────────────────┤
│ ─ ─ ─ Selection │
│ ───── Recommend │
│ ═════ Feedback  │
└─────────────────┘
```

## Mathematical Notation

### Symbols to Use

- π (pi): Trust distribution
- θ (theta): Model parameters
- λ (lambda): Regularization
- η (eta): Learning rate
- φ (phi): Feature vector
- Σ (sigma): Summation

### Formatting

- Variables: Italic
- Matrices: Bold italic
- Subscripts: Small font (0.7x)
- Superscripts: Small font (0.7x)

## Example Values for Diagram

Use realistic values from experiments:

**Coordinator**:
- π = [0.72, 0.28]
- L = [45.2, 89.7]
- η = 0.1

**Warmup Expert**:
- Sample count: 720
- Recommendation: gpt-4
- UCB: 0.85

**Tabula Rasa Expert**:
- Sample count: 280
- Recommendation: claude-opus
- UCB: 0.78

**Execution**:
- Selected: gpt-4 (from Warmup)
- Reward: 0.92

**Feedback**:
- Loss: (1 - 0.92) / 0.72 = 0.111
- Updated π: [0.73, 0.27] (slight increase for Warmup)

## TikZ Code Template

```latex
\begin{tikzpicture}[
    node distance=2cm,
    coordinator/.style={rectangle, draw=blue!50, fill=blue!10, thick, minimum width=10cm, minimum height=2cm},
    expert/.style={rectangle, draw=black, thick, minimum width=4cm, minimum height=3cm},
    warmup/.style={expert, fill=green!10},
    tabula/.style={expert, fill=orange!10},
    execution/.style={rectangle, draw=black, thick, minimum width=5cm, minimum height=1.5cm},
    arrow/.style={->, >=stealth, thick}
]

% Coordinator
\node[coordinator] (coord) at (0,0) {
    \begin{tabular}{l}
    \textbf{Coordinator Layer} \\
    Trust: $\pi = [0.72, 0.28]$ \\
    Losses: $L = [45.2, 89.7]$ \\
    Learning Rate: $\eta = 0.1$
    \end{tabular}
};

% Warmup Expert
\node[warmup] (warmup) at (-3,-4) {
    \begin{tabular}{l}
    \textbf{Warmup Expert} \\
    $A_0 = \lambda I + $ Prior \\
    $b_0 = $ Prior Data \\
    Samples: $n=720$
    \end{tabular}
};

% Tabula Rasa Expert
\node[tabula] (tabula) at (3,-4) {
    \begin{tabular}{l}
    \textbf{Tabula Rasa} \\
    $A_0 = \lambda I$ \\
    $b_0 = 0$ \\
    Samples: $n=280$
    \end{tabular}
};

% Execution
\node[execution] (exec) at (0,-7) {
    Selected: gpt-4, Reward: 0.92
};

% Arrows
\draw[arrow, dashed, blue] (coord) -- node[left] {$p=0.72$} (warmup);
\draw[arrow, dashed, blue] (coord) -- node[right] {$p=0.28$} (tabula);
\draw[arrow] (warmup) -- (exec);
\draw[arrow] (tabula) -- (exec);
\draw[arrow, very thick] (exec) -- (coord);

\end{tikzpicture}
```

## Alternative: Sequence Diagram

For showing temporal flow, consider a sequence diagram instead:

```
Time →
  │
  │  ┌─────────────────────────────────────────┐
  │  │ t=1000: Request arrives                 │
  │  └─────────────────────────────────────────┘
  │          │
  ▼          │ get_context()
Coordinator ─┼─→ [Context: x_t]
             │
             │ sample_expert(π)
             └─→ Expert_0 (Warmup)
                    │
                    │ select_model(x_t)
                    └─→ "gpt-4"
                           │
                           │ execute()
                           └─→ Reward: 0.92
                                  │
  ┌─────────────────────────────┘
  │ update()
  │
  │ loss = (1-0.92)/0.72 = 0.111
  │ L[0] ← 45.2 + 0.111 = 45.3
  │ π ← normalize(exp(-0.1 × L))
  │ π = [0.73, 0.27]
  │
  └─→ Expert_0.update(x_t, 0.92)
         │
         │ A ← A + xx^T
         │ b ← b + r×x
         │
         └─→ Ready for next request
```

## Figure Caption Integration

The diagram should support the caption in `figure_2_caption.tex` by visually showing:
1. Three-layer hierarchy (Coordinator, Experts, Execution)
2. Information flows (dashed = selection, solid = recommendation, bold = feedback)
3. Initialization differences (Warmup has priors, Tabula Rasa doesn't)
4. Trust-based selection (probabilistic sampling from π)
5. Unbiased feedback (only selected expert updates)

## Software Tools

**Recommended**:
- TikZ (LaTeX): Best for publication
- draw.io: Easy prototyping
- Inkscape: SVG editing
- Matplotlib: Programmatic generation

**Export Format**:
- PDF (vector): For LaTeX inclusion
- PNG (300 DPI): For presentations
- SVG: For web/slides

## Accessibility Considerations

- Use patterns in addition to colors (for color-blind readers)
- Include textual labels for all components
- Ensure sufficient contrast (WCAG AA standard)
- Provide alt-text description in LaTeX

## Review Checklist

- [ ] All mathematical notation is consistent with paper
- [ ] Arrow directions are clear and unambiguous
- [ ] Color scheme is color-blind friendly
- [ ] Font sizes are readable at 50% zoom
- [ ] Caption matches diagram content
- [ ] Example values are realistic (from actual runs)
- [ ] Legend explains all symbols and arrow types
- [ ] Fits within single column width (or explicitly spans two columns)

