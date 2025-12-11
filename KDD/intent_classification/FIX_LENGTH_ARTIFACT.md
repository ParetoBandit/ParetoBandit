# Fixing the Length Artifact: Practical Solutions (No Synthetic Data)

## Problem Recap

**Current Issue**: Model learned "length >1000 chars" → SUMMARIZATION because CNN/DailyMail is the only class with long texts.

**Current Distribution**:
```
CODING:         Mean 216 chars  (max ~400)
REASONING:      Mean 242 chars  (max ~350)
FACTUAL_QA:     Mean 46 chars   (max ~100)
GENERAL:        Mean 86 chars   (max ~200)
SUMMARIZATION:  Mean 1017 chars (max ~2000)  ← PROBLEM
```

**Goal**: Balance length distributions so model can't use length as a shortcut.

---

## Solution 1: Collect Long Examples for Other Classes (RECOMMENDED)

### Strategy
Add long real-world examples to CODING, REASONING, FACTUAL_QA, and GENERAL so length is no longer discriminative.

### A. Long CODING Examples

**Sources** (all real, public data):

1. **GitHub README Files** (many are 1000+ chars)
   - Dataset: GitHub repos with documentation
   - Example: "Explain this README: [full README text]"
   - Label: CODING (user wants code explanation/help)
   - Collection: Use GitHub API to fetch popular repo READMEs

2. **Stack Overflow Long Questions** (with code blocks)
   - Dataset: Stack Overflow data dump
   - Filter: Questions with >1000 chars including code
   - Example: "[Long problem description] [Stack trace] [Code attempt]"
   - Label: CODING

3. **Technical Documentation**
   - Dataset: Python docs, React docs, PostgreSQL docs
   - Example: User pastes doc section asking for clarification
   - Label: CODING
   - Collection: Official documentation from popular libraries

4. **Error Logs & Stack Traces**
   - Dataset: Real error logs from open source projects
   - Example: "Why am I getting this error? [long stack trace]"
   - Label: CODING
   - Collection: GitHub issues with error logs

**Implementation**:
```python
def collect_long_coding_samples(n=500):
    samples = []
    
    # GitHub READMEs
    for repo in get_popular_repos(limit=200):
        readme = fetch_readme(repo)
        if 1000 < len(readme) < 2000:
            samples.append({
                'prompt': f"Explain this project: {readme}",
                'intent_label': 'coding',
                'source': 'github_readme',
                'length': len(readme)
            })
    
    # Stack Overflow long questions
    for question in fetch_stackoverflow_questions(min_length=1000):
        if contains_code(question):
            samples.append({
                'prompt': question['body'],
                'intent_label': 'coding',
                'source': 'stackoverflow',
                'length': len(question['body'])
            })
    
    return samples[:n]
```

### B. Long GENERAL Examples

**Sources**:

1. **Email Datasets**
   - Dataset: Enron Email Corpus (public)
   - Example: "What should I respond? [full email thread]"
   - Label: GENERAL
   - Note: Filter for non-technical emails

2. **Reddit Posts** (long self-posts)
   - Dataset: Reddit data dumps (r/relationship_advice, r/AmItheAsshole)
   - Example: Long personal stories asking for advice
   - Label: GENERAL
   - Collection: Posts with >1000 chars from conversational subreddits

3. **Meeting Transcripts**
   - Dataset: AMI Meeting Corpus (public)
   - Example: "Summarize the key decisions: [transcript]" 
   - Wait... that's summarization
   - Better: "What was agreed about X? [transcript]"
   - Label: FACTUAL_QA or GENERAL (specific question, not summarization)

4. **Chat Logs**
   - Dataset: IRC/Slack public archives
   - Example: Long technical discussion
   - Label: GENERAL or CODING depending on content

**Implementation**:
```python
def collect_long_general_samples(n=500):
    samples = []
    
    # Enron emails (non-technical)
    for email in load_enron_corpus():
        if 1000 < len(email['body']) < 2000:
            if not is_technical(email['body']):
                samples.append({
                    'prompt': f"Help me respond to this: {email['body']}",
                    'intent_label': 'general',
                    'source': 'enron_email',
                    'length': len(email['body'])
                })
    
    # Reddit long posts
    for post in fetch_reddit_posts(subreddits=['relationship_advice'], min_length=1000):
        samples.append({
            'prompt': post['selftext'],
            'intent_label': 'general',
            'source': 'reddit',
            'length': len(post['selftext'])
        })
    
    return samples[:n]
```

### C. Long REASONING Examples

**Sources**:

1. **Mathematical Proofs**
   - Dataset: ProofWiki, MathOverflow
   - Example: Long proof with question about a step
   - Label: REASONING

2. **Logic Puzzles with Context**
   - Dataset: Puzzle books, online puzzle collections
   - Example: Long scenario description with reasoning question
   - Label: REASONING

3. **Scientific Reasoning**
   - Dataset: Science QA forums
   - Example: "Why does this experiment show X? [long description]"
   - Label: REASONING

**Challenges**: Harder to find naturally long reasoning examples (most are concise)

**Alternative**: Accept that REASONING is typically short - focus on CODING and GENERAL

### D. Long FACTUAL_QA Examples

**Sources**:

1. **ELI5 (Explain Like I'm 5)**
   - Dataset: Reddit ELI5 or HuggingFace ELI5 dataset
   - Example: Long explanation request about complex topic
   - Label: FACTUAL_QA
   - Note: Questions tend to be short, but can include context

2. **Quora Long Questions**
   - Dataset: Quora Question Pairs
   - Example: Questions with extensive background context
   - Label: FACTUAL_QA

---

## Solution 2: Collect Summarization Requests (Not Full Articles)

### Strategy
Instead of full CNN/DailyMail articles, collect REQUESTS to summarize that are shorter.

### Current Problem
```json
{
  "prompt": "[Full 1500-char news article embedded]",
  "intent_label": "summarization"
}
```

### Better Approach
```json
{
  "prompt": "Summarize this article about climate change: [article]",
  "intent_label": "summarization"
}
```

**But**: This doesn't solve the problem - still long!

### Alternative: Split Intent Detection from Content

**Two-stage approach**:
1. **Intent Detection**: "Summarize this article" (short, ~50 chars)
2. **Content Classification**: What to summarize (if needed)

**Implementation**:
```python
# Training data format
{
  "prompt": "Summarize this article",  # Just the request
  "intent_label": "summarization",
  "has_content": False
}

# vs

{
  "prompt": "Write a function to sort a list",  # Request
  "intent_label": "coding",
  "has_content": False
}
```

**Problem**: Most users embed content in their request. Need real examples of both.

---

## Solution 3: Explicitly Add Length-Balanced Data Collection

### Implementation Script

```python
#!/usr/bin/env python3
"""
Collect length-balanced intent data to fix summarization artifact.

Strategy:
1. Collect long (>1000 char) examples for CODING, GENERAL
2. Ensure similar length distributions across all classes
3. Validate no single class has unique length profile
"""

from datasets import load_dataset
import re

TARGET_DISTRIBUTION = {
    'coding': {
        'short': 250,  # <500 chars (existing)
        'long': 250    # >1000 chars (new)
    },
    'reasoning': {
        'short': 450,  # Mostly short is fine
        'long': 50     # Some long examples
    },
    'factual_qa': {
        'short': 450,
        'long': 50
    },
    'general': {
        'short': 250,
        'long': 250    # Balance with coding
    },
    'summarization': {
        'short': 100,  # Add some short requests
        'long': 400    # Keep some long
    }
}

def collect_long_coding():
    """Collect long coding examples."""
    samples = []
    
    # 1. GitHub READMEs
    print("Collecting GitHub READMEs...")
    # Use BigCode dataset or GitHub API
    dataset = load_dataset("bigcode/the-stack", split="train", streaming=True)
    
    for item in dataset:
        if item['language'] == 'Markdown':
            readme = item['content']
            if 1000 < len(readme) < 2000:
                # Frame as a question
                samples.append({
                    'prompt': f"Explain what this code does:\n\n{readme}",
                    'intent_label': 'coding',
                    'source': 'bigcode_readme'
                })
                
                if len(samples) >= 100:
                    break
    
    # 2. Stack Overflow
    print("Collecting Stack Overflow long questions...")
    dataset = load_dataset("koutch/stackoverflow_python", split="train")
    
    for item in dataset:
        question = item['question_body']
        if 1000 < len(question) < 2000:
            samples.append({
                'prompt': question,
                'intent_label': 'coding',
                'source': 'stackoverflow'
            })
            
            if len(samples) >= 200:
                break
    
    # 3. Python Documentation
    print("Collecting documentation...")
    # Scrape official docs or use existing datasets
    
    return samples[:250]

def collect_long_general():
    """Collect long general examples."""
    samples = []
    
    # 1. Enron Emails
    print("Collecting Enron emails...")
    dataset = load_dataset("enron_emails", split="train")
    
    for item in dataset:
        email_body = item['message']
        # Extract body (remove headers)
        body = extract_body(email_body)
        
        if 1000 < len(body) < 2000:
            # Filter technical
            if not is_technical_content(body):
                samples.append({
                    'prompt': f"Help me respond to: {body}",
                    'intent_label': 'general',
                    'source': 'enron'
                })
                
                if len(samples) >= 150:
                    break
    
    # 2. Reddit posts
    print("Collecting Reddit posts...")
    # Use PushShift or Reddit API
    subreddits = ['relationship_advice', 'AskReddit', 'NoStupidQuestions']
    
    for subreddit in subreddits:
        posts = fetch_reddit_posts(subreddit, min_length=1000, max_length=2000)
        for post in posts[:50]:
            samples.append({
                'prompt': post['selftext'],
                'intent_label': 'general',
                'source': f'reddit_{subreddit}'
            })
    
    return samples[:250]

def is_technical_content(text):
    """Check if text contains technical jargon."""
    technical_indicators = [
        r'\bAPI\b', r'\bJSON\b', r'\bHTTP\b', r'\bSQL\b',
        r'function\s*\(', r'class\s+\w+', r'import\s+\w+',
        r'\bdef\b', r'\bvar\b', r'\bconst\b'
    ]
    
    for pattern in technical_indicators:
        if re.search(pattern, text):
            return True
    return False

def main():
    print("="*80)
    print("Collecting Length-Balanced Intent Data")
    print("="*80)
    
    # Collect long examples
    long_coding = collect_long_coding()
    long_general = collect_long_general()
    
    # Load existing data
    with open('data/real_intent_prompts_labeled.json') as f:
        existing = json.load(f)
    
    # Merge
    new_samples = existing['samples'] + long_coding + long_general
    
    # Validate length distributions
    print("\nLength Distribution After Balancing:")
    for intent in ['coding', 'general', 'summarization']:
        intent_samples = [s for s in new_samples if s['intent_label'] == intent]
        lengths = [len(s['prompt']) for s in intent_samples]
        
        print(f"{intent}:")
        print(f"  Mean: {np.mean(lengths):.0f} chars")
        print(f"  >1000 chars: {sum(1 for l in lengths if l > 1000)}/{len(lengths)}")
    
    # Save
    with open('data/real_intent_prompts_labeled_balanced.json', 'w') as f:
        json.dump({'samples': new_samples}, f, indent=2)
    
    print("\n✅ Balanced dataset created!")
```

---

## Solution 4: Architectural Changes

### A. Explicit Length Normalization

**Idea**: Normalize embeddings by prompt length to make length-invariant features.

**Problem**: May hurt legitimate length-based signals (e.g., FACTUAL_QA are genuinely shorter)

### B. Multi-Task Learning

Train model to predict:
1. Intent (5-class)
2. Content type (code, prose, mixed)
3. Length bucket (<500, 500-1000, >1000)

**Benefit**: Forces model to separate concerns

**Implementation**:
```python
# Multi-task XGBoost
model = MultiOutputClassifier(
    XGBClassifier(...)
)

# Train on multiple targets
y_intent = [...] # Main task
y_length_bucket = [...]  # Auxiliary task
y_content_type = [...]   # Auxiliary task

model.fit(X, [y_intent, y_length_bucket, y_content_type])
```

### C. Adversarial Debiasing

**Idea**: Train classifier to predict intent while making it HARD to predict length from embeddings.

**Implementation**:
```python
# Two models
intent_classifier = XGBClassifier()  # Predict intent
length_predictor = XGBClassifier()   # Predict length bucket

# Train intent classifier
# Then check: Can length_predictor predict length from embeddings?
# If yes, penalize features that enable it
```

**Challenge**: XGBoost doesn't have built-in adversarial training

---

## Solution 5: Production Hybrid (Short-term Fix)

While collecting better data, deploy hybrid system:

```python
class RobustIntentClassifier:
    def __init__(self, semantic_model, length_threshold=800):
        self.semantic_model = semantic_model
        self.length_threshold = length_threshold
        
    def predict(self, prompt):
        length = len(prompt)
        
        # Short prompts: Use semantic model safely
        if length < self.length_threshold:
            return self.semantic_model.predict(prompt)
        
        # Long prompts: Additional checks
        else:
            semantic_pred = self.semantic_model.predict(prompt)
            
            # If predicted summarization, verify
            if semantic_pred == 'summarization':
                # Check for explicit markers
                markers = ['summarize', 'TLDR', 'tl;dr', 'summary of', 
                          'brief overview', 'key points']
                
                has_marker = any(m in prompt.lower() for m in markers)
                
                if has_marker:
                    return 'summarization'
                else:
                    # Length artifact suspected, try secondary classification
                    # Remove length bias by comparing to other high-prob classes
                    probs = self.semantic_model.predict_proba(prompt)
                    
                    # Penalize summarization for long texts without markers
                    probs['summarization'] *= 0.2
                    
                    return max(probs, key=probs.get)
            
            else:
                return semantic_pred
```

---

## Recommended Implementation Plan

### Phase 1: Data Collection (2-3 weeks)

**Week 1**: Collect long CODING examples
- GitHub READMEs: 150 samples
- Stack Overflow: 100 samples  
- Documentation: 50 samples
- **Target**: 300 long coding samples (>1000 chars)

**Week 2**: Collect long GENERAL examples
- Enron emails: 150 samples
- Reddit posts: 100 samples
- Meeting transcripts: 50 samples
- **Target**: 300 long general samples (>1000 chars)

**Week 3**: Validation and cleaning
- Remove duplicates
- Validate labels
- Check length distributions
- Test for data leakage

### Phase 2: Retrain and Validate (1 week)

**Day 1-2**: Retrain model with balanced data
- Add new long samples to training set
- Ensure stratified CV by both label AND length

**Day 3-4**: Re-run all tests
- Section 4.4: Wild prompts (should still pass)
- Section 4.5: Shortcut learning (should still pass)
- Section 4.6: Length artifact (should NOW pass!)

**Day 5**: Production validation
- Test on real user prompts (if available)
- Compare to hybrid baseline

### Phase 3: Paper Update (1 week)

Update sections:
- Section 2.2: Add "length-balanced collection" to methodology
- Section 4.6: Change from "Critical Limitation" to "Addressed Limitation"
- Section 6.2: Update limitations to reflect fix

---

## Expected Results After Fix

### Before (Current)
```
Length Artifact Test:
  Long coding text → SUMMARIZATION (4/4 failures)
  Long general text → SUMMARIZATION (100% failure rate)
  
Overall: Length is discriminative feature
```

### After (With Balanced Data)
```
Length Artifact Test:
  Long coding text → CODING (0/4 failures expected)
  Long general text → GENERAL (0/4 failures expected)
  
Overall: Length no longer discriminative
       Model relies on semantic content
```

### Validation Metrics

**Must achieve**:
- <20% failure rate on long non-summarization texts
- Maintain >90% accuracy on original test set
- No degradation on short prompts

**Stretch goals**:
- 0% failure rate on length artifact test
- >94% accuracy maintained
- Improved confidence calibration

---

## Datasets to Use (All Public, No Synthetic)

| Dataset | Source | Use Case | Size |
|---------|--------|----------|------|
| BigCode/The Stack | HuggingFace | Long coding (READMEs) | 6TB+ |
| Stack Overflow | Kaggle/HF | Long coding questions | 58M questions |
| Enron Email | HuggingFace | Long general (emails) | 500K emails |
| Reddit PushShift | Archive | Long general (posts) | Billions |
| AMI Meeting | Edinburgh | Long transcripts | 100 hours |
| ProofWiki | Web scrape | Long reasoning (proofs) | 20K+ |
| ELI5 | HuggingFace | Long FACTUAL_QA | 270K |

**All are**:
- ✅ Publicly available
- ✅ Real human-generated
- ✅ No synthetic augmentation
- ✅ Established datasets

---

## Implementation Timeline

**Minimum**: 1 week (prototype with hybrid classifier)
**Recommended**: 4 weeks (full data collection + retrain)
**Ideal**: 8 weeks (+ production validation + paper updates)

**For KDD submission**:
- If timeline permits: Fix and report improved results
- If deadline tight: Report current limitation + proposed fix in future work

---

## Success Criteria

✅ **Length distributions balanced**:
- Each class has samples across <500, 500-1000, >1000 char ranges
- No single class uniquely associated with length bucket

✅ **Length artifact test passes**:
- Long coding text → CODING (not SUMMARIZATION)
- Long general text → GENERAL (not SUMMARIZATION)
- <20% failure rate

✅ **No performance degradation**:
- Original 94.5% accuracy maintained
- Wild prompt tests still pass (Section 4.4)
- Shortcut tests still pass (Section 4.5)

✅ **Production ready**:
- Can deploy without hybrid workarounds
- Handles diverse real-world prompts
- Appropriate confidence calibration

---

## Bottom Line

**The fix is feasible**:
- Real data sources exist (no synthetic needed)
- Data collection is straightforward (APIs, public datasets)
- Expected timeline: 2-4 weeks

**Recommended approach**:
1. **Short-term**: Deploy hybrid classifier (Section 5 solution)
2. **Medium-term**: Collect balanced data and retrain
3. **Long-term**: Multi-task learning for robustness

**For the paper**:
- Report current limitation honestly (done)
- Propose concrete data collection plan (this document)
- If time permits, implement and report improved results
- Either way: demonstrates scientific rigor

The length artifact is fixable with real data - it just requires intentional data collection to balance distributions.
