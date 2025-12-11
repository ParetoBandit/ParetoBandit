#!/usr/bin/env python3
"""
Test decorrelated model on length artifact cases.

Does removing length correlation fix the 100% failure rate on long non-summarization texts?
"""

import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load decorrelated model
print("Loading decorrelated model...")
with open('../../results/intent_classification/xgboost_intent_classifier_decorrelated.pkl', 'rb') as f:
    checkpoint = pickle.load(f)
    model = checkpoint['model']
    projection = checkpoint['projection']
    labels = checkpoint['labels']

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def apply_decorrelation(embedding, length):
    """Apply same decorrelation used during training."""
    # Normalize length
    L_norm = (length - projection['length_mean']) / projection['length_std']
    
    # Predict length component
    length_component = projection['ridge'].predict([[L_norm]])[0]
    
    # Remove length component
    return embedding - length_component

def predict_decorrelated(prompt):
    """Predict with decorrelated embeddings."""
    # Get raw embedding
    embedding = embedder.encode([prompt], convert_to_numpy=True)[0]
    length = len(prompt)
    
    # Apply decorrelation
    embedding_clean = apply_decorrelation(embedding, length)
    
    # Predict
    pred = model.predict([embedding_clean])[0]
    probs = model.predict_proba([embedding_clean])[0]
    
    predicted_label = labels[int(pred)]
    confidence = float(probs[int(pred)])
    
    return {
        'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
        'length': length,
        'predicted': predicted_label,
        'confidence': confidence
    }

# ============================================================================
# Same test cases from length_artifact test
# ============================================================================

python_error_log = """
Traceback (most recent call last):
  File "app.py", line 45, in process_data
    result = calculate_metrics(data)
  File "metrics.py", line 123, in calculate_metrics
    return sum(values) / len(values)
ZeroDivisionError: division by zero

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "main.py", line 78, in run_pipeline
    output = process_data(input_data)
  File "app.py", line 48, in process_data
    logger.error(f"Failed to process: {e}")
NameError: name 'logger' is not defined

Additional context:
- Input data length: 0
- Expected non-empty array
- This error occurs intermittently with sparse datasets
- Stack trace suggests multiple issues:
  1. Division by zero when data is empty
  2. Logger not initialized properly
  3. Missing error handling in calculate_metrics

Suggested fixes:
- Add check for empty data before division
- Initialize logger at module level
- Add proper exception handling
- Validate input data before processing

This has been failing in production for the last 3 hours.
Multiple retries have not resolved the issue.
"""

email_thread = """
From: Sarah Johnson <sarah@company.com>
To: Team <team@company.com>
Subject: Re: Re: Re: Project timeline update

Hi everyone,

Following up on yesterday's discussion about the Q4 deliverables.

I've reviewed the timeline and have a few concerns:

1. The API integration is scheduled for week 3, but we haven't finalized the schema yet.
2. QA testing is only allocated 2 days - this seems insufficient given past projects.
3. No buffer time for unexpected issues.

From: Mike Chen <mike@company.com>
Agree with Sarah. We should add at least a week of buffer. Last quarter we went 3 weeks over schedule because of unexpected bugs in the authentication layer.

From: Lisa Park <lisa@company.com>
Good points. However, we're under pressure from leadership to ship by Nov 15. Can we parallelize some tasks? Maybe start QA earlier with partial implementations?

From: Sarah Johnson <sarah@company.com>
Parallel testing might work, but it adds coordination overhead. Let's schedule a call tomorrow at 2pm to discuss. I'll put together some alternative timelines.

Please review the attached Gantt chart before the meeting.
"""

meeting_notes = """
Meeting: Sprint Planning
Date: October 15, 2024
Attendees: Engineering team (12 people)
Duration: 90 minutes

Agenda:
1. Review last sprint's velocity
2. Plan upcoming sprint stories
3. Address technical debt
4. Discuss infrastructure upgrades

Key Discussion Points:

Sarah: Last sprint we completed 45 story points. That's below our target of 55. Main blockers were the database migration issues.

Mike: The migration took 3 days instead of 1. We need better estimation for infrastructure work.

Lisa: Agree. Proposal: Create separate tracking for infra vs feature work.

Team: Discussed pros/cons. Decision: Will trial for next 2 sprints.

Story Planning:
- User authentication refactor: 13 points (assigned to Mike's team)
- Payment integration: 8 points (assigned to Sarah)
- Performance optimization: 5 points (assigned to Lisa)
- Bug fixes: 8 points (unassigned, will distribute)

Technical Debt:
- Need to upgrade PostgreSQL from 12 to 15
- Migrate from REST to GraphQL for mobile APIs
- Update testing framework
- Decision: Allocate 20% of sprint capacity to debt

Action Items:
- Sarah: Draft migration plan by Friday
- Mike: Research GraphQL libraries
- Lisa: Set up performance monitoring
- Everyone: Review and comment on technical debt proposals

Next meeting: October 22, same time.
"""

code_documentation = """
# Data Processing Pipeline Documentation

## Overview
This module implements a distributed data processing pipeline for large-scale analytics.

## Architecture

### Components

1. **Ingestion Layer**
   - Reads from Kafka topics
   - Validates incoming data schemas
   - Routes to appropriate processors
   
   Configuration:
   - kafka.bootstrap.servers: localhost:9092
   - kafka.topic: data-ingest
   - schema.registry.url: http://localhost:8081

2. **Processing Layer**
   - Implements map-reduce patterns
   - Handles data transformations
   - Manages state checkpoints
   
   Key classes:
   - DataProcessor: Main processing logic
   - StateManager: Handles checkpointing
   - MetricsCollector: Performance monitoring

3. **Output Layer**
   - Writes to various sinks (S3, Database, Cache)
   - Handles partitioning and compression
   - Manages error handling and retries

## Usage Example

```python
from pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline(
    input_topic='raw-data',
    output_path='s3://bucket/processed/',
    checkpoint_interval=100
)

# Configure processors
pipeline.add_processor('validate', ValidationProcessor())
pipeline.add_processor('transform', TransformationProcessor())
pipeline.add_processor('aggregate', AggregationProcessor())

# Run pipeline
pipeline.start()
```

## Performance Characteristics
- Throughput: 10K events/second
- Latency: P50=50ms, P99=200ms
- Memory: 2GB baseline + 100MB per processor

## Error Handling
Pipeline implements at-least-once semantics. Failures trigger:
1. Local retry (max 3 attempts)
2. Dead letter queue routing
3. Alert notification

## Monitoring
Metrics exposed via Prometheus:
- pipeline_events_processed_total
- pipeline_processing_duration_seconds
- pipeline_errors_total
"""

test_cases = {
    "Long Python Error Log (CODING expected)": python_error_log,
    "Long Email Thread (GENERAL expected)": email_thread,
    "Long Meeting Notes (GENERAL expected)": meeting_notes,
    "Long Code Documentation (CODING expected)": code_documentation,
}

# ============================================================================
# Run Tests
# ============================================================================

print("\n" + "="*80)
print("DECORRELATED MODEL: Length Artifact Test")
print("="*80)
print("\nDoes orthogonal projection fix the length artifact?\n")

failures = 0
for category, prompt in test_cases.items():
    result = predict_decorrelated(prompt)
    
    if result['predicted'] == 'summarization':
        failures += 1
        indicator = "❌ STILL FAILS"
    else:
        indicator = "✅ FIXED"
    
    print(f"{indicator} {category}")
    print(f"   Length: {result['length']} chars")
    print(f"   Predicted: {result['predicted']} ({result['confidence']:.1%})")
    print()

# Summary
print("="*80)
print("RESULTS")
print("="*80)
print(f"\nTotal tests: 4")
print(f"Still failing (→ SUMMARIZATION): {failures}")
print(f"Fixed (→ correct class): {4 - failures}")
print(f"\nFailure rate: {failures/4*100:.0f}% (was 100% with original model)")

if failures == 0:
    print("\n✅ COMPLETE FIX: Decorrelation eliminated length artifact!")
elif failures < 2:
    print(f"\n🟡 PARTIAL FIX: {(4-failures)/4*100:.0f}% improvement")
else:
    print(f"\n❌ MINIMAL FIX: Only {(4-failures)/4*100:.0f}% improvement")

print("\nTrade-off Analysis:")
print(f"  Original model: 94.5% accuracy, 100% length artifact failures")
print(f"  Decorrelated:   88.1% accuracy, {failures/4*100:.0f}% length artifact failures")
print(f"  Accuracy cost:  {94.5 - 88.1:.1f}% for {100 - failures/4*100:.0f}% artifact fix")
