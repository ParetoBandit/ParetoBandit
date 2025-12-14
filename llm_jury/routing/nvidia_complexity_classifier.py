"""
NVIDIA Prompt Task and Complexity Classifier Integration.

Uses nvidia/prompt-task-and-complexity-classifier from HuggingFace to score prompts
on multiple complexity dimensions:
- creativity_scope (35% weight)
- reasoning (25% weight)  
- constraint_ct (15% weight)
- domain_knowledge (15% weight)
- contextual_knowledge (5% weight)
- number_of_few_shots (5% weight)

Also classifies into 11 task types:
- Brainstorming, Classification, Closed QA, Code Generation, Extraction,
- Math, Open QA, Rewriting, Structured Output, Summarization, Text Generation

Usage:
    from llm_jury.routing.nvidia_complexity_classifier import NvidiaComplexityClassifier
    
    classifier = NvidiaComplexityClassifier()
    
    # Single prompt
    result = classifier.classify("Write a recursive function to solve the N-queens problem")
    print(result['prompt_complexity_score'])  # 0.45
    print(result['reasoning'])                 # 0.6
    print(result['task_type_1'])               # "Code Generation"
    
    # Batch of prompts
    results = classifier.classify_batch([
        "What is 2+2?",
        "Design a distributed cache system with LRU eviction"
    ])
    
    # Filter for complex prompts
    complex_prompts = classifier.filter_by_complexity(prompts, min_score=0.4)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np

# IMPORTANT: Import torch at module level to avoid segfaults on Mac.
# When torch is lazily imported inside a function that loads transformers models,
# it can cause segmentation faults due to library initialization order issues.
# See: https://github.com/pytorch/pytorch/issues/78490
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Lazy load to avoid import overhead at module import time
_model = None
_tokenizer = None


@dataclass
class NvidiaComplexityResult:
    """Result from NVIDIA complexity classifier."""
    prompt: str
    
    # Task type (top 2 predictions)
    task_type_1: str
    task_type_2: str
    task_type_prob: float
    
    # Complexity dimensions (0-1 scale)
    creativity_scope: float
    reasoning: float
    constraint_ct: float
    domain_knowledge: float
    contextual_knowledge: float
    number_of_few_shots: float
    
    # Overall complexity score (weighted combination)
    prompt_complexity_score: float
    
    # Convenience properties
    @property
    def is_complex(self) -> bool:
        """True if prompt is considered complex (score >= 0.4)."""
        return self.prompt_complexity_score >= 0.4
    
    @property
    def is_reasoning_heavy(self) -> bool:
        """True if prompt requires significant reasoning."""
        return self.reasoning >= 0.5
    
    @property
    def complexity_level(self) -> str:
        """Categorical complexity level."""
        score = self.prompt_complexity_score
        if score < 0.2:
            return "trivial"
        elif score < 0.35:
            return "simple"
        elif score < 0.5:
            return "moderate"
        elif score < 0.7:
            return "complex"
        else:
            return "expert"


def _load_model():
    """
    Lazy load the NVIDIA model and tokenizer.
    
    Uses manual weight loading to avoid segfaults with PyTorchModelHubMixin.from_pretrained()
    in newer versions of transformers/huggingface_hub. The issue is related to meta device
    handling in _load_state_dict_into_meta_model.
    """
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    logger.info("Loading NVIDIA prompt-task-and-complexity-classifier...")
    
    # Load config and tokenizer
    config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
    _tokenizer = AutoTokenizer.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
    
    # Define model architecture (matching NVIDIA's original structure)
    class MeanPooling(nn.Module):
        def __init__(self):
            super(MeanPooling, self).__init__()

        def forward(self, last_hidden_state, attention_mask):
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            return mean_embeddings

    class MulticlassHead(nn.Module):
        def __init__(self, input_size, num_classes):
            super(MulticlassHead, self).__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    class CustomModel(nn.Module):
        def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
            super(CustomModel, self).__init__()

            self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
            self.target_sizes = list(target_sizes.values())
            self.task_type_map = task_type_map
            self.weights_map = weights_map
            self.divisor_map = divisor_map

            # Create heads and register them with add_module to match saved weight names
            # (head_0, head_1, etc. instead of heads.0, heads.1)
            self.heads = []
            for i, sz in enumerate(self.target_sizes):
                head = MulticlassHead(self.backbone.config.hidden_size, sz)
                self.add_module(f"head_{i}", head)
                self.heads.append(head)

            self.pool = MeanPooling()

        def compute_results(self, preds, target, decimal=4):
            if target == "task_type":
                top2_indices = torch.topk(preds, k=2, dim=1).indices
                softmax_probs = torch.softmax(preds, dim=1)
                top2_probs = softmax_probs.gather(1, top2_indices)
                top2 = top2_indices.detach().cpu().tolist()
                top2_prob = top2_probs.detach().cpu().tolist()

                top2_strings = [
                    [self.task_type_map[str(idx)] for idx in sample] for sample in top2
                ]
                top2_prob_rounded = [
                    [round(value, 3) for value in sublist] for sublist in top2_prob
                ]

                counter = 0
                for sublist in top2_prob_rounded:
                    if sublist[1] < 0.1:
                        top2_strings[counter][1] = "NA"
                    counter += 1

                task_type_1 = [sublist[0] for sublist in top2_strings]
                task_type_2 = [sublist[1] for sublist in top2_strings]
                task_type_prob = [sublist[0] for sublist in top2_prob_rounded]

                return (task_type_1, task_type_2, task_type_prob)

            else:
                preds = torch.softmax(preds, dim=1)

                weights = np.array(self.weights_map[target])
                weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
                scores = weighted_sum / self.divisor_map[target]

                scores = [round(value, decimal) for value in scores]
                if target == "number_of_few_shots":
                    scores = [x if x >= 0.05 else 0 for x in scores]
                return scores

        def process_logits(self, logits):
            result = {}

            task_type_logits = logits[0]
            task_type_results = self.compute_results(task_type_logits, target="task_type")
            result["task_type_1"] = task_type_results[0]
            result["task_type_2"] = task_type_results[1]
            result["task_type_prob"] = task_type_results[2]

            result["creativity_scope"] = self.compute_results(logits[1], target="creativity_scope")
            result["reasoning"] = self.compute_results(logits[2], target="reasoning")
            result["contextual_knowledge"] = self.compute_results(logits[3], target="contextual_knowledge")
            result["number_of_few_shots"] = self.compute_results(logits[4], target="number_of_few_shots")
            result["domain_knowledge"] = self.compute_results(logits[5], target="domain_knowledge")
            result["no_label_reason"] = self.compute_results(logits[6], target="no_label_reason")
            result["constraint_ct"] = self.compute_results(logits[7], target="constraint_ct")

            result["prompt_complexity_score"] = [
                round(
                    0.35 * creativity
                    + 0.25 * reasoning
                    + 0.15 * constraint
                    + 0.15 * domain_knowledge
                    + 0.05 * contextual_knowledge
                    + 0.05 * few_shots,
                    5,
                )
                for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge, few_shots in zip(
                    result["creativity_scope"],
                    result["reasoning"],
                    result["constraint_ct"],
                    result["domain_knowledge"],
                    result["contextual_knowledge"],
                    result["number_of_few_shots"],
                )
            ]

            return result

        def forward(self, batch):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

            last_hidden_state = outputs.last_hidden_state
            mean_pooled_representation = self.pool(last_hidden_state, attention_mask)

            logits = [
                self.heads[k](mean_pooled_representation)
                for k in range(len(self.target_sizes))
            ]

            return self.process_logits(logits)
    
    # Create model instance
    _model = CustomModel(
        target_sizes=config.target_sizes,
        task_type_map=config.task_type_map,
        weights_map=config.weights_map,
        divisor_map=config.divisor_map,
    )
    
    # Download and load weights manually (avoids PyTorchModelHubMixin segfault)
    weights_path = hf_hub_download(
        repo_id="nvidia/prompt-task-and-complexity-classifier",
        filename="model.safetensors"
    )
    state_dict = load_file(weights_path)
    
    # Load weights (strict=True since architecture should match exactly now)
    _model.load_state_dict(state_dict, strict=True)
    _model.eval()
    
    logger.info("NVIDIA classifier loaded successfully")
    
    return _model, _tokenizer


class NvidiaComplexityClassifier:
    """
    Wrapper for NVIDIA's prompt-task-and-complexity-classifier.
    
    Provides complexity scoring across multiple dimensions for intelligent
    model routing and prompt filtering.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the classifier.
        
        Args:
            device: Device to run model on ("cpu" or "cuda")
        """
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _ensure_loaded(self):
        """Ensure model is loaded (lazy loading)."""
        if self._model is None:
            self._model, self._tokenizer = _load_model()
            if self.device != "cpu":
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to(self.device)
    
    def classify(self, prompt: str) -> NvidiaComplexityResult:
        """
        Classify a single prompt's complexity.
        
        Args:
            prompt: The prompt text to classify
            
        Returns:
            NvidiaComplexityResult with all complexity dimensions
        """
        results = self.classify_batch([prompt])
        return results[0]
    
    def classify_batch(self, prompts: List[str]) -> List[NvidiaComplexityResult]:
        """
        Classify a batch of prompts.
        
        Args:
            prompts: List of prompt texts
            
        Returns:
            List of NvidiaComplexityResult objects
        """
        import torch
        
        self._ensure_loaded()
        
        # Format prompts as expected by the model
        formatted_prompts = [f"Prompt: {p}" for p in prompts]
        
        # Tokenize
        encoded = self._tokenizer(
            formatted_prompts,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        
        if self.device != "cpu":
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get predictions
        with torch.no_grad():
            raw_results = self._model(encoded)
        
        # Convert to result objects
        results = []
        for i, prompt in enumerate(prompts):
            result = NvidiaComplexityResult(
                prompt=prompt,
                task_type_1=raw_results["task_type_1"][i],
                task_type_2=raw_results["task_type_2"][i],
                task_type_prob=raw_results["task_type_prob"][i],
                creativity_scope=raw_results["creativity_scope"][i],
                reasoning=raw_results["reasoning"][i],
                constraint_ct=raw_results["constraint_ct"][i],
                domain_knowledge=raw_results["domain_knowledge"][i],
                contextual_knowledge=raw_results["contextual_knowledge"][i],
                number_of_few_shots=raw_results["number_of_few_shots"][i],
                prompt_complexity_score=raw_results["prompt_complexity_score"][i],
            )
            results.append(result)
        
        return results
    
    def filter_by_complexity(
        self, 
        prompts: List[str], 
        min_score: float = 0.4,
        min_reasoning: Optional[float] = None,
    ) -> List[str]:
        """
        Filter prompts to only include complex ones.
        
        Args:
            prompts: List of prompts to filter
            min_score: Minimum overall complexity score (0-1)
            min_reasoning: Optional minimum reasoning score (0-1)
            
        Returns:
            List of prompts meeting the complexity threshold
        """
        results = self.classify_batch(prompts)
        
        filtered = []
        for result in results:
            if result.prompt_complexity_score >= min_score:
                if min_reasoning is None or result.reasoning >= min_reasoning:
                    filtered.append(result.prompt)
        
        return filtered
    
    def get_complexity_distribution(self, prompts: List[str]) -> Dict:
        """
        Get complexity distribution statistics for a set of prompts.
        
        Args:
            prompts: List of prompts to analyze
            
        Returns:
            Dict with distribution statistics
        """
        results = self.classify_batch(prompts)
        
        scores = [r.prompt_complexity_score for r in results]
        reasoning = [r.reasoning for r in results]
        
        levels = {"trivial": 0, "simple": 0, "moderate": 0, "complex": 0, "expert": 0}
        for r in results:
            levels[r.complexity_level] += 1
        
        task_types = {}
        for r in results:
            task_types[r.task_type_1] = task_types.get(r.task_type_1, 0) + 1
        
        return {
            "count": len(prompts),
            "complexity_score": {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": min(scores),
                "max": max(scores),
            },
            "reasoning_score": {
                "mean": np.mean(reasoning),
                "std": np.std(reasoning),
            },
            "complexity_levels": levels,
            "task_types": task_types,
        }


# Convenience function
def classify_prompt_complexity(prompt: str) -> NvidiaComplexityResult:
    """Quick function to classify a single prompt."""
    classifier = NvidiaComplexityClassifier()
    return classifier.classify(prompt)


if __name__ == "__main__":
    # Demo
    classifier = NvidiaComplexityClassifier()
    
    test_prompts = [
        "What is 2+2?",
        "Write hello world in Python",
        "Write a function to reverse a string",
        "Design a distributed cache system with LRU eviction and consistent hashing",
        "Implement a recursive solution to the N-queens problem with backtracking",
    ]
    
    print("NVIDIA Prompt Complexity Analysis")
    print("=" * 70)
    
    for prompt in test_prompts:
        result = classifier.classify(prompt)
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"  Task Type: {result.task_type_1} ({result.task_type_prob:.2f})")
        print(f"  Complexity: {result.prompt_complexity_score:.3f} ({result.complexity_level})")
        print(f"  Reasoning:  {result.reasoning:.3f}")
        print(f"  Creativity: {result.creativity_scope:.3f}")
        print(f"  Constraints: {result.constraint_ct:.3f}")
