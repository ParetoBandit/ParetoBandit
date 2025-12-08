import unittest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_jury.routing.archetype_router import ArchetypeRouter
from llm_jury.core.models import ProductArchetype, PromptCategory

class TestRoutingOptimization(unittest.TestCase):
    """
    Verification suite for the Optimized ArchetypeRouter.
    
    Tests the "Hybrid Semantic-Symbolic" routing logic:
    1. Intent Classification (Regex/Zero-Shot)
    2. Complexity Classification (Regex/Zero-Shot)
    3. Logic Matrix (State -> Archetype)
    """
    
    @classmethod
    def setUpClass(cls):
        print("\nInitializing ArchetypeRouter (this may download models)...")
        # Use local mode (use_api=False) for testing
        cls.router = ArchetypeRouter(use_api=False, fallback_threshold=0.75)
        print("Router initialized.")

    def test_coding_simple(self):
        """Test simple coding task -> REASONING_SPECIALIST (or BULK_OPS if very simple)"""
        prompt = "Write a python function to add two numbers."
        decision = self.router.route(prompt)
        print(f"\nPrompt: {prompt}\nDecision: {decision.archetype.name} ({decision.reason})")
        
        # Should be code generation intent
        self.assertIn("Code Generation", decision.reason)
        # Should route to Reasoning Specialist (or Frontier)
        self.assertIn(decision.archetype, [ProductArchetype.REASONING_SPECIALIST, ProductArchetype.FRONTIER])

    def test_coding_complex(self):
        """Test complex coding task -> REASONING_SPECIALIST"""
        prompt = "Design a microservices architecture for a high-scale e-commerce platform using Python and Kafka. Include error handling and data consistency strategies."
        decision = self.router.route(prompt)
        print(f"\nPrompt: {prompt}\nDecision: {decision.archetype.name} ({decision.reason})")
        
        self.assertIn(decision.archetype, [ProductArchetype.REASONING_SPECIALIST, ProductArchetype.FRONTIER])
        self.assertTrue(decision.recommend_cot, "Should recommend CoT for complex tasks")

    def test_creative_writing(self):
        """Test creative writing -> FRONTIER"""
        prompt = "Write a poem about the rust programming language in the style of Shakespeare."
        decision = self.router.route(prompt)
        print(f"\nPrompt: {prompt}\nDecision: {decision.archetype.name} ({decision.reason})")
        
        self.assertEqual(decision.archetype, ProductArchetype.FRONTIER)
        self.assertIn("Creative", decision.reason)

    def test_trivial_query(self):
        """Test trivial query -> BULK_OPS"""
        prompt = "What is 2+2?"
        decision = self.router.route(prompt)
        print(f"\nPrompt: {prompt}\nDecision: {decision.archetype.name} ({decision.reason})")
        
        self.assertEqual(decision.archetype, ProductArchetype.BULK_OPS)
        self.assertIn("Trivial", decision.reason)

    def test_ambiguous_query(self):
        """Test ambiguous query -> FRONTIER"""
        prompt = "Help me with this."
        decision = self.router.route(prompt)
        print(f"\nPrompt: {prompt}\nDecision: {decision.archetype.name} ({decision.reason})")
        
        self.assertEqual(decision.archetype, ProductArchetype.FRONTIER)
        self.assertIn("Ambiguous", decision.reason)

    def test_rag_query(self):
        """Test RAG query -> RAG_SPECIALIST"""
        prompt = "Who is the current CEO of Google?"
        decision = self.router.route(prompt, has_search_tools=True)
        print(f"\nPrompt: {prompt}\nDecision: {decision.archetype.name} ({decision.reason})")
        
        self.assertEqual(decision.archetype, ProductArchetype.RAG_SPECIALIST)

if __name__ == "__main__":
    unittest.main()
