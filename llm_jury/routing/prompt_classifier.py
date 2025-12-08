"""
Prompt to UseCase Classifier.

Maps natural language prompts to predefined UseCase categories
for optimized model recommendations.

Uses a multi-signal approach:
1. Keyword/pattern matching for explicit signals
2. Semantic similarity (future: embedding-based)
3. Complexity analysis for fallback classification
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


# Import UseCase from orchestrator to avoid circular imports
# We'll reference it by string and resolve at runtime
class UseCaseCategory(Enum):
    """High-level categories that group related use cases."""
    DEVELOPMENT = "development"
    DATA_ANALYTICS = "data_analytics"
    CONTENT = "content"
    SPECIALIZED = "specialized"
    CONVERSATIONAL = "conversational"
    TECHNICAL = "technical"
    OPTIMIZATION = "optimization"


@dataclass
class ClassificationResult:
    """Result of prompt classification."""
    use_case: str  # UseCase enum value (e.g., "code_generation")
    confidence: float  # 0.0 to 1.0
    category: UseCaseCategory
    signals: List[str]  # What triggered this classification
    alternative_use_cases: List[Tuple[str, float]]  # Other potential matches


class PromptClassifier:
    """
    Classifies prompts to UseCase categories for optimized model selection.
    
    Uses pattern matching with weighted signals to determine the best
    UseCase for a given prompt. This enables automatic selection of
    optimized weights for quality, cost, latency, etc.
    
    Example:
        >>> classifier = PromptClassifier()
        >>> result = classifier.classify("Write a Python function to parse JSON")
        >>> print(result.use_case)  # "code_generation"
        >>> print(result.confidence)  # 0.92
    """
    
    def __init__(self):
        # Pattern definitions with weights
        # Higher weight = stronger signal for that use case
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize classification patterns for each use case."""
        
        # =====================================================================
        # Development & Engineering
        # =====================================================================
        self.patterns = {
            # CODE_GENERATION - Writing new code
            "code_generation": {
                "category": UseCaseCategory.DEVELOPMENT,
                "patterns": [
                    (r"\b(write|create|implement|build|develop)\b.*\b(function|class|method|code|script|program|api|module)\b", 0.97),  # Highest priority for explicit code writing
                    (r"\b(python|javascript|typescript|java|rust|go|c\+\+|ruby|swift|kotlin)\b.*\b(function|class|code|script)\b", 0.95),
                    (r"\b(implement|code)\b.*\b(algorithm|data structure|feature)\b", 0.92),
                    (r"\b(write|create)\b.*\b(python|javascript|typescript)\b", 0.93),  # "Write python" -> code
                    (r"\b(create|build)\b.*\b(backend|frontend|api|endpoint)\b", 0.88),
                    (r"\b(write)\b.*\b(test|unittest|pytest)\b", 0.85),
                    (r"\b(generate|scaffold)\b.*\b(code|boilerplate)\b", 0.85),
                ],
                "keywords": ["coding", "programming", "developer", "software", "engineer", "function"],
                "keyword_weight": 0.60,
            },
            
            # CODE_REVIEW - Reviewing existing code
            "code_review": {
                "category": UseCaseCategory.DEVELOPMENT,
                "patterns": [
                    (r"\b(review|analyze|check|audit)\b.*\b(code|function|class|implementation)\b", 0.95),
                    (r"\b(find|identify|spot)\b.*\b(bug|issue|problem|error|vulnerability)\b", 0.90),
                    (r"\b(what.*(wrong|issue)|is this.*correct)\b", 0.75),
                    (r"\b(security|vulnerability)\b.*\b(review|audit|check)\b", 0.90),
                    (r"\b(best practice|anti-?pattern|code smell)\b", 0.85),
                ],
                "keywords": ["review", "audit", "inspect", "examine"],
                "keyword_weight": 0.50,
            },
            
            # CODE_REFACTORING - Improving existing code
            "code_refactoring": {
                "category": UseCaseCategory.DEVELOPMENT,
                "patterns": [
                    (r"\b(refactor|improve|optimize|clean up|modernize)\b.*\b(code|function|class)\b", 0.95),
                    (r"\b(make|rewrite)\b.*\b(cleaner|better|faster|more efficient)\b", 0.85),
                    (r"\b(reduce|simplify)\b.*\b(complexity|duplication)\b", 0.85),
                    (r"\b(extract|inline)\b.*\b(method|function|variable)\b", 0.90),
                    (r"\b(convert|migrate|upgrade)\b.*\b(to|from)\b", 0.80),
                ],
                "keywords": ["refactor", "improve", "optimize", "cleanup"],
                "keyword_weight": 0.55,
            },
            
            # TECHNICAL_DOCS - Documentation
            "technical_docs": {
                "category": UseCaseCategory.DEVELOPMENT,
                "patterns": [
                    (r"\b(write|create|generate)\b.*\b(documentation|docs|readme|docstring)\b", 0.95),
                    (r"\b(document|explain)\b.*\b(api|function|class|code|module)\b", 0.90),
                    (r"\b(add|write)\b.*\b(comments|docstrings|jsdoc|typing)\b", 0.85),
                    (r"\b(api|sdk)\b.*\b(documentation|reference|guide)\b", 0.90),
                ],
                "keywords": ["documentation", "readme", "docstring", "comments"],
                "keyword_weight": 0.60,
            },
            
            # =====================================================================
            # Data & Analytics
            # =====================================================================
            
            # DATA_ANALYSIS - Analyzing datasets
            "data_analysis": {
                "category": UseCaseCategory.DATA_ANALYTICS,
                "patterns": [
                    (r"\b(analyze|analyse|explore|examine)\b.*\b(data|dataset|csv|dataframe)\b", 0.95),
                    (r"\b(find|identify|discover)\b.*\b(trend|pattern|insight|correlation|anomaly)\b", 0.90),
                    (r"\b(pandas|numpy|matplotlib|seaborn|plotly)\b", 0.85),
                    (r"\b(statistics|statistical)\b.*\b(analysis|test)\b", 0.90),
                    (r"\b(visualization|chart|graph|plot)\b", 0.75),
                ],
                "keywords": ["data", "analysis", "analytics", "insight", "trend"],
                "keyword_weight": 0.55,
            },
            
            # SQL_GENERATION - Database queries
            "sql_generation": {
                "category": UseCaseCategory.DATA_ANALYTICS,
                "patterns": [
                    (r"\b(write|create|generate|build)\b.*\b(sql|query|select|join)\b", 0.95),
                    (r"\b(select|insert|update|delete)\b.*\b(from|into|where)\b", 0.90),
                    (r"\b(database|table|schema|index)\b.*\b(design|create|optimize)\b", 0.90),
                    (r"\b(postgres|mysql|sqlite|oracle|bigquery|snowflake)\b", 0.85),
                    (r"\b(join|aggregate|group by|having|window function)\b", 0.80),
                ],
                "keywords": ["sql", "database", "query", "table"],
                "keyword_weight": 0.70,
            },
            
            # MATH_REASONING - Mathematical problems
            "math_reasoning": {
                "category": UseCaseCategory.DATA_ANALYTICS,
                "patterns": [
                    (r"\b(solve|calculate|compute|derive|prove)\b.*\b(equation|integral|derivative|formula)\b", 0.95),
                    (r"\b(math|mathematical|algebra|calculus|geometry)\b", 0.80),
                    (r"\b(∫|∑|∏|∂|∇|∞|√)\b", 0.95),  # Math symbols
                    (r"\b(theorem|proof|lemma|corollary)\b", 0.90),
                    (r"\b(optimization|maximize|minimize|constraint)\b.*\b(problem|function)\b", 0.85),
                    (r"\b(probability|statistics|variance|distribution)\b", 0.80),
                ],
                "keywords": ["math", "calculate", "solve", "equation", "proof"],
                "keyword_weight": 0.65,
            },
            
            # =====================================================================
            # Content & Communication
            # =====================================================================
            
            # CREATIVE_WRITING - Stories, marketing, creative
            "creative_writing": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(write|create|compose)\b.*\b(story|poem|novel|essay|article|blog)\b", 0.95),
                    (r"\b(marketing|ad|advertisement|copy|slogan|tagline)\b", 0.85),
                    (r"\b(character|plot|narrative|dialogue|scene)\b", 0.80),
                    (r"\b(creative|imaginative|artistic)\b.*\b(writing|content)\b", 0.85),
                    (r"\b(tone|voice|style)\b.*\b(casual|formal|professional|friendly)\b", 0.70),
                ],
                "keywords": ["story", "creative", "write", "poem", "narrative", "marketing"],
                "keyword_weight": 0.60,
            },
            
            # SUMMARIZATION - Condensing content
            "summarization": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(summarize|summarise|condense|tldr|tl;dr)\b", 0.95),
                    (r"\b(key points|main ideas|highlights|takeaways)\b", 0.85),
                    (r"\b(brief|concise|short)\b.*\b(summary|overview|version)\b", 0.90),
                    (r"\b(in (a )?few (words|sentences)|briefly)\b", 0.80),
                ],
                "keywords": ["summary", "summarize", "brief", "condense"],
                "keyword_weight": 0.75,
            },
            
            # TRANSLATION - Language translation
            "translation": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(translate|translation)\b.*\b(to|from|into)\b", 0.95),
                    (r"\b(in|to)\b\s+(english|spanish|french|german|chinese|japanese|korean|arabic|portuguese|russian|italian|dutch|hindi)\b", 0.80),
                    (r"\b(localize|localization)\b", 0.85),
                ],
                "keywords": ["translate", "translation", "language"],
                "keyword_weight": 0.70,
            },
            
            # =====================================================================
            # Specialized Domains
            # =====================================================================
            
            # LEGAL_REVIEW - Legal analysis
            "legal_review": {
                "category": UseCaseCategory.SPECIALIZED,
                "patterns": [
                    (r"\b(review|analyze|draft)\b.*\b(contract|agreement|clause|terms)\b", 0.95),
                    (r"\b(legal|law|lawyer|attorney)\b.*\b(advice|opinion|analysis)\b", 0.90),
                    (r"\b(compliance|regulation|regulatory)\b", 0.85),
                    (r"\b(liability|indemnity|warranty|breach|damages)\b", 0.85),
                    (r"\b(nda|non-disclosure|confidentiality|ip|intellectual property)\b", 0.90),
                ],
                "keywords": ["legal", "contract", "law", "compliance", "attorney"],
                "keyword_weight": 0.70,
            },
            
            # FINANCIAL_ANALYSIS - Financial modeling
            "financial_analysis": {
                "category": UseCaseCategory.SPECIALIZED,
                "patterns": [
                    (r"\b(analyze|analyse|evaluate)\b.*\b(stock|market|investment|portfolio)\b", 0.95),
                    (r"\b(financial|finance)\b.*\b(analysis|model|forecast|projection)\b", 0.90),
                    (r"\b(dcf|npv|irr|ebitda|p/e|eps|roi)\b", 0.90),
                    (r"\b(balance sheet|income statement|cash flow)\b", 0.85),
                    (r"\b(valuation|risk assessment|due diligence)\b", 0.85),
                ],
                "keywords": ["financial", "investment", "stock", "market", "valuation"],
                "keyword_weight": 0.65,
            },
            
            # RESEARCH_ASSISTANT - Academic research
            "research_assistant": {
                "category": UseCaseCategory.SPECIALIZED,
                "patterns": [
                    (r"\b(research|literature|academic)\b.*\b(review|survey|analysis)\b", 0.95),
                    (r"\b(find|search)\b.*\b(paper|study|publication|journal)\b", 0.90),
                    (r"\b(cite|citation|reference|bibliography)\b", 0.85),
                    (r"\b(methodology|hypothesis|experiment|findings)\b", 0.80),
                    (r"\b(peer.?review|scientific|scholarly)\b", 0.85),
                ],
                "keywords": ["research", "academic", "study", "paper", "scientific"],
                "keyword_weight": 0.60,
            },
            
            # =====================================================================
            # Conversational & Support
            # =====================================================================
            
            # CUSTOMER_SUPPORT - Help desk, chatbots
            "customer_support": {
                "category": UseCaseCategory.CONVERSATIONAL,
                "patterns": [
                    (r"\b(help|assist|support)\b.*\b(customer|user|client)\b", 0.90),
                    (r"\b(respond|reply|answer)\b.*\b(ticket|inquiry|question|complaint)\b", 0.90),
                    (r"\b(faq|frequently asked|common question)\b", 0.85),
                    (r"\b(polite|friendly|empathetic)\b.*\b(response|reply)\b", 0.80),
                ],
                "keywords": ["support", "customer", "help", "assist", "service"],
                "keyword_weight": 0.55,
            },
            
            # TUTORING - Educational explanations
            "tutoring": {
                "category": UseCaseCategory.CONVERSATIONAL,
                "patterns": [
                    (r"\b(explain|teach|help me understand)\b", 0.85),
                    (r"\b(learn|understand|comprehend)\b.*\b(how|why|what)\b", 0.80),
                    (r"\b(tutor|tutorial|lesson|lecture)\b", 0.90),
                    (r"\b(step by step|breakdown|explain like|eli5)\b", 0.85),
                    (r"\b(beginner|introduction|basics)\b", 0.70),
                ],
                "keywords": ["explain", "teach", "learn", "understand", "tutorial"],
                "keyword_weight": 0.55,
            },
            
            # GENERAL_QA - General questions
            "general_qa": {
                "category": UseCaseCategory.CONVERSATIONAL,
                "patterns": [
                    (r"^(what|who|when|where|why|how)\b", 0.60),
                    (r"\b(tell me|can you|could you)\b.*\b(about|explain)\b", 0.65),
                    (r"\b(what is|define|meaning of)\b", 0.70),
                ],
                "keywords": ["what", "who", "when", "where", "why", "how", "question"],
                "keyword_weight": 0.30,
            },
            
            # =====================================================================
            # Technical Capabilities
            # =====================================================================
            
            # RAG_PIPELINE - Retrieval-augmented generation / Document Q&A
            "rag_pipeline": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    # Explicit RAG mentions
                    (r"\b(rag|retrieval.?augmented)\b", 0.95),
                    
                    # Document-based Q&A patterns (with plurals)
                    (r"\b(using|from|based on|according to)\b.*\b(these|this|the|provided|attached)\b.*\b(documents?|files?|pdfs?|texts?|data)\b", 0.92),
                    (r"\b(these|this|the)\b.*\b(documents?|files?|pdfs?|context)\b.*\b(answer|question|tell|explain)\b", 0.90),
                    (r"\b(answer|respond|reply)\b.*\b(based on|from|using)\b.*\b(documents?|context|texts?|files?)\b", 0.90),
                    (r"\b(questions?|ask)\b.*\b(about|regarding|from)\b.*\b(documents?|files?|texts?|policy|report)\b", 0.88),
                    
                    # "Using X, do Y" pattern (very common for RAG)
                    (r"\busing\b.{0,20}\b(documents?|files?|context|sources?)\b.{0,30}\b(answer|explain|tell|find|help)\b", 0.90),
                    
                    # Context/source patterns
                    (r"\b(based on|according to|from)\b.*\b(documents?|context|passages?|texts?)\b", 0.88),
                    (r"\b(retrieve|search|find)\b.*\b(relevant|information|answer)\b", 0.85),
                    (r"\b(using|given)\b.*\b(context|sources?|references?|knowledge base)\b", 0.85),
                    
                    # Knowledge base patterns
                    (r"\b(knowledge base|vector|embedding|semantic search)\b", 0.90),
                    (r"\b(chat with|talk to)\b.*\b(documents?|pdfs?|files?|data|this)\b", 0.90),
                    
                    # Q&A with documents
                    (r"\b(help me|i need to)\b.*\b(documents?|files?)\b", 0.82),
                    
                    # "Based on the provided X" patterns
                    (r"\bbased on\b.{0,15}\b(provided|given|attached|uploaded)\b.{0,10}\b(pdf|document|file|text|data)\b", 0.93),
                    (r"\b(provided|given|attached|uploaded)\b.{0,10}\b(pdf|document|file|context)\b", 0.88),
                    (r"\bthe provided\b.{0,10}\b(pdf|document|file)\b", 0.90),
                    
                    # Chat/interact with document (more flexible)
                    (r"\b(chat|interact|work)\b.{0,15}\bwith\b.{0,15}\b(document|pdf|file)\b", 0.92),
                    (r"\bchat with this\b", 0.88),
                ],
                "keywords": ["context", "document", "documents", "retrieve", "source", "knowledge", "provided", "attached", "files", "pdf"],
                "keyword_weight": 0.65,
            },
            
            # FUNCTION_CALLING - Structured API/function calls
            "function_calling": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(function calling|api calling)\b", 0.98),
                    (r"\bcall\b.*\b(the|an|this)\s*(api|function)\b", 0.95),  # "call the API"
                    (r"\b(invoke|execute)\b.*\b(function|api)\b", 0.92),
                    (r"\b(openai|anthropic)\b.*\b(function|tool)\b.*\b(calling|call)\b", 0.95),
                    (r"\b(json|schema)\b.*\b(function|tool)\b", 0.90),
                    (r"\b(structured|typed)\b.*\b(output|response|call)\b", 0.85),
                ],
                "keywords": ["function calling", "api call", "schema", "invoke"],
                "keyword_weight": 0.55,
            },
            
            # STRUCTURED_EXTRACTION - JSON/schema extraction (not NER or code writing)
            "structured_extraction": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(extract|parse)\b.*\b(json|xml|yaml)\b", 0.95),  # Explicit format
                    (r"\b(convert|transform)\b.*\b(to|into)\b.*\b(json|structured|format)\b", 0.92),
                    (r"\b(schema|structured output|structured data)\b", 0.90),
                    (r"\b(fill|populate)\b.*\b(form|template|fields)\b", 0.88),
                    (r"\b(extract)\b.*\b(fields|values|properties)\b.*\b(json|schema)\b", 0.90),
                    (r"\b(output|return)\b.*\b(as|in)\b.*\b(json|structured)\b", 0.88),
                ],
                "keywords": ["json", "structured", "schema", "format", "yaml", "xml"],
                "keyword_weight": 0.55,
            },
            
            # LONG_CONTEXT - Very long documents
            "long_context": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(long|large|big)\b.*\b(document|file|text|context)\b", 0.85),
                    (r"\b(entire|whole|full)\b.*\b(document|book|codebase|repo)\b", 0.85),
                    (r"\b(100k|128k|200k|\d+k tokens)\b", 0.90),
                ],
                "keywords": ["long", "entire", "full", "complete"],
                "keyword_weight": 0.40,
            },
            
            # =====================================================================
            # Cost/Performance Focused
            # =====================================================================
            
            # COST_OPTIMIZED - Budget-conscious
            "cost_optimized": {
                "category": UseCaseCategory.OPTIMIZATION,
                "patterns": [
                    (r"\b(cheap|budget|cost.?effective|affordable|inexpensive)\b", 0.95),
                    (r"\b(minimize|reduce|lower)\b.*\b(cost|price|expense)\b", 0.90),
                    (r"\b(high.?volume|batch|bulk)\b.*\b(process|task)\b", 0.80),
                ],
                "keywords": ["cheap", "budget", "cost", "affordable"],
                "keyword_weight": 0.75,
            },
            
            # LOW_LATENCY - Speed focused
            "low_latency": {
                "category": UseCaseCategory.OPTIMIZATION,
                "patterns": [
                    (r"\b(fast|quick|rapid|instant|real.?time)\b", 0.85),
                    (r"\b(low.?latency|streaming|interactive)\b", 0.95),
                    (r"\b(minimize|reduce)\b.*\b(latency|delay|response time)\b", 0.90),
                ],
                "keywords": ["fast", "quick", "instant", "realtime", "streaming"],
                "keyword_weight": 0.70,
            },
            
            # MAXIMUM_QUALITY - Best results regardless of cost
            "maximum_quality": {
                "category": UseCaseCategory.OPTIMIZATION,
                "patterns": [
                    (r"\b(best|highest|maximum|top)\b.*\b(quality|accuracy|performance)\b", 0.95),
                    (r"\b(most (accurate|capable|intelligent))\b", 0.90),
                    (r"\b(doesn.t matter|regardless of)\b.*\b(cost|price)\b", 0.85),
                ],
                "keywords": ["best", "highest", "maximum", "quality"],
                "keyword_weight": 0.65,
            },
            
            # =====================================================================
            # Multimodal & Vision
            # =====================================================================
            
            # IMAGE_UNDERSTANDING - Analyze/describe images
            "image_understanding": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(describe|analyze|explain|look at)\b.*\b(image|picture|photo|screenshot|diagram)\b", 0.95),
                    (r"\b(what('s| is) in)\b.*\b(image|picture|photo)\b", 0.95),
                    (r"\b(image|picture|photo|screenshot)\b.*\b(show|contain|depict)\b", 0.90),
                    (r"\b(read|extract|ocr)\b.*\b(text|content)\b.*\b(image|screenshot|photo)\b", 0.92),
                    (r"\b(caption|title)\b.*\b(image|photo)\b", 0.85),
                    (r"\bvisual\s+(analysis|understanding|question)\b", 0.90),
                ],
                "keywords": ["image", "picture", "photo", "screenshot", "diagram", "visual", "vision"],
                "keyword_weight": 0.70,
            },
            
            # VISION_QA - Question answering about images
            "vision_qa": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(answer|tell me|what|how|why)\b.*\b(about|from)\b.*\b(image|picture|photo|diagram)\b", 0.92),
                    (r"\b(question|ask)\b.*\b(image|picture|visual)\b", 0.90),
                    (r"\b(based on|according to)\b.*\b(image|diagram|chart|graph)\b", 0.88),
                    (r"\b(chart|graph|plot)\b.*\b(show|indicate|mean)\b", 0.85),
                ],
                "keywords": ["image", "visual", "chart", "graph", "diagram"],
                "keyword_weight": 0.60,
            },
            
            # =====================================================================
            # Embeddings & Similarity
            # =====================================================================
            
            # EMBEDDINGS - Vector representations
            "embeddings": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(generate|create|compute|get)\b.*\b(embedding|embeddings|vector|vectors)\b", 0.98),
                    (r"\b(embed|vectorize|encode)\b.*\b(text|sentence|document|query)\b", 0.95),
                    (r"\b(embedding|vector)\b.*\b(representation|space|model)\b", 0.90),
                    (r"\b(semantic|dense)\b.*\b(search|retrieval|index)\b", 0.88),
                    (r"\b(vector\s*(database|store|db|index))\b", 0.92),
                ],
                "keywords": ["embedding", "vector", "encode", "semantic search", "retrieval"],
                "keyword_weight": 0.80,
            },
            
            # SEMANTIC_SIMILARITY - Compare text similarity
            "semantic_similarity": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(similar|similarity|compare)\b.*\b(text|sentence|document|meaning)\b", 0.92),
                    (r"\b(how (similar|different|close))\b.*\b(text|sentence|meaning)\b", 0.90),
                    (r"\b(semantic|meaning)\b.*\b(match|compare|distance)\b", 0.88),
                    (r"\b(duplicate|near.?duplicate)\b.*\b(detection|finding)\b", 0.85),
                ],
                "keywords": ["similar", "similarity", "compare", "duplicate", "match"],
                "keyword_weight": 0.55,
            },
            
            # =====================================================================
            # Agentic & Tool Use
            # =====================================================================
            
            # AGENT_WORKFLOW - Multi-step autonomous tasks
            "agent_workflow": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(agent|autonomous|multi.?step)\b.*\b(task|workflow|process)\b", 0.95),
                    (r"\b(break down|decompose|plan out)\b.*\b(task|problem|steps)\b", 0.88),
                    (r"\b(langchain|autogpt|crewai|langgraph)\b", 0.95),
                    (r"\b(research|browse|search)\b.*\b(web|internet|online)\b.*\b(for me|autonomously)\b", 0.90),
                    (r"\b(complete|accomplish|do)\b.*\b(on (your|my) own|by yourself|automatically)\b", 0.85),
                ],
                "keywords": ["agent", "autonomous", "workflow", "langchain", "autogpt", "multi-step"],
                "keyword_weight": 0.70,
            },
            
            # TOOL_USE - LLM using external tools
            "tool_use": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(use|call|invoke)\b.*\b(this\s+)?(tool|api|function|plugin)\b", 0.92),
                    (r"\b(tool|plugin)\b.*\b(integration|calling|use)\b", 0.90),
                    (r"\b(mcp|model context protocol)\b", 0.95),
                    (r"\b(connect to|interact with)\b.*\b(api|service|database|tool)\b", 0.85),
                    (r"\b(action|execute|run)\b.*\b(command|function|tool)\b", 0.82),
                    (r"\buse this tool\b", 0.95),
                    (r"\bcall (this|the) (api|function|tool)\b", 0.92),
                ],
                "keywords": ["tool", "plugin", "api", "function", "action", "mcp"],
                "keyword_weight": 0.70,
            },
            
            # PLANNING - Breaking complex tasks into steps
            "planning": {
                "category": UseCaseCategory.TECHNICAL,
                "patterns": [
                    (r"\b(create|make|develop)\b.*\b(plan|roadmap|strategy|schedule)\b", 0.92),
                    (r"\b(step.?by.?step|detailed)\b.*\b(plan|approach|guide)\b", 0.90),
                    (r"\b(break (down|into)|decompose)\b.*\b(steps|tasks|phases)\b", 0.88),
                    (r"\b(project|task)\b.*\b(plan|planning|breakdown)\b", 0.85),
                    (r"\b(how (should|do) (i|we))\b.*\b(approach|tackle|handle)\b", 0.80),
                ],
                "keywords": ["plan", "planning", "roadmap", "strategy", "steps", "approach"],
                "keyword_weight": 0.55,
            },
            
            # =====================================================================
            # Classification & Analysis
            # =====================================================================
            
            # TEXT_CLASSIFICATION - Categorize text
            "text_classification": {
                "category": UseCaseCategory.DATA_ANALYTICS,
                "patterns": [
                    (r"\b(classify|categorize|label|tag)\b.*\b(text|document|content|message)\b", 0.95),
                    (r"\b(which (category|type|class))\b", 0.88),
                    (r"\b(assign|determine)\b.*\b(label|category|class|tag)\b", 0.90),
                    (r"\b(is this|does this)\b.*\b(belong|fall under|fit)\b", 0.82),
                    (r"\b(spam|not spam|topic|intent)\b.*\b(detection|classification)\b", 0.92),
                ],
                "keywords": ["classify", "categorize", "label", "tag", "category", "classification"],
                "keyword_weight": 0.65,
            },
            
            # SENTIMENT_ANALYSIS - Detect sentiment/opinion
            "sentiment_analysis": {
                "category": UseCaseCategory.DATA_ANALYTICS,
                "patterns": [
                    (r"\b(sentiment|tone|mood|attitude)\b.*\b(analysis|detect|determine)\b", 0.95),
                    (r"\b(positive|negative|neutral)\b.*\b(sentiment|feeling|opinion)\b", 0.90),
                    (r"\b(how (does|do)|what('s| is))\b.*\b(feel|feeling|tone|sentiment)\b", 0.88),
                    (r"\b(opinion|feedback)\b.*\b(mining|analysis|detection)\b", 0.92),
                    (r"\b(is this|does this)\b.*\b(positive|negative|happy|angry|sad)\b", 0.85),
                ],
                "keywords": ["sentiment", "tone", "mood", "opinion", "feeling", "positive", "negative"],
                "keyword_weight": 0.70,
            },
            
            # ENTITY_EXTRACTION - Extract named entities (NER)
            "entity_extraction": {
                "category": UseCaseCategory.DATA_ANALYTICS,
                "patterns": [
                    (r"\b(extract|identify|find|pull out)\b.*\b(entities|names|people|places|organizations)\b", 0.95),
                    (r"\b(named entity|ner)\b", 0.98),
                    (r"\b(extract)\b.*\b(named\s+)?entities\b", 0.95),
                    (r"\b(extract|identify)\b.*\b(dates|emails|phone|addresses|urls)\b", 0.90),
                    (r"\b(who|what|where|when)\b.*\b(mentioned|named|referred)\b", 0.82),
                    (r"\b(parse|extract)\b.*\b(contact|company|person|location)\b.*\b(information|details)\b", 0.88),
                    (r"\b(extract)\b.*\b(from this|from the)\b.*\b(document|text)\b", 0.85),
                ],
                "keywords": ["extract", "entity", "ner", "names", "parse", "identify", "named entities"],
                "keyword_weight": 0.70,
            },
            
            # CONTENT_MODERATION - Safety/toxicity detection
            "content_moderation": {
                "category": UseCaseCategory.SPECIALIZED,
                "patterns": [
                    (r"\b(moderate|moderation|filter|flag)\b.*\b(content|text|comment|message)\b", 0.95),
                    (r"\b(detect|identify|check)\b.*\b(toxic|harmful|inappropriate|offensive|spam)\b", 0.95),
                    (r"\b(safe|safety|nsfw|sfw)\b.*\b(check|filter|detect)\b", 0.92),
                    (r"\b(is this|does this|check if)\b.*\b(appropriate|safe|toxic|harmful)\b", 0.92),
                    (r"\b(hate speech|harassment|abuse)\b.*\b(detect|filter)\b", 0.95),
                    (r"\b(appropriate|inappropriate)\b.*\b(content|text|this)\b", 0.88),
                    (r"\bcheck.*(content|this).*(appropriate|safe)\b", 0.90),
                    (r"\b(content|this).*(appropriate|safe)\b", 0.82),
                ],
                "keywords": ["moderate", "moderation", "toxic", "harmful", "safe", "inappropriate", "nsfw", "appropriate"],
                "keyword_weight": 0.75,
            },
            
            # =====================================================================
            # Text Transformation
            # =====================================================================
            
            # PARAPHRASING - Rewrite without changing meaning
            "paraphrasing": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(paraphrase|rephrase|reword)\b", 0.98),  # Explicit paraphrasing
                    (r"\b(say|write)\b.*\b(differently|another way|other words)\b", 0.92),
                    (r"\b(same meaning|preserve meaning)\b.*\b(different words)\b", 0.90),
                    (r"\b(simplify|make simpler)\b.*\b(text|sentence|paragraph)\b", 0.85),
                    (r"\brewrite\b.*\b(without changing|same meaning)\b", 0.95),
                ],
                "keywords": ["paraphrase", "rephrase", "reword", "simplify"],
                "keyword_weight": 0.70,
            },
            
            # STYLE_TRANSFER - Change tone/style
            "style_transfer": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(change|convert|transform)\b.*\b(tone|style|voice|register)\b", 0.95),
                    (r"\b(make|rewrite)\b.*\b(more|less)\b.*\b(formal|casual|professional|friendly)\b", 0.92),
                    (r"\b(formal|casual|professional|academic|conversational)\b.*\b(tone|style|version)\b", 0.88),
                    (r"\b(write|rewrite)\b.*\b(like|as if|style of)\b", 0.85),
                    (r"\brewrite.*(in a|with a).*(tone|style)\b", 0.92),
                    (r"\b(more|less)\s+(formal|casual|professional)\b", 0.88),
                    (r"\b(in a)\b.*(formal|casual|professional|friendly)\b.*(tone|way|style)\b", 0.90),
                ],
                "keywords": ["tone", "style", "formal", "casual", "professional", "voice", "rewrite"],
                "keyword_weight": 0.70,
            },
            
            # GRAMMAR_CORRECTION - Fix grammar/spelling
            "grammar_correction": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(fix|correct|check)\b.*\b(grammar|spelling|punctuation|typos)\b", 0.95),
                    (r"\b(proofread|proof.?read|edit)\b.*\b(text|document|writing)\b", 0.92),
                    (r"\b(grammar|spelling|typo)\b.*\b(error|mistake|issue)\b", 0.88),
                    (r"\b(is this|are there)\b.*\b(grammatically correct|errors|mistakes)\b", 0.85),
                ],
                "keywords": ["grammar", "spelling", "proofread", "correct", "typo", "edit"],
                "keyword_weight": 0.70,
            },
            
            # =====================================================================
            # Creative & Ideation
            # =====================================================================
            
            # BRAINSTORMING - Generate ideas
            "brainstorming": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(brainstorm|ideate|generate)\b.*\b(ideas?|concepts?|options?|suggestions?)\b", 0.95),
                    (r"\b(give me|list|come up with)\b.*\b(ideas?|suggestions?|options?)\b", 0.88),
                    (r"\b(what (are|could be)|suggest)\b.*\b(ideas?|ways?|options?|approaches?)\b", 0.85),
                    (r"\b(creative|innovative)\b.*\b(ideas?|solutions?|approaches?)\b", 0.85),
                    (r"\b(help me think|thinking of)\b", 0.75),
                ],
                "keywords": ["brainstorm", "ideas", "suggestions", "options", "creative", "innovative"],
                "keyword_weight": 0.60,
            },
            
            # ROLEPLAY - Character-based conversations
            "roleplay": {
                "category": UseCaseCategory.CONTENT,
                "patterns": [
                    (r"\b(roleplay|role.?play|pretend|act as|play as)\b", 0.95),
                    (r"\b(you are|be|act like)\b.*\b(character|person|role)\b", 0.90),
                    (r"\b(in.?character|stay in character|character voice)\b", 0.92),
                    (r"\b(rpg|game master|dm|dungeon master|npc)\b", 0.90),
                    (r"\b(persona|character)\b.*\b(creation|development|play)\b", 0.85),
                ],
                "keywords": ["roleplay", "character", "persona", "pretend", "act", "rpg"],
                "keyword_weight": 0.75,
            },
        }
    
    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify a prompt to the most appropriate UseCase.
        
        Args:
            prompt: Natural language prompt to classify
            
        Returns:
            ClassificationResult with use_case, confidence, and signals
        """
        prompt_lower = prompt.lower()
        scores: Dict[str, Tuple[float, List[str]]] = {}
        
        # Score each use case
        for use_case, config in self.patterns.items():
            score = 0.0
            signals = []
            
            # Pattern matching
            for pattern, weight in config["patterns"]:
                if re.search(pattern, prompt_lower, re.IGNORECASE):
                    score = max(score, weight)
                    signals.append(f"pattern:{pattern[:30]}...")
            
            # Keyword matching (additive, lower weight)
            keyword_matches = 0
            for keyword in config.get("keywords", []):
                if keyword in prompt_lower:
                    keyword_matches += 1
                    signals.append(f"keyword:{keyword}")
            
            if keyword_matches > 0:
                keyword_boost = min(
                    config.get("keyword_weight", 0.5) * (keyword_matches / len(config.get("keywords", [1]))),
                    config.get("keyword_weight", 0.5)
                )
                score = max(score, score + keyword_boost * 0.3)  # Small additive boost
            
            if score > 0:
                scores[use_case] = (score, signals)
        
        # Find best match
        if not scores:
            # Default to general QA if nothing matches
            return ClassificationResult(
                use_case="general_qa",
                confidence=0.3,
                category=UseCaseCategory.CONVERSATIONAL,
                signals=["default:no_match"],
                alternative_use_cases=[]
            )
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        best_use_case, (best_score, best_signals) = sorted_scores[0]
        
        # Get alternatives (other high-scoring use cases)
        alternatives = [
            (uc, score) for uc, (score, _) in sorted_scores[1:4] if score > 0.5
        ]
        
        # AMBIGUITY HANDLING: If confidence is low, fall back to general_qa
        # This ensures we recommend well-rounded models rather than mis-routing
        AMBIGUITY_THRESHOLD = 0.70
        
        if best_score < AMBIGUITY_THRESHOLD:
            # Low confidence - use general_qa for best all-around models
            return ClassificationResult(
                use_case="general_qa",
                confidence=best_score,  # Keep original score to indicate uncertainty
                category=UseCaseCategory.CONVERSATIONAL,
                signals=["ambiguous:low_confidence"] + best_signals[:4],
                alternative_use_cases=[(best_use_case, best_score)] + alternatives[:2]
            )
        
        # Check for close competition (ambiguity between top choices)
        # Only consider ambiguous if:
        # 1. Best score is below high-confidence threshold (0.90)
        # 2. AND gap is very small (< 0.05)
        # This ensures we trust high-confidence matches even with close alternatives
        if len(sorted_scores) >= 2:
            second_use_case, (second_score, _) = sorted_scores[1]
            score_gap = best_score - second_score
            
            # Only fall back to general_qa if genuinely uncertain
            if score_gap < 0.05 and best_score < 0.90 and second_score >= 0.70:
                return ClassificationResult(
                    use_case="general_qa",
                    confidence=best_score,
                    category=UseCaseCategory.CONVERSATIONAL,
                    signals=["ambiguous:close_competition", f"tied:{best_use_case}≈{second_use_case}"],
                    alternative_use_cases=[(best_use_case, best_score), (second_use_case, second_score)]
                )
        
        return ClassificationResult(
            use_case=best_use_case,
            confidence=min(best_score, 1.0),
            category=self.patterns[best_use_case]["category"],
            signals=best_signals[:5],  # Top 5 signals
            alternative_use_cases=alternatives
        )
    
    def classify_with_context(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[str]] = None
    ) -> ClassificationResult:
        """
        Classify with additional context from system prompt or conversation.
        
        This is useful for chatbot scenarios where context evolves over time.
        """
        # Combine all context
        full_context = prompt
        
        if system_prompt:
            full_context = f"[SYSTEM: {system_prompt}]\n{full_context}"
        
        if conversation_history:
            # Use last 2 messages for context
            recent = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
            history_text = " ".join(recent)
            full_context = f"[HISTORY: {history_text}]\n{full_context}"
        
        return self.classify(full_context)
    
    def get_use_case_description(self, use_case: str) -> str:
        """Get human-readable description of a use case."""
        descriptions = {
            "code_generation": "Writing new code, implementing features, algorithms",
            "code_review": "Reviewing code for bugs, security issues, best practices",
            "code_refactoring": "Improving, modernizing, or restructuring existing code",
            "technical_docs": "API documentation, READMEs, technical writing",
            "data_analysis": "Analyzing datasets, generating insights, statistics",
            "sql_generation": "Database queries, schema design, query optimization",
            "math_reasoning": "Complex calculations, proofs, mathematical problems",
            "creative_writing": "Stories, marketing copy, creative content",
            "summarization": "Summarizing documents, articles, meeting notes",
            "translation": "Language translation between languages",
            "legal_review": "Contract review, regulatory compliance, legal analysis",
            "financial_analysis": "Financial modeling, market analysis, risk assessment",
            "research_assistant": "Academic research, literature review, scientific analysis",
            "customer_support": "Chatbots, help desk, customer service",
            "tutoring": "Educational explanations, teaching, learning assistance",
            "general_qa": "General question answering, factual queries",
            "rag_pipeline": "Retrieval-augmented generation with external documents",
            "function_calling": "Tool use, API integration, function execution",
            "structured_extraction": "JSON extraction, form filling, schema-based output",
            "long_context": "Processing very long documents (100K+ tokens)",
            "cost_optimized": "Budget-conscious applications, high-volume processing",
            "low_latency": "Real-time applications, interactive chat, streaming",
            "maximum_quality": "Best possible results regardless of cost",
        }
        return descriptions.get(use_case, "General purpose task")


# Convenience function for quick classification
def classify_prompt(prompt: str) -> ClassificationResult:
    """Quick classification helper."""
    classifier = PromptClassifier()
    return classifier.classify(prompt)

