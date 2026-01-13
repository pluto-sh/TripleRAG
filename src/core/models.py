"""
Triple RAG Data Model Definitions
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import time

class QueryType(Enum):
    """Query type enumeration"""
    MYSQL = "mysql"
    NEO4J = "neo4j"
    VECTOR = "vector"
    HYBRID = "hybrid"

@dataclass
class RetrievalMetadata:
    """Retrieval metadata"""
    query_statement: str
    execution_time: float
    result_count: int
    source_type: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class RetrievalResult:
    """Retrieval result"""
    content: str
    score: float
    metadata: RetrievalMetadata
    source_id: Optional[str] = None
    source: str = ""

@dataclass
class QueryPlan:
    """Query plan - removed specific query statements, let LLM decide autonomously during execution"""
    query_types: List[QueryType]
    inference: str
    # Optional fusion weights (data source to weight mapping)
    weights: Optional[Dict[str, float]] = None

    def to_dict(self):
        return {
            "query_types": [qt.value for qt in self.query_types],
            "inference": self.inference,
            "weights": self.weights or {}
        }

@dataclass
class ConflictInfo:
    """Conflict information"""
    has_conflict: bool
    conflicting_results: List[RetrievalResult]
    resolution: str = ""
    confidence: float = 0.0

@dataclass
class FusedResult:
    """Fused result"""
    content: str
    sources: List[str]
    confidence: float
    metadata_summary: Dict[str, Any]
    conflict_info: Optional[ConflictInfo] = None
    score: float = 0.0
    understood_question: str = ""  # LLM-understood specific question

    def __post_init__(self):
        if self.score == 0.0:
            self.score = self.confidence

@dataclass
class TripleRAGResponse:
    """Triple RAG final response"""
    query: str
    answer: str
    fused_results: List[FusedResult]
    query_plan: QueryPlan
    total_execution_time: float
    explanation: str = ""
    retrieval_results: List[RetrievalResult] = None
    final_answer: str = ""
    retrieved_contexts: List[str] = None  # Store all original retrieval contexts

    def __post_init__(self):
        if self.retrieval_results is None:
            self.retrieval_results = []
        if self.retrieved_contexts is None:
            self.retrieved_contexts = []
        if not self.final_answer:
            self.final_answer = self.answer
