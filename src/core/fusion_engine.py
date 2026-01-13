"""
Explainable Fusion Module
LLM-driven, references retrieval results and metadata from different modalities, handles conflicts and fuses results
"""
import json
from typing import List, Dict, Any
from openai import OpenAI
from .models import RetrievalResult, FusedResult, ConflictInfo, QueryType
from config.config import config
from ..utils.llm_client import build_openai_client

class FusionEngine:
    """Fusion Engine"""

    def __init__(self):
        self.client = build_openai_client()

    def fuse_results(self, query: str, mysql_results: List[RetrievalResult],
                    sqlite_graph_results: List[RetrievalResult], vector_results: List[RetrievalResult],
                    weights: Dict[str, float], parent_context: str = "") -> List[FusedResult]:
        """
        Fuse retrieval results from different modalities (optimized version: single LLM call)

        Args:
            query: query question
            mysql_results: MySQL/SQLite retrieval results
            sqlite_graph_results: Neo4j graph database retrieval results
            vector_results: vector retrieval results
            weights: weight configuration (deprecated, kept for interface compatibility)
            parent_context: parent node context

        Returns:
            list of fused results
        """

        print(f"Starting result fusion: SQL={len(mysql_results)}, Graph={len(sqlite_graph_results)}, Vector={len(vector_results)}")

        # Step 1: Simple deduplication (based on content similarity, non-LLM)
        deduplicated_results = self._simple_deduplicate(
            mysql_results, sqlite_graph_results, vector_results
        )
        print(f"After deduplication: {len(deduplicated_results)} results")

        if not deduplicated_results:
            print("All channel results are empty")
            return []

        # Step 2: Single LLM call for fusion (including dynamic priority)
        fused_result = self._llm_direct_fusion(
            query=query,
            mysql_results=mysql_results,
            neo4j_results=sqlite_graph_results,
            vector_results=vector_results,
            parent_context=parent_context
        )

        print(f"Fusion complete: generated 1 fused result")

        return [fused_result]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simple implementation)"""
        if not text1 or not text2:
            return 0.0

        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _simple_deduplicate(self, mysql_results: List[RetrievalResult],
                           neo4j_results: List[RetrievalResult],
                           vector_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Simple deduplication: based on content similarity (non-LLM)

        Args:
            mysql_results: SQL retrieval results
            neo4j_results: graph retrieval results
            vector_results: vector retrieval results

        Returns:
            deduplicated result list
        """
        all_results = mysql_results + neo4j_results + vector_results

        if len(all_results) <= 1:
            return all_results

        deduplicated = []

        for result in all_results:
            is_duplicate = False

            for existing in deduplicated:
                # Similarity threshold 0.7 (lower threshold for stronger deduplication)
                if self._calculate_similarity(result.content, existing.content) > 0.7:
                    is_duplicate = True
                    # Keep result with higher score
                    if result.score > existing.score:
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    break

            if not is_duplicate:
                deduplicated.append(result)

        print(f"   Deduplication: {len(all_results)} → {len(deduplicated)} results")
        return deduplicated

    def _llm_direct_fusion(self, query: str,
                          mysql_results: List[RetrievalResult],
                          neo4j_results: List[RetrievalResult],
                          vector_results: List[RetrievalResult],
                          parent_context: str = "") -> FusedResult:
        """
        Single LLM call for fusion (core optimization)

        Args:
            query: query question
            mysql_results: SQL retrieval results
            neo4j_results: graph retrieval results
            vector_results: vector retrieval results
            parent_context: parent node context

        Returns:
            single fused result
        """

        # Dynamically determine priority
        priority_info = self._determine_priority(mysql_results, neo4j_results, vector_results)

        # Build three-channel result text
        mysql_text = self._format_results(mysql_results, "SQL")
        neo4j_text = self._format_results(neo4j_results, "Graph Database")
        vector_text = self._format_results(vector_results, "Vector Retrieval")

        # Build prompt
        system_prompt = """You are an intelligent data fusion assistant responsible for integrating retrieval results from multiple data sources.

**Fusion Principles**:
1. Priority: Structured database information > Graph database information > Text vector retrieval information (but adjust flexibly based on actual situation)
2. If one channel has no results, automatically use other channels
3. If multiple channels have relevant information, consider comprehensively, mainly use the most reliable one
4. Remove duplicate information, keep the most accurate and complete content

**Important Note**: You need to output a structured JSON response, including the specific question you understood and the answer after fusing the three-channel retrieval results.

**Output Requirements**:
1. understood_question: Based on the original question and retrieval context, the specific problem you understood that needs to be addressed
2. fused_content: The fused content, without prefix (such as "According to retrieval results")
3. Keep professional terms in their original names (e.g., "CATL" should not be translated)
4. Briefly explain the fusion inference (1 sentence)

**Output JSON Format**:
{
  "understood_question": "Based on context information, the specific problem you understood that needs to be addressed",
  "fused_content": "The fused content",
  "inference": "Fusion inference (1 sentence)",
  "primary_source": "The primary data source used"
}"""

        user_prompt = f"""**Query Question**: {query}

**Priority Hint**: {priority_info}

**Parent Node Context**:
{parent_context if parent_context else "None"}

**Structured Database Retrieval Results**:
{mysql_text}

**Graph Database Retrieval Results**:
{neo4j_text}

**Text Vector Retrieval Results**:
{vector_text}

Please fuse the above information and output the result in JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON
            import json
            from ..utils.llm_output_cleaner import safe_parse_json

            fusion_data = safe_parse_json(result_text)

            if not fusion_data:
                raise ValueError("LLM returned JSON parsing failed")

            # Extract structured data (including question understanding)
            understood_question = fusion_data.get("understood_question", query)
            fused_content = fusion_data.get("fused_content", "")
            inference = fusion_data.get("inference", "")
            primary_source = fusion_data.get("primary_source", "unknown")

            # Determine actually used data sources
            sources_used = []
            if mysql_results:
                sources_used.append("mysql")
            if neo4j_results:
                sources_used.append("sqlite_graph")
            if vector_results:
                sources_used.append("vector")

            # Calculate average confidence
            all_results = mysql_results + neo4j_results + vector_results
            avg_confidence = sum(r.score for r in all_results) / len(all_results) if all_results else 0.5

            fused_result = FusedResult(
                content=fused_content,
                sources=sources_used,
                confidence=avg_confidence,
                understood_question=understood_question,  # Include LLM understood question
                metadata_summary={
                    "primary_source": primary_source,
                    "inference": inference,
                    "total_results": len(all_results),
                    "mysql_count": len(mysql_results),
                    "neo4j_count": len(neo4j_results),
                    "vector_count": len(vector_results)
                },
                conflict_info=None  # No longer handle conflicts separately
            )

            print(f"   LLM understood question: {understood_question}")
            print(f"   Fusion inference: {inference}")
            print(f"   Primary source: {primary_source}")

            return fused_result

        except Exception as e:
            print(f"LLM fusion failed: {e}")
            # Fallback: simply concatenate all results (including basic question understanding)
            return self._fallback_fusion(mysql_results, neo4j_results, vector_results, query)

    def _determine_priority(self, mysql_results: List[RetrievalResult],
                           neo4j_results: List[RetrievalResult],
                           vector_results: List[RetrievalResult]) -> str:
        """
        Dynamically determine priority strategy

        Args:
            mysql_results: SQL results
            neo4j_results: graph results
            vector_results: vector results

        Returns:
            priority description text
        """
        has_mysql = len(mysql_results) > 0
        has_neo4j = len(neo4j_results) > 0
        has_vector = len(vector_results) > 0

        if has_mysql and has_neo4j and has_vector:
            return "All channels have results, default priority: SQL > Graph > Vector"
        elif has_mysql and has_neo4j:
            return "SQL and graph channels have results, prioritize SQL, graph as supplement"
        elif has_mysql and has_vector:
            return "SQL and vector channels have results, prioritize SQL, vector as supplement"
        elif has_neo4j and has_vector:
            return "Graph and vector channels have results, prioritize graph, vector as supplement"
        elif has_mysql:
            return "Only SQL channel has results, fully based on SQL"
        elif has_neo4j:
            return "Only graph channel has results, fully based on graph"
        elif has_vector:
            return "Only vector channel has results, fully based on vector"
        else:
            return "All channels have no results"

    def _format_results(self, results: List[RetrievalResult], source_name: str) -> str:
        """
        Format retrieval results as text

        Args:
            results: retrieval result list
            source_name: data source name

        Returns:
            formatted text
        """
        if not results:
            return f"({source_name} has no results)"

        formatted_lines = []
        for i, result in enumerate(results, 1):
            formatted_lines.append(f"{i}. {result.content}")

        return "\n".join(formatted_lines)

    def _fallback_fusion(self, mysql_results: List[RetrievalResult],
                        neo4j_results: List[RetrievalResult],
                        vector_results: List[RetrievalResult],
                        query: str = "") -> FusedResult:
        """
        Fallback fusion solution: simple concatenation (when LLM fails)

        Args:
            mysql_results: SQL results
            neo4j_results: graph results
            vector_results: vector results

        Returns:
            concatenated fusion result
        """
        all_results = mysql_results + neo4j_results + vector_results

        if not all_results:
            return FusedResult(
                content="No relevant information found",
                sources=[],
                confidence=0.0,
                understood_question=query or "Unknown question",  # basic question understanding
                metadata_summary={"fallback": True},
                conflict_info=None
            )

        # Sort by priority: MySQL > Neo4j > Vector
        priority_results = []
        priority_results.extend(mysql_results)
        priority_results.extend(neo4j_results)
        priority_results.extend(vector_results)

        # Concatenate first 3 results
        content_parts = [r.content for r in priority_results[:3]]
        fused_content = " | ".join(content_parts)

        sources_used = []
        if mysql_results:
            sources_used.append("mysql")
        if neo4j_results:
            sources_used.append("sqlite_graph")
        if vector_results:
            sources_used.append("vector")

        avg_confidence = sum(r.score for r in all_results) / len(all_results)

        return FusedResult(
            content=fused_content,
            sources=sources_used,
            confidence=avg_confidence,
            understood_question=query or "Processed query question",  # basic question understanding
            metadata_summary={"fallback": True, "method": "simple_concat"},
            conflict_info=None
        )
    
    
