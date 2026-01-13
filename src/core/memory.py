"""
Memory Mechanism
Responsible for information passing and storage between nodes
"""
from typing import Dict, List, Any, Optional
import json
import time
from src.core.dag_models import UnifiedNodeResult
from config.config import config


class Memory:
    """
    Global Memory for information passing between DAG nodes

    Features:
    - Basic read/write operations
    - Storage indexed by node ID
    - Vector context pruning
    - Automatic compression
    - Persistence
    """

    def __init__(self):
        """Initialize Memory"""
        self._storage: Dict[str, UnifiedNodeResult] = {}
        self._access_log: List[Dict[str, Any]] = []  # access log (for debugging)

        # Vector-related
        self.vector_usage_count: int = 0
        self.vector_summary: Optional[str] = None

        self.created_at = time.time()

    def write_node_result(self, node_id: str, result: UnifiedNodeResult):
        """
        Write node execution result

        Args:
            node_id: node ID
            result: node execution result
        """
        if node_id != result.node_id:
            raise ValueError(f"Node ID mismatch: {node_id} != {result.node_id}")

        self._storage[node_id] = result

        # record access log
        self._log_access("write", node_id)

        print(f"[Memory] Wrote result for node {node_id}")

    def read_node_result(self, node_id: str) -> Optional[UnifiedNodeResult]:
        """
        Read single node result

        Args:
            node_id: node ID

        Returns:
            node result, or None if not exists
        """
        result = self._storage.get(node_id)

        if result:
            self._log_access("read", node_id)
            print(f"[Memory] Read result for node {node_id}")
        else:
            print(f"[Memory] Result for node {node_id} does not exist")

        return result

    def read_parent_results(self, parent_ids: List[str]) -> Dict[str, UnifiedNodeResult]:
        """
        Batch read parent node results

        Args:
            parent_ids: list of parent node IDs

        Returns:
            dictionary in format {node_id: result}
        """
        results = {}

        for parent_id in parent_ids:
            result = self.read_node_result(parent_id)
            if result:
                results[parent_id] = result
            else:
                print(f"[Memory] Warning: result for parent node {parent_id} does not exist")

        return results

    def has_node_result(self, node_id: str) -> bool:
        """
        Check if node result exists

        Args:
            node_id: node ID

        Returns:
            whether exists
        """
        return node_id in self._storage

    def get_all_results(self) -> Dict[str, UnifiedNodeResult]:
        """
        Get all stored results

        Returns:
            dictionary of all results
        """
        return self._storage.copy()

    def get_all_answers(self) -> Dict[str, str]:
        """
        Get answer text from all nodes

        Returns:
            dictionary in format {node_id: answer_text}
        """
        return {
            node_id: result.answer_text
            for node_id, result in self._storage.items()
        }

    def get_all_structured_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get structured data from all nodes (deprecated, kept for backward compatibility)

        Returns:
            empty dictionary (structured data has been removed)
        """
        # structured data has been removed, return empty dict for backward compatibility
        return {node_id: {} for node_id in self._storage.keys()}
    
    def build_context_for_node(self, parent_ids: List[str], dag: Optional["QueryDAG"] = None, to_node_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Build query context for node (semantic version)
        Only keeps key information channels, completely removes old constraint output fields
        """
        parent_results = self.read_parent_results(parent_ids)
        context = {
            "parent_answers": {},          # complete answers (for inter-node passing)
            "parent_ids": parent_ids,
            "has_parents": len(parent_results) > 0,
            "is_multi_input": len(parent_ids) > 1,
        }
        for parent_id, result in parent_results.items():
            context["parent_answers"][parent_id] = result.answer_text
        return context



    def _merge_multi_inputs(self, parent_results: Dict[str, UnifiedNodeResult], multi_constraint) -> Dict[str, Any]:
        """
        Intelligently merge inputs from multiple parent nodes

        Args:
            parent_results: parent node execution results
            multi_constraint: multi-input constraint object

        Returns:
            merged input dictionary
        """
        try:
            from src.core.dag_models import MultiInputConstraint
            if not isinstance(multi_constraint, MultiInputConstraint):
                return

            # use multi-input constraint to extract and merge data
            merged = multi_constraint.extract_inputs(parent_results)

            # enhanced merge: add statistics info
            merged["_merge_info"] = {
                "source_count": len(parent_results),
                "source_nodes": list(parent_results.keys()),
                "merge_strategy": multi_constraint.merge_strategy,
                "required_inputs": multi_constraint.required_inputs,
                "optional_inputs": multi_constraint.optional_inputs
            }

            print(f"[Memory] Multi-input merge complete: {len(parent_results)} sources → {len(merged)} fields")
            return merged

        except Exception as e:
            print(f"[Memory] Multi-input merge failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def build_multi_input_context_text(self, parent_results: Dict[str, UnifiedNodeResult], multi_constraint) -> str:
        """
        Build text context for multi-input nodes

        Args:
            parent_results: parent node execution results
            multi_constraint: multi-input constraint

        Returns:
            formatted multi-input context text
        """
        try:
            from src.core.dag_models import MultiInputConstraint
            if not isinstance(multi_constraint, MultiInputConstraint):
                return self._build_simple_multi_context_text(parent_results)

            context_parts = ["[Multi-source Input Information]"]

            # organize information according to constraint requirements
            for node_id, result in parent_results.items():
                if node_id in multi_constraint.input_mappings:
                    required_fields = multi_constraint.input_mappings[node_id]
                    context_parts.append(f"\nSource node {node_id}:")
                    context_parts.append(f"  Answer: {result.answer_text}")

                    # show constraint-required fields
                    data = getattr(result, 'constraint_outputs', {}) or getattr(result, 'structured_data', {})
                    context_parts.append("  Key information:")
                    for field in required_fields:
                        value = data.get(field, "Not found")
                        context_parts.append(f"    {field}: {value}")
                else:
                    # non-constraint node, simple display
                    context_parts.append(f"\nReference node {node_id}:")
                    context_parts.append(f"  Answer: {result.answer_text}")

            # add constraint info
            if multi_constraint.output_requirements:
                context_parts.append(f"\nRequired output fields: {', '.join(multi_constraint.output_requirements)}")

            return "\n".join(context_parts)

        except Exception as e:
            print(f"[Memory] Build multi-input context failed: {e}")
            return self._build_simple_multi_context_text(parent_results)

    def _build_simple_multi_context_text(self, parent_results: Dict[str, UnifiedNodeResult]) -> str:
        """Simple multi-input context building (fallback method)"""
        context_parts = ["[Information from Multiple Previous Steps]"]

        for node_id, result in parent_results.items():
            context_parts.append(f"\nResult from {node_id}:")
            context_parts.append(f"  {result.answer_text}")

        return "\n".join(context_parts)

    def get_multi_input_summary(self, parent_ids: List[str]) -> Dict[str, Any]:
        """
        Get summary information for multi-input

        Args:
            parent_ids: list of parent node IDs

        Returns:
            multi-input summary dictionary
        """
        if len(parent_ids) <= 1:
            return {"is_multi_input": False}

        parent_results = self.read_parent_results(parent_ids)

        # statistics info
        total_chars = sum(len(r.answer_text) for r in parent_results.values())
        # structured data has been removed, set to 0 for compatibility
        total_fields = 0

        # check input conflicts (structured data related logic removed)
        all_fields = set()
        field_conflicts = []

        return {
            "is_multi_input": True,
            "source_count": len(parent_results),
            "successful_sources": len(parent_results),  # number of results that can be read
            "total_text_length": total_chars,
            "total_fields": total_fields,
            "unique_fields": len(all_fields),
            "field_conflicts": field_conflicts,
            "source_channels": list(set(
                channel for result in parent_results.values()
                for channel in getattr(result, 'channels_used', [])
            )),
            "complexity_score": len(parent_ids) * 0.3  # simplified complexity score (field-related removed)
        }

    def clear(self):
        """Clear all data"""
        self._storage.clear()
        self._access_log.clear()
        self.vector_usage_count = 0
        self.vector_summary = None
        print("[Memory] Cleared all data")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Memory statistics

        Returns:
            statistics dictionary
        """
        total_nodes = len(self._storage)

        return {
            "total_nodes": total_nodes,
            "node_ids": list(self._storage.keys()),
            "total_accesses": len(self._access_log),
            "vector_usage_count": self.vector_usage_count,
            "created_at": self.created_at,
            "lifetime": time.time() - self.created_at
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export to dictionary format (for serialization)

        Returns:
            serializable dictionary
        """
        return {
            "results": {
                node_id: result.to_dict()
                for node_id, result in self._storage.items()
            },
            "statistics": self.get_statistics()
        }

    def export_to_json(self) -> str:
        """
        Export to JSON string

        Returns:
            JSON string
        """
        return json.dumps(self.export_to_dict(), ensure_ascii=False, indent=2)

    def _log_access(self, operation: str, node_id: str):
        """
        Record access log (internal method)

        Args:
            operation: operation type (read/write)
            node_id: node ID
        """
        self._access_log.append({
            "operation": operation,
            "node_id": node_id,
            "timestamp": time.time()
        })

    def get_access_log(self) -> List[Dict[str, Any]]:
        """
        Get access log

        Returns:
            access log list
        """
        return self._access_log.copy()

    def print_memory_content(self):
        """Print Memory content (for debugging)"""
        print("\n" + "="*60)
        print("Memory Content:")
        print("="*60)

        if not self._storage:
            print("(Empty)")
        else:
            for node_id, result in self._storage.items():
                print(f"\nNode {node_id}:")

                # adapt to new UnifiedNodeResult structure
                if hasattr(result, 'original_question'):
                    print(f"  Original question: {result.original_question}")
                if hasattr(result, 'actual_question'):
                    print(f"  Actual processed question: {result.actual_question}")
                else:
                    # backward compatible with old question field
                    question = getattr(result, 'question', result.original_question if hasattr(result, 'original_question') else 'Unknown')
                    print(f"  Question: {question}")

                print(f"  Answer: {result.answer_text}")
                print(f"  Channels: {result.channels_used}")

                # confidence may have been removed, use getattr for safe access
                confidence = getattr(result, 'confidence', 0.0)
                print(f"  Confidence: {confidence:.2f}")

                # structured data has been removed, but keep check for backward compatibility
                structured_data = getattr(result, 'structured_data', {})
                if structured_data:
                    print(f"  Structured data: {structured_data}")
                else:
                    print(f"  Structured data: None (optimized and removed)")

        print("="*60 + "\n")


    def increment_vector_usage(self):
        """
        Increment Vector usage count
        Used to trigger pruning
        """
        self.vector_usage_count += 1

    def get_vector_context(self) -> Optional[str]:
        """
        Get Vector context

        Returns:
            Vector context summary, or None if not available
        """

        return self.vector_summary

    def compress_vector_context(self, llm_client) -> str:
        """
        Compress Vector context (using LLM)

        Args:
            llm_client: LLM client

        Returns:
            compressed summary
        """

        print("[Memory] Warning: compress_vector_context not implemented")
        return ""