"""
DAG Adaptive Optimizer
Dynamically optimizes execution plans during DAG execution

Core Features:
1. Node Skipping Decision - Determines whether retrieval can be skipped for direct answer generation
2. Final Layer Completeness Check - Checks if the answer is complete and if additional information is needed

Design Principles:
- Unified Decision Center: All optimization decisions are centralized in this module
- Layer-based Triggering: Optimization checks are triggered after each layer execution
- Configurability: All thresholds and switches can be adjusted through configuration
"""
from typing import Dict, List, Any, Optional, Tuple
from config.config import config
from src.core.dag_models import (
    QueryDAG, DAGNode, NodeExecutionMode,
    OptimizationResult
)
from src.core.memory import Memory
from src.utils.output_manager import output_manager
from src.utils.llm_output_cleaner import extract_json_from_llm_output


class DAGAdaptiveOptimizer:
    """DAG Adaptive Optimizer"""

    def __init__(self, llm_client, query_router):
        """
        Initialize optimizer

        Args:
            llm_client: LLM client
            query_router: QueryRouter instance (kept for compatibility, but skip logic has been moved here)
        """
        self.llm_client = llm_client
        self.query_router = query_router

        # Read optimizer settings from configuration
        opt_config = config.dag.adaptive_optimizer

        self.enabled = opt_config.enabled

        # Node skipping configuration
        self.skip_enabled = opt_config.skip_enabled
        self.skip_confidence_threshold = opt_config.skip_confidence_threshold

        # Final layer completeness check configuration
        self.final_completion_check_enabled = opt_config.final_completion_check_enabled
        self.final_insertion_max = opt_config.final_insertion_max
        self.final_completion_threshold = opt_config.final_completion_threshold
        self.final_completion_llm_temperature = opt_config.final_completion_llm_temperature

        # Counters
        self.final_insertions_count = 0

        if self.enabled:
            output_manager.info(f"[DAGAdaptiveOptimizer] ✓ Optimizer enabled")
            output_manager.info(f"  - Node skipping: {'Enabled' if self.skip_enabled else 'Disabled'} (threshold: {self.skip_confidence_threshold})")
            output_manager.info(f"  - Completeness check: {'Enabled' if self.final_completion_check_enabled else 'Disabled'} (threshold: {self.final_completion_threshold})")
        else:
            output_manager.info(f"[DAGAdaptiveOptimizer] ✗ Optimizer disabled")

    # ========== Main Entry Point ==========

    def optimize_after_layer(
        self,
        current_rank: int,
        total_ranks: int,
        dag: QueryDAG,
        memory: Memory
    ) -> OptimizationResult:
        """
        Optimization entry point after each layer execution
        Simplified version: Only retains final layer completeness check, node skipping logic removed

        Args:
            current_rank: Current completed layer (0-based)
            total_ranks: Total number of layers
            dag: DAG object
            memory: Memory object

        Returns:
            OptimizationResult: Contains completeness check decisions
        """
        if not self.enabled:
            return OptimizationResult()

        output_manager.info(f"\n{'='*60}")
        output_manager.info(f"[Optimizer] Checking optimization opportunities after Rank {current_rank}/{total_ranks-1} execution...")
        output_manager.info(f"{'='*60}")

        result = OptimizationResult()

        # Node skipping logic removed, only final layer completeness check retained
        # Node type switching is now determined by NodeAgent during execution

        # Final layer completeness check
        if self.final_completion_check_enabled and current_rank == total_ranks - 1:
            if self.final_insertions_count < self.final_insertion_max:
                needs_supplement, missing_info, supplement_question = self._check_final_completeness(dag, memory)
                if needs_supplement:
                    result.needs_final_insertion = True
                    result.final_insertion_reason = missing_info
                    result.final_insertion_question = supplement_question
                    self.final_insertions_count += 1
                    output_manager.info(f"[Optimizer] ⚠️ Supplementary information needed: {missing_info}")
                else:
                    output_manager.info(f"[Optimizer] ✓ Answer is complete, no supplement needed")
            else:
                output_manager.info(f"[Optimizer] Final layer insertion limit reached ({self.final_insertion_max})")

        if not result.should_apply_optimizations():
            output_manager.info(f"[Optimizer] No optimization operations needed")

        output_manager.info(f"{'='*60}\n")

        return result

    # ========== Deprecated Node Skipping Logic (Moved to NodeAgent) ==========
    # The following methods are deprecated, kept only to avoid potential reference errors

    def _decide_execution_modes(self, *args, **kwargs):
        """Deprecated: Node skipping decision has been moved to NodeAgent dynamic switching mechanism"""
        output_manager.warning("[Optimizer] _decide_execution_modes is deprecated, please use NodeAgent's dynamic switching")
        return {}

    def _should_skip_retrieval(self, *args, **kwargs):
        """Deprecated: Retrieval judgment has been moved to QueryRouter.judge_retrieval_necessity"""
        output_manager.warning("[Optimizer] _should_skip_retrieval is deprecated, please use QueryRouter.judge_retrieval_necessity")
        return {"need_retrieval": True, "reason": "Method deprecated", "confidence": 0.0}

    # ========== Feature: Final Layer Completeness Check ==========

    def _check_final_completeness(self, dag: QueryDAG, memory: Memory) -> Tuple[bool, str, str]:
        """
        Check answer completeness after final layer execution

        Returns:
            (needs_supplement, missing_info, supplement_question)
            - needs_supplement: Whether supplementary information is needed
            - missing_info: Description of missing information
            - supplement_question: Suggested supplementary query question
        """
        original_query = dag.original_query
        completed_nodes = memory.get_all_results()

        if not completed_nodes:
            return False, "", ""

        # Build summary of completed node answers
        completed_summary = self._format_completed_answers(completed_nodes)

        prompt = f"""You are an answer completeness evaluation expert. Please determine whether the currently completed subtask answers are sufficiently complete to answer the original question.

[Original Question]
{original_query}

[Completed Subtasks and Answers]
{completed_summary}

[Evaluation Criteria]
1. Has core information been obtained (key data required by the original question)
2. Is the answer sufficiently detailed and complete
3. Is more data needed to completely answer the question

[Output Requirements]
Please return strict JSON format (do not include code block markers):
{{
    "is_complete": true/false,
    "confidence": 0.0-1.0,
    "reason": "Justification for why it is complete/incomplete",
    "missing_info": "If incomplete, what key information is still missing",
    "supplement_question": "If incomplete, recommended supplementary query question"
}}

Note:
- Only return is_complete=true when highly confident (confidence >= {self.final_completion_threshold}) and all key information has been obtained
- If there is any doubt, conservatively choose is_complete=false
- supplement_question should be a directly executable query question
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.final_completion_llm_temperature,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            decision = extract_json_from_llm_output(result_text)

            if not decision:
                output_manager.warning("[Optimizer-Completeness] Unable to parse JSON from LLM response, conservatively judging as complete")
                return False, "", ""

            is_complete = decision.get('is_complete', True)
            confidence = decision.get('confidence', 1.0)
            reason = decision.get('reason', '')
            missing_info = decision.get('missing_info', '')
            supplement_question = decision.get('supplement_question', '')

            output_manager.info(f"[Optimizer-Completeness] Completeness judgment:")
            output_manager.info(f"  Complete: {is_complete}")
            output_manager.info(f"  Confidence: {confidence:.2f}")
            output_manager.info(f"  Reason: {reason}")

            if not is_complete and missing_info:
                output_manager.info(f"  Missing info: {missing_info}")
                output_manager.info(f"  Suggested supplement: {supplement_question}")

            # Return whether supplement is needed
            needs_supplement = (not is_complete) and (confidence >= self.final_completion_threshold)
            return needs_supplement, missing_info, supplement_question

        except Exception as e:
            output_manager.error(f"[Optimizer-Completeness] Completeness judgment failed: {e}")
            return False, "", ""

    # ========== Helper Methods ==========

    def _build_parent_context(self, node: DAGNode, memory: Memory) -> str:
        """Build parent node answer context"""
        parent_answers = memory.read_parent_results(node.parent_ids)

        context_parts = []
        for parent_id, result in parent_answers.items():
            if result and result.answer_text:
                context_parts.append(f"Parent node {parent_id} answer: {result.answer_text}")

        return "\n".join(context_parts)

    def _format_completed_answers(self, completed_nodes: Dict[str, Any]) -> str:
        """Format completed node answers"""
        lines = []
        for node_id, result in completed_nodes.items():
            if hasattr(result, 'original_question') and hasattr(result, 'answer_text'):
                lines.append(f"Node {node_id}:")
                lines.append(f"  Question: {result.original_question}")
                lines.append(f"  Answer: {result.answer_text}")
                lines.append("")

        return "\n".join(lines)

    def _get_nodes_at_rank(self, dag: QueryDAG, rank: int) -> List[DAGNode]:
        """Get all nodes at specified rank"""
        nodes = []
        for node in dag.nodes.values():
            if node.topological_rank == rank:
                nodes.append(node)
        return nodes

    # ========== Statistics and Debugging ==========

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        return {
            "enabled": self.enabled,
            "skip_enabled": self.skip_enabled,
            "skip_confidence_threshold": self.skip_confidence_threshold,
            "final_completion_check_enabled": self.final_completion_check_enabled,
            "final_insertions_count": self.final_insertions_count
        }

    def reset_counters(self):
        """Reset counters (called before each new query)"""
        self.final_insertions_count = 0
