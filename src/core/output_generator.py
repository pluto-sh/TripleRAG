"""
Retrieval Results and Explainability Output Module
Handles result formatting and DAG multi-node answer aggregation
"""
from typing import List, Dict, Any
from openai import OpenAI
from .models import FusedResult, TripleRAGResponse, QueryPlan, RetrievalResult, QueryType
from config.config import config
from ..utils.llm_client import build_openai_client
from ..utils.output_manager import output_manager

class OutputGenerator:
    """Output Generator"""

    def __init__(self):
        self.client = build_openai_client()

    def generate_response(self, query: str, fused_results: List[FusedResult],
                         query_plan: QueryPlan, total_execution_time: float,
                         retrieval_results: List[RetrievalResult] = None) -> TripleRAGResponse:
        """Generate final response"""

        # Generate main answer
        answer = self._generate_answer(query, fused_results)

        # Generate explanation
        explanation = self._generate_explanation(query, fused_results, query_plan, total_execution_time)

        # Create response object
        response = TripleRAGResponse(
            query=query,
            answer=answer,
            fused_results=fused_results,
            query_plan=query_plan,
            total_execution_time=total_execution_time,
            explanation=explanation,
            retrieval_results=retrieval_results or []
        )

        return response

    def _generate_answer(self, query: str, fused_results: List[FusedResult], query_plan: QueryPlan = None, parent_context: str = "") -> str:
        """
        Simplified answer generation - directly use fusion_engine output, avoid duplicate LLM calls

        Args:
            query: original query
            fused_results: fused results (from fusion_engine)
            query_plan: query plan (optional)
            parent_context: parent node context (optional)

        Returns:
            structured response dictionary {understood_question, answer}
        """

        # Handle empty fusion results case (skip retrieval)
        if not fused_results:
            # Handle empty fusion results case (skip retrieval)
            if parent_context and parent_context.strip():
                # If there is parent node context, answer based on context
                answer = f"Based on the previous step information: {parent_context[:200]}{'...' if len(parent_context) > 200 else ''}"
            else:
                # No context case, provide basic answer
                answer = f"Unable to provide specific answer based on current query: {query}"

            return {
                'understood_question': query,
                'answer': answer
            }

        # Directly use fusion_engine's structured output (this is the main flow)
        fused_result = fused_results[0]  # take first fused result

        # fusion_engine already provides complete understood question and answer content
        understood_question = getattr(fused_result, 'understood_question', query)
        answer_content = getattr(fused_result, 'content', '')

        return {
            'understood_question': understood_question,
            'answer': answer_content
        }


    def _generate_explanation(self, query: str, fused_results: List[FusedResult],
                            query_plan: QueryPlan, total_execution_time: float) -> str:
        """Generate explanation"""

        explanation_parts = []

        # Query analysis explanation
        explanation_parts.append("## Query Analysis")
        explanation_parts.append(f"**Query Types**: {', '.join([qt.value for qt in query_plan.query_types])}")
        explanation_parts.append(f"**Analysis Inference**: {query_plan.inference}")

        if query_plan.weights:
            weights_str = ", ".join([f"{k}: {v:.1%}" for k, v in query_plan.weights.items()])
            explanation_parts.append(f"**Weight Allocation**: {weights_str}")

        # Retrieval process explanation
        explanation_parts.append("\n## Retrieval Process")

        # Query information is already reflected in query types, no longer display specific query statements separately

        # Result statistics
        explanation_parts.append("\n## Retrieval Results")
        explanation_parts.append(f"**Total Execution Time**: {total_execution_time:.3f} seconds")
        explanation_parts.append(f"**Fused Result Count**: {len(fused_results)}")

        # Data source statistics
        all_sources = []
        for result in fused_results:
            all_sources.extend(result.sources)

        source_counts = {}
        for source in all_sources:
            source_counts[source] = source_counts.get(source, 0) + 1

        if source_counts:
            source_stats = ", ".join([f"{k}: {v}" for k, v in source_counts.items()])
            explanation_parts.append(f"**Data Source Distribution**: {source_stats}")

        # Conflict handling explanation
        conflicts = [r for r in fused_results if r.conflict_info and r.conflict_info.has_conflict]
        if conflicts:
            explanation_parts.append("\n## Conflict Handling")
            for i, result in enumerate(conflicts):
                explanation_parts.append(f"**Conflict {i+1}**: {result.conflict_info.resolution}")
                explanation_parts.append(f"**Confidence**: {result.conflict_info.confidence:.1%}")

        # Credibility assessment
        explanation_parts.append("\n## Credibility Assessment")
        avg_confidence = sum(r.confidence for r in fused_results) / len(fused_results) if fused_results else 0
        explanation_parts.append(f"**Average Confidence**: {avg_confidence:.1%}")

        confidence_level = "High" if avg_confidence > 0.8 else "Medium" if avg_confidence > 0.5 else "Low"
        explanation_parts.append(f"**Credibility Level**: {confidence_level}")

        return "\n".join(explanation_parts)
    
    def format_detailed_response(self, response: TripleRAGResponse) -> str:
        """Format detailed response"""

        output_parts = []

        # Title
        output_parts.append("# Triple RAG Retrieval Results")
        output_parts.append(f"**Query**: {response.query}")
        output_parts.append("")

        # Main answer
        output_parts.append("## Answer")
        output_parts.append(response.answer)
        output_parts.append("")

        # Detailed information
        if response.fused_results:
            output_parts.append("## Detailed Information")
            for i, result in enumerate(response.fused_results, 1):
                output_parts.append(f"### Information Source {i}")
                output_parts.append(f"**Content**: {result.content}")
                output_parts.append(f"**Data Sources**: {', '.join(result.sources)}")
                output_parts.append(f"**Confidence**: {result.confidence:.1%}")

                if result.metadata_summary:
                    # Handle case where metadata_summary might be a string
                    if isinstance(result.metadata_summary, dict):
                        output_parts.append(f"**Execution Time**: {result.metadata_summary.get('avg_execution_time', 0):.3f} seconds")
                        output_parts.append(f"**Result Count**: {result.metadata_summary.get('total_results', 0)}")
                    else:
                        # If it's a string, display directly
                        output_parts.append(f"**Metadata**: {result.metadata_summary}")

                output_parts.append("")

        # Explanation
        output_parts.append(response.explanation)

        return "\n".join(output_parts)

    def format_simple_response(self, response: TripleRAGResponse) -> str:
        """Format simple response"""

        output_parts = []

        # Main answer
        output_parts.append(response.answer)
        output_parts.append("")

        # Brief information
        source_types = set()
        if hasattr(response, 'fused_results') and response.fused_results:
            for result in response.fused_results:
                # Ensure result is an object not a string
                if hasattr(result, 'metadata_summary') and isinstance(result.metadata_summary, dict):
                    source_types.update(result.metadata_summary.get('source_types', []))

        if source_types:
            output_parts.append(f"*Information Sources: {', '.join(source_types)}*")

        # Ensure total_execution_time attribute exists
        execution_time = getattr(response, 'total_execution_time', 0.0)
        output_parts.append(f"*Retrieval Time: {execution_time:.3f} seconds*")

        return "\n".join(output_parts)

    def export_to_json(self, response: TripleRAGResponse) -> Dict[str, Any]:
        """Export to JSON format"""

        # Use retrieved_contexts field already in response object
        retrieved_contexts = response.retrieved_contexts if hasattr(response, 'retrieved_contexts') else []

        # If retrieved_contexts is empty, try to extract from other fields (backward compatibility)
        if not retrieved_contexts:
            # Extract context from fused results
            if hasattr(response, 'fused_results') and response.fused_results:
                for result in response.fused_results:
                    if hasattr(result, 'content') and result.content:
                        retrieved_contexts.append(result.content)

            # Extract context from retrieval results (if exists)
            if hasattr(response, 'retrieval_results') and response.retrieval_results:
                for retrieval_result in response.retrieval_results:
                    if hasattr(retrieval_result, 'content') and retrieval_result.content:
                        retrieved_contexts.append(retrieval_result.content)

        return {
            "query": response.query,
            "answer": response.answer,
            "total_execution_time": response.total_execution_time,
            "query_plan": {
                "query_types": [qt.value for qt in response.query_plan.query_types],
                "weights": response.query_plan.weights,
                "inference": response.query_plan.inference
            },
            "fused_results": [
                {
                    "content": result.content,
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "metadata_summary": result.metadata_summary,
                    "has_conflict": result.conflict_info.has_conflict if result.conflict_info else False
                }
                for result in response.fused_results
            ],
            "explanation": response.explanation,
            "retrieved_contexts": retrieved_contexts  # Use real retrieval contexts
        }

    def _get_test_format_prompt(self) -> str:
        """Get system prompt for test format"""
        return """You are an intelligent answer integrator specialized in generating structured answers for test sets.

🎯 **Test Format Requirements**:
1. Analyze the original query and identify the number of sub-questions it contains
2. Provide concise and precise answers for each sub-question in order
3. **Output format must be**: [answer1],[answer2],[answer3]...
4. Each answer is surrounded by square brackets [], multiple answers are separated by commas
5. Answer content should be concise and precise, containing only core values/information, no explanatory text

⚠️ **Important Requirements**:
- Must strictly base on the provided query results, do not add any additional knowledge
- If the query results mention specific values, company names, etc., must use these precise information
- Keep professional terms and company names in their original form

**Example**:
- Query: "What is the proportion of Russian nickel production to global demand? What is the proportion of Russian aluminum production to global demand?"
- Output: "[10%],[6%]"

**Prohibited**:
- Do not output any explanatory text
- Do not use prefixes like "According to the query results"
- Do not include step descriptions"""

    def _get_standard_format_prompt(self) -> str:
        """Get system prompt for standard format"""
        return """You are an intelligent answer integrator responsible for consolidating results from multiple query steps into one accurate final answer.

⚠️ Important Requirements:
1. Must strictly base on the provided query results, do not add any additional knowledge
2. If the query results mention specific company names, data, etc., must use these precise information
3. Maintain clear logic and fully answer the original query
4. If a step does not find results, clearly state it

**Strictly Prohibited**:
- Do not output any additional explanatory information, such as prefixes or suffixes like "According to the query results", "Based on the above data", "Key inference steps"
- Do not include explanatory step descriptions
- Directly provide the final answer, keep it concise and clear

**Term Preservation Requirements**:
- Professional terms, company names, industry names, etc. must maintain the original names retrieved, do not translate or rewrite
- If "Lithium Mining" is retrieved, must use "Lithium Mining"
- If "Tianqi Lithium" is retrieved, must use "Tianqi Lithium"
- If "BYD Company" is retrieved, must use "BYD Company"

Output Requirements:
- Answer in English
- Directly answer the question, avoid any explanatory language
- Strictly base on query results, avoid hallucination"""

    def generate_final_answer(self, original_query: str, all_answers: List[str], all_results: List[Any]) -> str:
        """
        Generate final answer for DAG execution (optimized version)
        Focus on core information: original query + question and answer for each node

        Args:
            original_query: original query
            all_answers: list of all node answers (dictionary format {node_id: answer_text})
            all_results: list of all node results (dictionary format {node_id: UnifiedNodeResult})

        Returns:
            final answer string
        """
        try:
            # Check if test mode is enabled
            is_test_mode = getattr(config, 'test_mode', False)

            # Choose different system prompt based on mode
            if is_test_mode:
                system_prompt = self._get_test_format_prompt()
            else:
                system_prompt = self._get_standard_format_prompt()

            # Build concise context, including question and answer for each node
            context_parts = []

            # Handle case where answers parameter might be a dictionary
            if isinstance(all_answers, dict):
                answers_dict = all_answers
                results_dict = all_results if isinstance(all_results, dict) else {}
            else:
                # If it's a list, convert to dictionary (backward compatibility)
                answers_dict = {f"step_{i+1}": answer for i, answer in enumerate(all_answers) if answer}
                results_dict = {f"step_{i+1}": result for i, result in enumerate(all_results) if result}

            # Build information for each step (only include core information)
            for node_id, answer in answers_dict.items():
                if not answer or not answer.strip():
                    continue

                context_parts.append(f"[Step {node_id}]")

                # Include the specific question understood by LLM (if available)
                if node_id in results_dict:
                    result = results_dict[node_id]
                    if hasattr(result, 'understood_question') and result.understood_question:
                        context_parts.append(f"LLM understood question: {result.understood_question}")
                    elif hasattr(result, 'actual_question') and result.actual_question:
                        context_parts.append(f"Processing question: {result.actual_question}")
                    elif hasattr(result, 'original_question') and result.original_question:
                        context_parts.append(f"Original question: {result.original_question}")

                context_parts.append(f"Answer: {answer}")
                context_parts.append("")  # Empty line separator

            answers_context = "\n".join(context_parts)

            user_prompt = f"""Original query: {original_query}

Query execution results:
{answers_context}

Please strictly base on the above query results to generate an accurate and complete final answer for the original query."""

            # Add specific requirements for test mode
            if is_test_mode:
                user_prompt += """

**Test Format Requirements**:
1. Analyze the number of sub-questions contained in the original query
2. Provide concise answers for each sub-question in order
3. Output format must be: [answer1],[answer2],[answer3]...
4. Do not output any explanatory text, only output the formatted answer"""
            else:
                user_prompt += """

**Important Requirements**:
1. Only use specific information from the query results, do not add any other knowledge
2. **Do not output any additional explanatory information**, such as prefixes or suffixes like "According to the query results", "Based on the above data", "Key inference steps"
3. Directly answer the question, keep it concise and clear
4. If the information is insufficient to fully answer the question, clearly state it, but do not use explanatory language"""

            response = self.client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # lower temperature to reduce hallucination
                max_tokens=config.llm.max_tokens
            )

            final_answer = response.choices[0].message.content.strip()
            return final_answer if final_answer else "Sorry, unable to generate final answer."

        except Exception as e:
            output_manager.error(f"OutputGenerator failed to generate final answer: {e}")
            # Fallback: directly combine all answers
            if all_answers:
                if isinstance(all_answers, dict):
                    combined = "\n\n".join([f"**{node_id}**: {answer}" for node_id, answer in all_answers.items() if answer and answer.strip()])
                else:
                    combined = "\n\n".join([f"**Step {i+1}**: {answer}" for i, answer in enumerate(all_answers) if answer and answer.strip()])
                return combined if combined else "Sorry, unable to generate answer."
            else:
                return "Sorry, no available answer information."