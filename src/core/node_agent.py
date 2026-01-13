"""
NodeAgent - Minimal Refactoring
Responsible for complete execution flow and retry logic of individual DAG nodes.

Maintains consistency with existing DAGExecutor._execute_single_node behavior,
but encapsulates execution details as independent component for future extension.
"""
from typing import Dict, Any, List, Optional, Set
import time
from ..utils.output_manager import output_manager, log_node_execution, log_sql_generation, log_node_detailed_execution
from config.config import config


class NodeAgent:
    """Node Execution Agent (encapsulates node execution and retry logic)"""

    def __init__(
        self,
        query_router,
        mysql_retriever,
        neo4j_retriever,
        vector_retriever,
        fusion_engine,
        output_generator,
        database_type: str,
        max_retry: int = 2,
        channel_retry_max: int = 2,
    ):
        self.query_router = query_router
        self.mysql_retriever = mysql_retriever
        self.neo4j_retriever = neo4j_retriever
        # add attribute alias to support SQLite graph database
        self.sqlite_graph_retriever = neo4j_retriever
        self.vector_retriever = vector_retriever
        self.fusion_engine = fusion_engine
        self.output_generator = output_generator
        self.database_type = (database_type or "sqlite").lower()
        self.max_retry = max_retry or 2
        # retry count for each retrieval channel (local retry), default 2
        self.channel_retry_max = channel_retry_max or 2

    # ===== Public Interface =====
    def execute(self, node, memory) -> "UnifiedNodeResult":
        """Alias: equivalent to execute_node_with_retry, for unified interface alignment"""
        return self.execute_node_with_retry(node, memory)

    def execute_node_with_retry(self, node, memory, dag=None) -> "UnifiedNodeResult":
        """
        Execute single node (with retry mechanism)
        Integrated dynamic node type switching mechanism
        Support ablation experiment: disable dynamic switching
        """
        # ========== Ablation Experiment: Check if dynamic switching is enabled ==========
        switching_enabled = getattr(
            getattr(getattr(config, 'dag', None), 'dynamic_type_switching', None),
            'enabled',
            True  # default: enabled
        )

        if not switching_enabled:
            # ablation mode: no switching, strictly execute by predicted type
            output_manager.info("="*60)
            output_manager.info(f"[NodeAgent] 🚫 Dynamic switching disabled (ablation experiment)")
            output_manager.info("="*60)

            predicted_type = getattr(node, 'predicted_node_type', 'retrieval')
            node.actual_node_type = predicted_type  # no switching

            output_manager.info(f"Predicted type: {predicted_type}")
            output_manager.info(f"Actual type: {predicted_type} (forced)")

            # execute by predicted type, no intelligent judgment
            if predicted_type == "retrieval":
                # execute retrieval node (normal mode)
                if dag is None:
                    try:
                        dag = getattr(self.query_router, 'current_dag', None)
                    except Exception:
                        dag = None
                context = memory.build_context_for_node(node.parent_ids, dag=dag, to_node_id=node.id)
                return self._execute_single_node_strict(node, memory, dag, context, node_type="retrieval")
            else:  # inference
                # execute inference node (strict mode: no retrieval)
                if dag is None:
                    try:
                        dag = getattr(self.query_router, 'current_dag', None)
                    except Exception:
                        dag = None
                context = memory.build_context_for_node(node.parent_ids, dag=dag, to_node_id=node.id)
                return self._execute_inference_node_strict(node, memory, dag, context)

        # ========== Original dynamic switching logic (enabled state) ==========
        # Dynamic node type switching logic
        output_manager.info("="*60)
        output_manager.info(f"[NodeAgent] ✅ Dynamic switching enabled")
        output_manager.info(f"[NodeAgent] Node {node.id} type switching judgment")
        output_manager.info("="*60)

        # Step 1: Get predicted type
        predicted_type = getattr(node, 'predicted_node_type', 'retrieval')
        output_manager.info(f"Predicted type: {predicted_type}")

        # Step 2: Build parent node context
        if dag is None:
            try:
                dag = getattr(self.query_router, 'current_dag', None)
            except Exception:
                dag = None

        context = memory.build_context_for_node(node.parent_ids, dag=dag, to_node_id=node.id)
        parent_answers_context = "\n".join(context.get('parent_answers', ).values())

        # Step 3: Judge whether retrieval is needed
        decision = self.query_router.judge_retrieval_necessity(
            query=node.question,
            context=parent_answers_context,
            predicted_type=predicted_type  # pass predicted type
        )

        need_retrieval = decision.get("need_retrieval", True)
        judge_reason = decision.get("reason", "")
        confidence = decision.get("confidence", 0.0)
        information_gap = decision.get("information_gap", "")
        incremental_query = decision.get("incremental_query", "")
        target_channels = decision.get("target_channels", [])

        # Step 4: Determine actual type and whether switching occurred
        actual_type, switched = self._determine_actual_type(
            predicted_type, need_retrieval, node.id, judge_reason, confidence
        )

        # Step 5: Update node actual type
        node.actual_node_type = actual_type

        output_manager.info(f"Actual type: {actual_type}")
        if switched:
            output_manager.info(f"✓ Type switching occurred: {predicted_type} → {actual_type}")
            output_manager.info(f"Reason: {judge_reason}")
            if actual_type == "retrieval" and incremental_query:
                output_manager.info(f"🔍 Incremental retrieval mode")
                output_manager.info(f"  Information gap: {information_gap}")
                output_manager.info(f"  Incremental query: {incremental_query}")
                output_manager.info(f"  Target channels: {target_channels}")
        else:
            output_manager.info(f"✓ Keep predicted type: {predicted_type}")
        output_manager.info("="*60)

        # Step 6: Execute node based on actual type
        # Check configuration: whether incremental retrieval is enabled (experiment 4 ablation)
        incremental_enabled = True  # default enabled
        if hasattr(config.dag, 'incremental_retrieval') and hasattr(config.dag.incremental_retrieval, 'enabled'):
            incremental_enabled = config.dag.incremental_retrieval.enabled

        last_error = None
        last_error_hint = ""

        for attempt in range(1, self.max_retry + 1):
            try:
                output_manager.info(f"Attempt {attempt}/{self.max_retry}...")

                if actual_type == "retrieval":
                    # judge whether it's incremental retrieval
                    is_incremental = (
                        incremental_enabled  # add configuration switch check
                        and predicted_type == "inference"
                        and switched
                        and incremental_query
                    )

                    if is_incremental:
                        # execute incremental retrieval mode
                        result = self._execute_with_incremental_retrieval(
                            node, memory, context,
                            incremental_query=incremental_query,
                            information_gap=information_gap,
                            target_channels=target_channels,
                            dag=dag
                        )
                    else:
                        # execute normal retrieval mode (full retrieval)
                        result = self._execute_with_retrieval(node, memory, context, error_hint=last_error_hint, dag=dag)
                else:  # inference
                    # execute inference mode (no retrieval)
                    result = self._execute_without_retrieval(node, memory, context)

                # Step 7: Record switching information to result
                result.predicted_node_type = predicted_type
                result.actual_node_type = actual_type
                result.node_type_switched = switched
                result.switch_reason = judge_reason

                # Record node execution details (including type information)
                log_node_detailed_execution(
                    node_id=node.id,
                    original_question=result.original_question,
                    understood_question=result.understood_question,
                    answer_text=result.answer_text,
                    execution_time=result.execution_time,
                    channels=result.channels_used,
                    predicted_node_type=predicted_type,
                    actual_node_type=actual_type,
                    node_type_switched=switched,
                    switch_reason=judge_reason
                )

                return result

            except Exception as e:
                last_error = e
                output_manager.error(f"Attempt {attempt} failed: {e}")
                last_error_hint = str(e)

                if attempt < self.max_retry:
                    output_manager.info("Retrying... (adjusting generation based on error hint)")
                    time.sleep(0.5)

        raise RuntimeError(f"Node execution failed (after {self.max_retry} retries): {last_error}")

    def _call_llm_for_extraction(self, prompt: str) -> str:
        """Lightweight LLM call: for key information extraction, temperature=0, max tokens=150"""
        try:
            from config.config import config
            response = self.query_router.client.chat.completions.create(
                model=config.llm.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return ""

    def _determine_actual_type(
        self,
        predicted_type: str,
        need_retrieval: bool,
        node_id: str,
        reason: str,
        confidence: float
    ) -> tuple[str, bool]:
        """
        Determine actual node type and whether switching occurred

        Args:
            predicted_type: predicted node type ("retrieval" or "inference")
            need_retrieval: whether retrieval is needed
            node_id: node ID
            reason: judgment reason
            confidence: judgment confidence

        Returns:
            (actual_type, switched): actual type and whether switching occurred
        """
        # Case 1: predicted as inference, but actually needs retrieval
        if predicted_type == "inference" and need_retrieval:
            output_manager.info(
                f"[NodeAgent] 🔄 Node {node_id} switching: "
                f"inference → retrieval (confidence: {confidence:.2f})"
            )
            return "retrieval", True

        # Case 2: predicted as retrieval, but actually doesn't need retrieval
        elif predicted_type == "retrieval" and not need_retrieval:
            output_manager.info(
                f"[NodeAgent] 🔄 Node {node_id} switching: "
                f"retrieval → inference (confidence: {confidence:.2f})"
            )
            return "inference", True

        # Case 3: no switching needed
        else:
            output_manager.info(
                f"[NodeAgent] ✓ Node {node_id} keeping: {predicted_type}"
            )
            return predicted_type, False
    def execute_once(self, node, memory, error_hint: str = "", dag=None) -> "UnifiedNodeResult":
        """Alias: single execution without retry (for specific scheduling strategies or testing)"""
        # for compatibility, directly call execution with retrieval
        if dag is None:
            try:
                dag = getattr(self.query_router, 'current_dag', None)
            except Exception:
                dag = None
        context = memory.build_context_for_node(node.parent_ids, dag=dag, to_node_id=node.id)
        return self._execute_with_retrieval(node, memory, context, error_hint=error_hint, dag=dag)

    def _execute_with_retrieval(self, node, memory, context: Dict[str, Any], error_hint: str = "", dag=None) -> "UnifiedNodeResult":
        """Execute node (retrieval mode) - renamed from _execute_single_node"""
        node_start_time = time.time()

        # Step 1: Build query context (simplified output)
        output_manager.debug(f"Starting execution of node {node.id}: {node.question}")
        # use passed dag parameter, if not available try to get from query_router
        if dag is None:
            try:
                # QueryRouter generates DAG during decomposition phase; DAGExecutor holds the DAG during execution.
                # To avoid strong coupling, only try to get it here; pass None if unavailable.
                dag = getattr(self.query_router, 'current_dag', None)
            except Exception:
                dag = None
        context = memory.build_context_for_node(node.parent_ids, dag=dag, to_node_id=node.id)
        # ensure DAG object is available in context
        context['dag'] = dag
        # directly use parent node complete answers as context
        context_str = "\n".join(context.get('parent_answers', {}).values())

        # if there's error hint from previous round, include it in context to assist strict mode repair
        context_str_with_hint = context_str
        if error_hint:
            output_manager.debug(f"Previous execution error hint: {error_hint}")
            context_str_with_hint = (
                context_str
                + "\n\n[Previous execution failure hint] "
                + error_hint
                + "\nPlease generate valid SQL/Cypher and only reference entities that exist at runtime."
            )

        if context.get("has_parents"):
            output_manager.debug(f"Parent node count: {len(context['parent_ids'])}")
            for parent_id in context['parent_ids']:
                parent_answer = context['parent_answers'].get(parent_id, "")
                preview = parent_answer
                output_manager.debug(f"Parent node {parent_id}: {preview}")
        else:
            output_manager.debug("No parent nodes (root node)")

        # build parent node answer context (for subsequent processing)
        parent_answers_context = ""
        if context.get('parent_answers'):
            parent_answers = context['parent_answers']
            if parent_answers:
                context_parts = []
                for parent_id, answer in parent_answers.items():
                    if answer and answer.strip():
                        context_parts.append(f"Parent node {parent_id} answer: {answer}")
                parent_answers_context = "\n".join(context_parts)

        output_manager.debug(f"Original question: {node.question}")
        if parent_answers_context:
            output_manager.debug(f"Parent node answer context:\n{parent_answers_context}")
        else:
            output_manager.debug("No parent node answers")

        # Record actually processed question (after template replacement)
        # Note: template replacement is kept as fallback here, but mainly relies on LLM output
        fallback_question = self._replace_template_placeholders(node.question, context)
        output_manager.debug(f"Fallback processed question: {fallback_question}")

        # Comment: retrieval skipping judgment logic has been moved to DAGAdaptiveOptimizer for unified management
        # NodeAgent now only handles execution, no longer makes independent skip decisions

        # remove query routing analysis (no longer use QueryRouter's decision)
        # output_manager.info("Query routing analysis")
        # directly execute three channels in parallel, let each channel autonomously judge if suitable

        # define retrieval query (use fallback question, can be used if LLM understanding fails)
        question_for_retrieval = fallback_question

        # Step 2: 🚀 Execute three-channel retrieval in parallel
        output_manager.info("🚀 Executing retrieval channels in parallel")
        mysql_results, sqlite_graph_results, vector_results = self._execute_parallel_channels(
            question=question_for_retrieval,
            parent_context=parent_answers_context
        )

        # create a simple query_plan object for compatibility (no longer use QueryRouter decision)
        from src.core.models import QueryPlan, QueryType
        query_plan = QueryPlan(
            query_types=[QueryType.MYSQL, QueryType.NEO4J, QueryType.VECTOR],
            inference="Execute three channels in parallel, each channel judges autonomously",
            weights=config.retrieval_weights if hasattr(config, 'retrieval_weights') else {}
        )

        # Step 3: Fuse results
        output_manager.info("Fusing results")
        all_results = mysql_results + sqlite_graph_results + vector_results
        total_results = len(all_results)
        output_manager.debug(f"Total results: {total_results} (MySQL={len(mysql_results)}, SQLite graph={len(sqlite_graph_results)}, Vector={len(vector_results)})")

        # Graceful degradation: if no retrieval results, let LLM generate answer directly based on parent node context
        if total_results == 0:
            output_manager.warning("⚠️ All channels returned no results, degrading to inference mode")
            output_manager.debug(f"Query keywords: '{fallback_question}', database type: {self.database_type.upper()}")
            output_manager.info("🔄 Attempting to generate answer based on parent node context...")

            # build LLM prompt, explicitly inform no retrieval results
            no_result_prompt = f"""You are an intelligent QA system.

**Current Question**: {fallback_question}

**Parent Context** (if available):
{parent_answers_context if parent_answers_context else "No parent context available."}

**Database Retrieval Status**: No relevant data found in the database.

**Task**: Based on the parent context, answer the question. If the parent context is insufficient, explicitly state that no information was found in the database.

**Output Format**: Provide a direct, concise answer. If no information is available, say "No relevant information found in the database."

**Answer**:"""

            try:
                response = self.query_router.client.chat.completions.create(
                    model=config.llm.model,
                    messages=[{"role": "user", "content": no_result_prompt}],
                    temperature=0.0,
                    max_tokens=config.llm.max_tokens
                )
                answer_content = response.choices[0].message.content.strip()
                output_manager.info(f"✓ LLM generated answer based on context: {answer_content[:100]}...")

                # build result (mark as inference mode)
                execution_time = time.time() - node_start_time
                from src.core.dag_models import UnifiedNodeResult
                return UnifiedNodeResult(
                    node_id=node.id,
                    original_question=node.question,
                    understood_question=fallback_question,
                    answer_text=answer_content,
                    execution_time=execution_time,
                    channels_used=[],  # no retrieval channels
                    retrieval_contexts=[],  # no retrieval contexts
                    predicted_node_type=node.predicted_node_type,
                    actual_node_type="inference",  # actually degraded to inference
                    node_type_switched=True,  # mark as type switching
                    switch_reason="No retrieval results, fallback to inference with parent context",
                    channel_results={},
                    confidence=0.5  # lower confidence (no retrieval support)
                )

            except Exception as e:
                output_manager.error(f"❌ LLM degradation failed: {e}")
                # if LLM also fails, return explicit error message
                execution_time = time.time() - node_start_time
                from src.core.dag_models import UnifiedNodeResult
                return UnifiedNodeResult(
                    node_id=node.id,
                    original_question=node.question,
                    understood_question=fallback_question,
                    answer_text="No information available (retrieval failed and LLM fallback also failed)",
                    execution_time=execution_time,
                    channels_used=[],
                    retrieval_contexts=[],
                    predicted_node_type=node.predicted_node_type,
                    actual_node_type="inference",
                    node_type_switched=True,
                    switch_reason=f"Fallback failed: {str(e)}",
                    channel_results={},
                    confidence=0.0
                )

        fused_results = self.fusion_engine.fuse_results(
            query=fallback_question,
            mysql_results=mysql_results,
            sqlite_graph_results=sqlite_graph_results,
            vector_results=vector_results,
            weights=(getattr(query_plan, 'weights', None) or {}),
            parent_context=parent_answers_context  # pass parent node context
        )

        output_manager.debug(f"Fused result count: {len(fused_results)}")

        # Step 4: Directly use FusionEngine's structured output (no additional LLM call needed)
        if not fused_results:
            raise RuntimeError("No results obtained after fusion")

        # FusionEngine already provides understood_question and content
        fused_result = fused_results[0]  # take first fused result
        understood_question = fused_result.understood_question
        answer_content = fused_result.content

        # FIX: Ensure answer_content is string type (fix DAG template replacement error)
        if not isinstance(answer_content, str):
            if isinstance(answer_content, (dict, list)):
                answer_content = str(answer_content)
            else:
                answer_content = str(answer_content) if answer_content is not None else ""

        output_manager.info(f"Using FusionEngine's structured output")
        output_manager.debug(f"LLM understood question: {understood_question}")
        output_manager.debug(f"Answer length: {len(answer_content)} characters")
        output_manager.debug(f"Complete answer: {answer_content}")

        # Step 5: Build UnifiedNodeResult (using FusionEngine's structured output)
        execution_time = time.time() - node_start_time
        channels_used = []
        if mysql_results: channels_used.append("mysql")
        if sqlite_graph_results: channels_used.append("sqlite_graph")
        if vector_results: channels_used.append("vector")

        from src.core.dag_models import UnifiedNodeResult

        # FIX: Ensure storing original retrieval results, not just fused answer
        # collect all channels' original retrieval results for building retrieved_contexts
        original_retrieval_results = []

        # add MySQL results
        for mysql_result in mysql_results:
            if hasattr(mysql_result, 'content') and mysql_result.content:
                original_retrieval_results.append(mysql_result.content)

        # add graph database results
        for neo4j_result in sqlite_graph_results:
            if hasattr(neo4j_result, 'content') and neo4j_result.content:
                original_retrieval_results.append(neo4j_result.content)

        # add vector retrieval results
        for vector_result in vector_results:
            if hasattr(vector_result, 'content') and vector_result.content:
                original_retrieval_results.append(vector_result.content)

        result = UnifiedNodeResult(
            node_id=node.id,
            original_question=node.question,  # original question (with template)
            understood_question=understood_question,  # FusionEngine understood question
            answer_text=answer_content,  # FusionEngine provided answer
            execution_time=execution_time,
            channels_used=channels_used,
            # keep fields for statistics
            channel_results={
                "mysql": len(mysql_results),
                "sqlite_graph": len(sqlite_graph_results),
                "vector": len(vector_results)
            },
            confidence=fused_results[0].confidence if fused_results else 0.0,
            query_plan=query_plan.to_dict(),
            # FIX: store original retrieval result contexts
            retrieval_contexts=original_retrieval_results
        )

        output_manager.success(f"Node execution completed, time: {execution_time:.3f}s")

        return result

    # ====== Following are auxiliary methods required for execution (simplified) ======
    # old constraint output extraction methods deleted, unified to use key information summary

    def _replace_template_placeholders(self, question: str, context: Dict[str, Any]) -> str:
        """Replace template placeholders in question with actual values"""
        processed_question = question

        # replace parent node answer placeholders
        if context.get("parent_answers"):
            for parent_id, answer in context["parent_answers"].items():
                if answer and answer.strip():
                    # replace {{parent_id.answer}} format
                    placeholder = f"{{{{{parent_id}.answer}}}}"
                    processed_question = processed_question.replace(placeholder, answer)
                    # replace {{pX.answer}} format (simplified form)
                    simple_placeholder = f"{{{{p{parent_id[-1]}.answer}}}}"
                    processed_question = processed_question.replace(simple_placeholder, answer)

        return processed_question

    def _parse_candidate_ids_from_text(self, text: str) -> List[int]:
        """Parse possible company ID list from text"""
        import re
        if not text:
            return []

        candidate_ids: List[int] = []

        lines = [l.strip() for l in text.splitlines()]
        for line in lines:
            m = re.search(r"\bid\s*:\s*(\d+)\b", line, re.IGNORECASE)
            if m:
                candidate_ids.append(int(m.group(1)))
                continue

            m2 = re.match(r"^-\s*(\d+)\s*$", line)
            if m2:
                candidate_ids.append(int(m2.group(1)))
                continue

            m3 = re.match(r"^(\d+)\s*$", line)
            if m3:
                candidate_ids.append(int(m3.group(1)))
                continue

            for n in re.findall(r"\b(\d{1,8})\b", line):
                lowered = line.lower()
                if ("id" in lowered) or lowered.strip().startswith("-"):
                    candidate_ids.append(int(n))

        seen = set()
        ordered_unique: List[int] = []
        for cid in candidate_ids:
            if cid not in seen:
                seen.add(cid)
                ordered_unique.append(cid)

        return ordered_unique

    def _resolve_placeholders_in_question(self, question: str, context: Dict[str, Any]) -> (str, List[int]):
        """Parse placeholders in question
        Supports:
        - {{pX.answer}}: inject parent node answer (if ID can be parsed, prioritize injecting ID list)
        - {{pX.id}}: parse ID list from parent node structured data or answer
        - {{pX.out.key}}: inject specified key from parent node structured output (list will be joined with ", " and prefixed with field name)

        Key fix: preserve field name prefix when injecting values, e.g. "company_id: 4" instead of "4"

        Returns: (replaced question text, candidate ID list collected during parsing)
        """
        import re

        resolved = question
        all_ids: List[int] = []

        # unified matching of three types of placeholders: answer/id/out.key
        pattern = r"\{\{\s*(p\d+)\.(answer|id|out)(?:\.(\w+))?\s*\}\}"
        matches = re.findall(pattern, question)
        if not matches:
            return resolved, all_ids

        parent_answers = context.get("parent_answers", {})
        parent_structs = context.get("parent_structured_data", {})
        parent_out = context.get("parent_out", {})

        for pid, kind, subkey in matches:
            replacement = ""
            # 1) out.key: prioritize using structured output, preserve field name prefix
            if kind == "out":
                data = parent_out.get(pid) or parent_structs.get(pid) or {}
                val = None
                if isinstance(data, dict):
                    val = data.get(subkey)
                if isinstance(val, list):
                    # collect possible IDs
                    for v in val:
                        try:
                            iv = int(v)
                            all_ids.append(iv)
                        except Exception:
                            pass
                    # key fix: prefix with field name, format as "field_name: value1, value2"
                    # FIX: ensure each element is string, avoid join error
                    safe_values = []
                    for x in val:
                        if isinstance(x, str):
                            safe_values.append(x)
                        else:
                            safe_values.append(str(x) if x is not None else "")
                    value_str = ", ".join(safe_values)
                    replacement = f"{subkey}: {value_str}"
                elif val is not None:
                    try:
                        iv = int(val)
                        all_ids.append(iv)
                    except Exception:
                        pass
                    # key fix: prefix with field name
                    replacement = f"{subkey}: {val}"
                else:
                    replacement = ""
                resolved = re.sub(rf"\{{\{{\s*{pid}\.out\.{subkey}\s*\}}\}}", replacement, resolved)
                continue

            # 2) id: parse ID from structured or text, prefix with field name "id"
            if kind == "id":
                struct = parent_structs.get(pid, {}) if isinstance(parent_structs.get(pid, {}), dict) else {}
                ids_from_struct: List[int] = []
                val = struct.get("ids")
                if isinstance(val, list):
                    try:
                        ids_from_struct = [int(x) for x in val]
                    except Exception:
                        ids_from_struct = []
                ids_from_text = self._parse_candidate_ids_from_text(parent_answers.get(pid, "")) if not ids_from_struct else []
                candidate_ids = ids_from_struct or ids_from_text
                if candidate_ids:
                    for cid in candidate_ids:
                        if cid not in all_ids:
                            all_ids.append(cid)
                    # key fix: prefix with field name "id:"
                    replacement = "id: " + ", ".join(str(x) for x in candidate_ids)
                else:
                    replacement = ""
                resolved = re.sub(rf"\{{\{{\s*{pid}\.id\s*\}}\}}", replacement, resolved)
                continue

            # 3) answer: directly use complete answer
            if kind == "answer":
                # directly use complete answer, ensure it's string type
                raw_answer = parent_answers.get(pid, "")
                if isinstance(raw_answer, str):
                    replacement = raw_answer
                else:
                    # if not string, convert to string
                    replacement = str(raw_answer) if raw_answer is not None else ""
                resolved = re.sub(rf"\{{\{{\s*{pid}\.answer\s*\}}\}}", replacement, resolved)

        # deduplicate IDs
        if all_ids:
            seen = set()
            uniq = []
            for cid in all_ids:
                if cid not in seen:
                    seen.add(cid)
                    uniq.append(cid)
            all_ids = uniq

        return resolved, all_ids

    def _execute_without_retrieval(self, node, memory, context: Dict[str, Any]) -> "UnifiedNodeResult":
        """Execute node (inference mode, no retrieval) - renamed from _generate_answer_without_retrieval"""
        node_start_time = time.time()

        output_manager.info("Inference mode execution (skip retrieval)")
        output_manager.debug(f"Node: {node.id}, question: {node.question}")

        # build parent node information context
        parent_answers_context = ""
        if context.get('parent_answers'):
            parent_answers = context['parent_answers']
            if parent_answers:
                context_parts = []
                for parent_id, answer in parent_answers.items():
                    if answer and answer.strip():
                        context_parts.append(f"Parent node {parent_id} answer: {answer}")
                parent_answers_context = "\n".join(context_parts)

        # KEY IMPROVEMENT: create mock fused result containing parent node information, let main LLM handle uniformly
        from src.core.models import FusedResult, ConflictInfo
        mock_fused_result = FusedResult(
            content=parent_answers_context,
            sources=["parent_nodes"],
            confidence=0.9,
            metadata_summary={"source": "parent node information", "type": "direct_context"}
        )

        # create query plan
        from src.core.models import QueryPlan, QueryType
        mock_query_plan = QueryPlan(
            query_types=[],  # skip retrieval, no specific query types
            inference="Generate answer directly based on parent node information",
            weights={}
        )

        # GOAL: mock FusionEngine to generate understood_question
        # generate question understanding for skip retrieval case
        fallback_question = self._replace_template_placeholders(node.question, context)

        try:
            # create mock FusedResult, containing LLM-based question understanding and answer
            from src.core.models import FusedResult

            # use LLM to generate question understanding and answer for skip retrieval case
            skip_retrieval_response = self.output_generator._generate_answer(
                query=fallback_question,
                fused_results=[],  # empty results, indicating skip retrieval
                query_plan=mock_query_plan,
                parent_context=parent_answers_context
            )

            # parse LLM returned structured response
            if isinstance(skip_retrieval_response, dict):
                understood_question = skip_retrieval_response.get('understood_question', fallback_question)
                answer_content = skip_retrieval_response.get('answer', "Failed to generate answer")
            else:
                understood_question = fallback_question
                answer_content = str(skip_retrieval_response)

            # FIX: ensure answer_content is string type (fix DAG template replacement error)
            if not isinstance(answer_content, str):
                answer_content = str(answer_content) if answer_content is not None else "Failed to generate answer"

            output_manager.debug(f"Inference mode - LLM understood question: {understood_question}")
            output_manager.debug(f"Inference mode - generated answer length: {len(answer_content)} characters")
            output_manager.debug(f"Inference mode - generated answer: {answer_content}")

            # build UnifiedNodeResult
            execution_time = time.time() - node_start_time

            from src.core.dag_models import UnifiedNodeResult
            result = UnifiedNodeResult(
                node_id=node.id,
                original_question=node.question,
                understood_question=understood_question,
                answer_text=answer_content,
                execution_time=execution_time,
                channels_used=["inference"],  # mark as inference mode
                channel_results={
                    "mysql": 0,
                    "neo4j": 0,
                    "vector": 0,
                    "inference": 1  # inference mode marker
                },
                confidence=0.9,
                query_plan={"inference_mode": True},  # mark as inference mode
                retrieval_contexts=[]  # inference node has no retrieval contexts
            )

            output_manager.success(f"Inference mode execution completed, time: {execution_time:.3f}s")
            return result

        except Exception as e:
            output_manager.error(f"Inference mode execution failed: {e}")
            raise RuntimeError(f"Inference mode execution failed: {e}, need to fallback to retrieval mode")

    def _execute_with_incremental_retrieval(
        self,
        node,
        memory,
        context: Dict[str, Any],
        incremental_query: str,
        information_gap: str,
        target_channels: List[str],
        dag=None
    ) -> "UnifiedNodeResult":
        """
        Execute incremental retrieval: only retrieve missing information, then merge with parent node information

        Args:
            node: node object
            memory: Memory object
            context: node context
            incremental_query: incremental retrieval query (precise query)
            information_gap: information gap description
            target_channels: target retrieval channel list
            dag: DAG object

        Returns:
            UnifiedNodeResult
        """
        node_start_time = time.time()

        output_manager.info("=" * 60)
        output_manager.info("🔍 Incremental retrieval mode")
        output_manager.info("=" * 60)
        output_manager.info(f"Node: {node.id}")
        output_manager.info(f"Original question: {node.question}")
        output_manager.info(f"Information gap: {information_gap}")
        output_manager.info(f"Incremental query: {incremental_query}")
        output_manager.info(f"Target channels: {target_channels}")

        # Step 1: Get parent node information
        parent_answers_context = ""
        if context.get('parent_answers'):
            parent_answers = context['parent_answers']
            if parent_answers:
                context_parts = []
                for parent_id, answer in parent_answers.items():
                    if answer and answer.strip():
                        context_parts.append(f"Parent node {parent_id} answer: {answer}")
                parent_answers_context = "\n".join(context_parts)

        output_manager.info(f"Parent node information length: {len(parent_answers_context)} characters")

        # Step 2: Execute incremental retrieval (using precise incremental query and specified channels)
        try:
            output_manager.info("🚀 Starting incremental retrieval")
            mysql_results, graph_results, vector_results = self._execute_parallel_channels(
                question=incremental_query,  # KEY: use incremental query, not original question
                parent_context="",  # incremental retrieval doesn't need parent node context
                active_channels=target_channels  # KEY: only use specified channels
            )

            # check if incremental retrieval succeeded
            total_incremental_results = len(mysql_results) + len(graph_results) + len(vector_results)

            if total_incremental_results == 0:
                # incremental retrieval failed (no results), degrade to inference mode
                output_manager.warning("⚠️ Incremental retrieval failed: no retrieval results")
                output_manager.warning("⤵️ Degrading to inference mode, generate answer based on parent node information only")

                return self._fallback_to_inference_after_incremental_failure(
                    node, memory, context, parent_answers_context,
                    information_gap, incremental_query, node_start_time
                )

            output_manager.success(f"✅ Incremental retrieval succeeded: {total_incremental_results} results")

        except Exception as e:
            # incremental retrieval failed (exception), degrade to inference mode
            output_manager.error(f"❌ Incremental retrieval failed: {e}")
            output_manager.warning("⤵️ Degrading to inference mode, generate answer based on parent node information only")

            return self._fallback_to_inference_after_incremental_failure(
                node, memory, context, parent_answers_context,
                information_gap, incremental_query, node_start_time
            )

        # Step 3: Fuse incremental retrieval results
        output_manager.info("Fusing incremental retrieval results")
        from src.core.models import QueryPlan, QueryType

        # create query plan (mark as incremental retrieval)
        incremental_query_plan = QueryPlan(
            query_types=[],  # incremental retrieval doesn't need full types
            inference=f"Incremental retrieval: {information_gap}",
            weights=config.retrieval_weights if hasattr(config, 'retrieval_weights') else {}
        )

        fused_results = self.fusion_engine.fuse_results(
            query=incremental_query,
            mysql_results=mysql_results,
            sqlite_graph_results=graph_results,
            vector_results=vector_results,
            weights=(getattr(incremental_query_plan, 'weights', None) or {}),
            parent_context=""  # don't pass parent node context during incremental retrieval
        )

        if not fused_results:
            output_manager.warning("⚠️ Incremental retrieval fusion result is empty")
            return self._fallback_to_inference_after_incremental_failure(
                node, memory, context, parent_answers_context,
                information_gap, incremental_query, node_start_time
            )

        # Step 4: Extract incremental retrieval content
        incremental_content = fused_results[0].content
        understood_question = fused_results[0].understood_question

        output_manager.info(f"Incremental retrieval content length: {len(incremental_content)} characters")

        # Step 5: Merge parent node information + incremental retrieval results
        output_manager.info("Merging parent node information with incremental retrieval results")
        full_context = f"""【Parent node existing information】
{parent_answers_context}

【Incrementally retrieved information ({information_gap})】
{incremental_content}"""

        output_manager.debug(f"Complete context length: {len(full_context)} characters")

        # Step 6: Generate final answer based on complete context
        output_manager.info("Generating final answer based on complete context")

        try:
            # call LLM to generate answer
            answer_generation_response = self.output_generator._generate_answer(
                query=node.question,  # use original question
                fused_results=fused_results,  # pass fused results
                query_plan=incremental_query_plan,
                parent_context=parent_answers_context  # pass parent node context
            )

            # parse answer
            if isinstance(answer_generation_response, dict):
                answer_content = answer_generation_response.get('answer', incremental_content)
            else:
                answer_content = str(answer_generation_response) if answer_generation_response else incremental_content

            # ensure answer_content is string type
            if not isinstance(answer_content, str):
                answer_content = str(answer_content) if answer_content is not None else incremental_content

            output_manager.debug(f"Final answer length: {len(answer_content)} characters")

        except Exception as e:
            output_manager.error(f"Answer generation failed: {e}, using incremental retrieval content")
            answer_content = incremental_content
            understood_question = node.question

        # Step 7: Build UnifiedNodeResult (incremental retrieval mode)
        execution_time = time.time() - node_start_time

        # build retrieval channel list
        channels_used = []
        if mysql_results:
            channels_used.append("sql")
        if graph_results:
            channels_used.append("graph")
        if vector_results:
            channels_used.append("vector")

        # collect original incremental retrieval results (excluding parent node information)
        original_incremental_contexts = []
        for result in mysql_results:
            if hasattr(result, 'content') and result.content:
                original_incremental_contexts.append(result.content)
        for result in graph_results:
            if hasattr(result, 'content') and result.content:
                original_incremental_contexts.append(result.content)
        for result in vector_results:
            if hasattr(result, 'content') and result.content:
                original_incremental_contexts.append(result.content)

        from src.core.dag_models import UnifiedNodeResult
        result = UnifiedNodeResult(
            node_id=node.id,
            original_question=node.question,
            understood_question=understood_question,
            answer_text=answer_content,
            execution_time=execution_time,
            channels_used=channels_used,
            # incremental retrieval fields
            is_incremental=True,
            information_gap=information_gap,
            incremental_query=incremental_query,
            incremental_retrieval_failed=False,
            # channel statistics
            channel_results={
                "sql": len(mysql_results),
                "graph": len(graph_results),
                "vector": len(vector_results)
            },
            confidence=fused_results[0].confidence if fused_results else 0.0,
            query_plan=incremental_query_plan.to_dict(),
            # KEY: only store incremental retrieval contexts (excluding parent nodes)
            retrieval_contexts=original_incremental_contexts
        )

        output_manager.success(f"✅ Incremental retrieval mode execution completed, time: {execution_time:.3f}s")
        output_manager.info("=" * 60)

        return result

    def _fallback_to_inference_after_incremental_failure(
        self,
        node,
        memory,
        context: Dict[str, Any],
        parent_answers_context: str,
        information_gap: str,
        incremental_query: str,
        node_start_time: float
    ) -> "UnifiedNodeResult":
        """
        Degradation handling after incremental retrieval failure: degrade to inference mode

        Args:
            node: node object
            memory: Memory object
            context: node context
            parent_answers_context: parent node answer context
            information_gap: information gap
            incremental_query: incremental query
            node_start_time: node start time

        Returns:
            UnifiedNodeResult (marked as incremental retrieval failed)
        """
        output_manager.info("Executing degraded inference mode")

        try:
            # use LLM to generate answer based on parent node information
            from src.core.models import QueryPlan

            fallback_query_plan = QueryPlan(
                query_types=[],
                inference="Incremental retrieval failed, direct inference based on parent node information",
                weights={}
            )

            answer_generation_response = self.output_generator._generate_answer(
                query=node.question,
                fused_results=[],  # no retrieval results
                query_plan=fallback_query_plan,
                parent_context=parent_answers_context
            )

            # parse answer
            if isinstance(answer_generation_response, dict):
                understood_question = answer_generation_response.get('understood_question', node.question)
                answer_content = answer_generation_response.get('answer', "Unable to answer due to incremental retrieval failure and insufficient parent node information")
            else:
                understood_question = node.question
                answer_content = str(answer_generation_response) if answer_generation_response else "Unable to answer due to incremental retrieval failure and insufficient parent node information"

            # ensure answer_content is string type
            if not isinstance(answer_content, str):
                answer_content = str(answer_content) if answer_content is not None else "Unable to generate answer"

        except Exception as e:
            output_manager.error(f"Degraded inference mode also failed: {e}")
            understood_question = node.question
            answer_content = f"Incremental retrieval failed, and unable to answer based on parent node information: {str(e)}"

        # build result
        execution_time = time.time() - node_start_time

        from src.core.dag_models import UnifiedNodeResult
        result = UnifiedNodeResult(
            node_id=node.id,
            original_question=node.question,
            understood_question=understood_question,
            answer_text=answer_content,
            execution_time=execution_time,
            channels_used=["inference"],  # mark as inference mode
            # incremental retrieval failure marker
            is_incremental=True,
            information_gap=information_gap,
            incremental_query=incremental_query,
            incremental_retrieval_failed=True,  # KEY: mark as failed
            channel_results={
                "inference": 1
            },
            confidence=0.3,  # low confidence
            query_plan={"fallback": True},
            retrieval_contexts=[]  # no retrieval contexts
        )

        output_manager.warning(f"⚠️ Degraded inference mode completed, time: {execution_time:.3f}s")
        return result

    def _basic_sql_is_valid(self, sql: str) -> bool:
        """Lightweight SQL validity check"""
        import re
        if not sql:
            return False
        s = sql.strip()
        if s.startswith("SQL:") or s.startswith("sql:"):
            s = s.split(":", 1)[1].strip()
        s = s.replace("```sql", "").replace("```", "").strip()
        return bool(re.match(r"^(SELECT|WITH|INSERT|UPDATE|DELETE)\b", s, flags=re.IGNORECASE))

    def _detect_missing_tables(self, sql: str, tables: Set[str]) -> Set[str]:
        """Detect tables referenced in SQL that don't exist"""
        import re
        if not sql or not tables:
            return set()
        found = set(re.findall(r"\bFROM\s+([a-zA-Z_][\w]*)\b", sql, flags=re.IGNORECASE))
        found |= set(re.findall(r"\bJOIN\s+([a-zA-Z_][\w]*)\b", sql, flags=re.IGNORECASE))
        missing = {t for t in found if t not in tables}
        return missing

    def _detect_invalid_cypher_entities(self, cypher: str, node_labels: set[str], rel_types: set[str]) -> set[str]:
        """Detect labels or relationship types referenced in Cypher that don't exist (heuristic)"""
        import re
        if not cypher:
            return set()
        invalid = set()
        # label matching: (n:Label)
        labels = set(re.findall(r":([A-Za-z][\w]*)", cypher))
        for lab in labels:
            if node_labels and (lab not in node_labels):
                invalid.add(f"label:{lab}")
        # relationship type matching -[:TYPE]-
        rels = set(re.findall(r"-\s*\[\s*:\s*([A-Za-z][\w]*)\s*\]\s*-", cypher))
        for rt in rels:
            if rel_types and (rt not in rel_types):
                invalid.add(f"rel:{rt}")
        return invalid

    def _execute_parallel_channels(
        self,
        question: str,
        parent_context: str = "",
        active_channels: List[str] = None  # specify active channels
    ) -> tuple:
        """
        🚀 Execute three-channel retrieval in parallel (SQL/Graph/Vector)
        Support ablation experiment: limit channels (vector_only/sql_only/graph_only/full)

        Args:
            question: query question
            parent_context: parent node answer context
            active_channels: active channel list (e.g. ["sql", "graph"], None means all)

        Returns:
            (mysql_results, sqlite_graph_results, vector_results)
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # ========== Ablation Experiment: Check fusion mode ==========
        fusion_mode = getattr(
            getattr(getattr(config, 'dag', None), 'fusion', None),
            'mode',
            'full'  # default: full fusion
        )

        # override active_channels based on fusion mode
        if fusion_mode == "vector_only":
            active_channels = ["vector"]
            output_manager.info("[Ablation Experiment] Fusion mode: VECTOR ONLY")
        elif fusion_mode == "sql_only":
            active_channels = ["sql"]
            output_manager.info("[Ablation Experiment] Fusion mode: SQL ONLY")
        elif fusion_mode == "graph_only":
            active_channels = ["graph"]
            output_manager.info("[Ablation Experiment] Fusion mode: GRAPH ONLY")
        else:  # full
            if active_channels is None:
                active_channels = ["sql", "graph", "vector"]
            output_manager.info("[Ablation Experiment] Fusion mode: FULL (all three channels)")

        output_manager.info("🚀 Starting parallel retrieval execution")
        output_manager.info(f"  Active channels: {active_channels}")

        # detect database type, handle thread safety
        is_sqlite = self.database_type.lower() == 'sqlite'
        if is_sqlite:
            output_manager.debug("SQLite mode: using thread-safe retriever")

        # ===== Define execution functions for three channels =====

        def execute_mysql_channel():
            """MySQL/SQLite channel execution"""
            channel_start = time.time()
            try:
                thread_name = threading.current_thread().name
                output_manager.info(f"📋 SQL channel started [thread: {thread_name}] [time: {channel_start:.3f}]")

                # create thread-safe retriever instance (if needed)
                if is_sqlite:
                    from src.retrievers.sqlite_retriever import MySQLRetrieverAdapter
                    retriever = MySQLRetrieverAdapter()
                else:
                    retriever = self.mysql_retriever

                # ⚡ Integrated intelligent judgment + query generation
                llm_start = time.time()
                output_manager.debug(f"📋 SQL channel LLM started [thread: {thread_name}] [time: {llm_start:.3f}]")

                sql_or_placeholder = retriever.generate_sql_or_placeholder(question, parent_context)

                llm_end = time.time()
                output_manager.debug(f"📋 SQL channel LLM completed [thread: {thread_name}] [time: {llm_end:.3f}] [duration: {llm_end-llm_start:.3f}s]")

                # ✅ Intelligent skip logic: if LLM returns placeholder, skip this channel
                if sql_or_placeholder.startswith("SQL_PLACEHOLDER"):
                    output_manager.info(f"⏭️ SQL channel intelligently skipped: LLM judged question irrelevant to SQL retrieval")
                    results = []
                    channel_end = time.time()
                    output_manager.info(f"📋 SQL channel total time: {channel_end-channel_start:.3f}s [thread: {thread_name}] [status: skipped]")
                    return results
                else:
                    # execute SQL query (with retry, limit result count)
                    db_start = time.time()
                    output_manager.debug(f"📋 SQL channel DB query started [thread: {thread_name}] [time: {db_start:.3f}]")

                    results = retriever.execute_sql_with_retry(sql_or_placeholder, parent_context, result_limit=5)

                    db_end = time.time()
                    output_manager.debug(f"📋 SQL channel DB query completed [thread: {thread_name}] [time: {db_end:.3f}] [duration: {db_end-db_start:.3f}s]")

                if len(results) == 0:
                    output_manager.debug(f"⚪ SQL channel query: 0 results")
                else:
                    output_manager.debug(f"✅ SQL channel completed: {len(results)} results")

                channel_end = time.time()
                output_manager.info(f"📋 SQL channel total time: {channel_end-channel_start:.3f}s [thread: {thread_name}]")
                return results

            except Exception as e:
                output_manager.error(f"❌ SQL channel failed: {e}")
                return []
            finally:
                # ensure closing thread-local SQLite connection
                if is_sqlite and 'retriever' in locals():
                    try:
                        if hasattr(retriever, 'close'):
                            retriever.close()
                    except Exception as close_error:
                        output_manager.warning(f"⚠ Error closing SQL thread connection: {close_error}")

        def execute_neo4j_channel():
            """Graph database channel execution"""
            channel_start = time.time()
            try:
                thread_name = threading.current_thread().name
                output_manager.info(f"🕸️ Graph channel started [thread: {thread_name}] [time: {channel_start:.3f}]")

                # ⚡ Integrated intelligent judgment + query generation
                llm_start = time.time()
                output_manager.debug(f"🕸️ Graph channel LLM started [thread: {thread_name}] [time: {llm_start:.3f}]")

                query_or_placeholder = self.sqlite_graph_retriever.generate_query_or_placeholder(
                    question, parent_context
                )

                llm_end = time.time()
                output_manager.debug(f"🕸️ Graph channel LLM completed [thread: {thread_name}] [time: {llm_end:.3f}] [duration: {llm_end-llm_start:.3f}s]")

                # ✅ Intelligent skip logic: if LLM returns placeholder, skip this channel
                if query_or_placeholder.startswith("GRAPH_PLACEHOLDER"):
                    output_manager.info(f"⏭️ Graph channel intelligently skipped: LLM judged question irrelevant to graph retrieval")
                    results = []
                    channel_end = time.time()
                    output_manager.info(f"🕸️ Graph channel total time: {channel_end-channel_start:.3f}s [thread: {thread_name}] [status: skipped]")
                    return results
                else:
                    # execute graph query (with retry, limit result count)
                    db_start = time.time()
                    output_manager.debug(f"🕸️ Graph channel DB query started [thread: {thread_name}] [time: {db_start:.3f}]")

                    results = self.sqlite_graph_retriever.execute_query_with_retry(query_or_placeholder, parent_context, result_limit=5)

                    db_end = time.time()
                    output_manager.debug(f"🕸️ Graph channel DB query completed [thread: {thread_name}] [time: {db_end:.3f}] [duration: {db_end-db_start:.3f}s]")

                if len(results) == 0:
                    output_manager.debug(f"⚪ Graph channel query: 0 results")
                else:
                    output_manager.debug(f"✅ Graph channel completed: {len(results)} results")

                channel_end = time.time()
                output_manager.info(f"🕸️ Graph channel total time: {channel_end-channel_start:.3f}s [thread: {thread_name}]")
                return results

            except Exception as e:
                output_manager.error(f"❌ Graph channel failed: {e}")
                return []

        def execute_vector_channel():
            """Vector retrieval channel execution"""
            channel_start = time.time()
            try:
                thread_name = threading.current_thread().name
                output_manager.info(f"🔍 Vector channel started [thread: {thread_name}] [time: {channel_start:.3f}]")

                # vector retrieval (using existing query rewriting logic, limit result count)
                vector_start = time.time()
                output_manager.debug(f"🔍 Vector channel retrieval started [thread: {thread_name}] [time: {vector_start:.3f}]")

                results = self.vector_retriever.search(
                    query=question,
                    parent_context=parent_context,
                    limit=3  # limit result count
                )

                vector_end = time.time()
                output_manager.debug(f"🔍 Vector channel retrieval completed [thread: {thread_name}] [time: {vector_end:.3f}] [duration: {vector_end-vector_start:.3f}s]")

                if len(results) == 0:
                    output_manager.debug(f"⚪ VECTOR channel query: 0 results")
                else:
                    output_manager.debug(f"✅ VECTOR channel completed: {len(results)} results")

                channel_end = time.time()
                output_manager.info(f"🔍 Vector channel total time: {channel_end-channel_start:.3f}s [thread: {thread_name}]")
                return results

            except Exception as e:
                output_manager.error(f"❌ Vector channel failed: {e}")
                return []

        # ===== Parallel execution =====
        parallel_start = time.time()
        output_manager.info(f"🚀 Parallel execution started [time: {parallel_start:.3f}]")

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Only submit tasks for active channels
            submit_start = time.time()
            futures = {}

            if "sql" in active_channels:
                futures[executor.submit(execute_mysql_channel)] = 'sql'
            if "graph" in active_channels:
                futures[executor.submit(execute_neo4j_channel)] = 'graph'
            if "vector" in active_channels:
                futures[executor.submit(execute_vector_channel)] = 'vector'

            submit_end = time.time()
            output_manager.debug(f"🚀 Task submission completed [time: {submit_end:.3f}] [duration: {submit_end-submit_start:.3f}s]")

            # Collect results (lenient strategy)
            mysql_results = []
            sqlite_graph_results = []
            vector_results = []

            successful_channels = 0
            skipped_channels = 0

            collect_start = time.time()
            for future in as_completed(futures):
                channel_name = futures[future]
                completion_time = time.time()
                try:
                    results = future.result()
                    output_manager.debug(f"📥 {channel_name.upper()} channel collection completed [time: {completion_time:.3f}] [since parallel start: {completion_time-parallel_start:.3f}s]")

                    if results:  # non-empty results
                        successful_channels += 1

                        if channel_name == 'sql':
                            mysql_results = results
                        elif channel_name == 'graph':
                            sqlite_graph_results = results
                        elif channel_name == 'vector':
                            vector_results = results

                        output_manager.debug(f"✅ {channel_name.upper()} channel: {len(results)} results")
                    else:
                        # Empty results: already distinguished by internal channel output (skipped or 0 results)
                        skipped_channels += 1
                        # No output here to avoid duplicate logs

                except Exception as e:
                    output_manager.error(f"❌ {channel_name.upper()} channel exception: {e}")

        parallel_end = time.time()
        output_manager.info(f"🚀 Parallel execution completed [time: {parallel_end:.3f}] [total duration: {parallel_end-parallel_start:.3f}s]")

        # ===== Result check (lenient strategy) =====

        total_channels = len(futures)
        output_manager.info(f"📊 Parallel execution statistics:")
        output_manager.info(f"   Total channels: {total_channels}")
        output_manager.info(f"   Successful channels: {successful_channels}")
        output_manager.info(f"   Skipped channels: {skipped_channels}")

        if successful_channels == 0:
            output_manager.warning(f"🚫 All channels returned no valid results")
        else:
            output_manager.success(f"🎉 {successful_channels}/{total_channels} channels executed successfully")

        return mysql_results, sqlite_graph_results, vector_results

    # ========== Ablation Experiment Specific Methods ==========

    def _execute_single_node_strict(self, node, memory, dag, context, node_type="retrieval") -> "UnifiedNodeResult":
        """
        Strict mode execution of single node (ablation experiment - disable dynamic switching)

        Args:
            node: DAG node
            memory: Memory object
            dag: DAG object
            context: parent node context
            node_type: forced node type ("retrieval" or "inference")

        Returns:
            UnifiedNodeResult object
        """
        node_start_time = time.time()

        # Force retrieval execution (no judgment on necessity)
        output_manager.info(f"[Strict Mode] Retrieval node (forced retrieval, no judgment)")

        # Call regular retrieval flow
        result = self._execute_with_retrieval(node, memory, context, error_hint="", dag=dag)

        # Mark as no switching
        result.predicted_node_type = "retrieval"
        result.actual_node_type = "retrieval"
        result.node_type_switched = False
        result.switch_reason = "Strict mode: no switching allowed"

        return result

    def _execute_inference_node_strict(self, node, memory, dag, context) -> "UnifiedNodeResult":
        """
        Strict mode execution of inference node (ablation experiment - disable dynamic switching)

        If parent node information is insufficient, LLM will honestly say "Unable to answer"
        (instead of switching to retrieval mode)

        Args:
            node: DAG node
            memory: Memory object
            dag: DAG object
            context: parent node context

        Returns:
            UnifiedNodeResult object
        """
        node_start_time = time.time()

        output_manager.info(f"[Strict Mode] Inference node (no retrieval, strict mode)")

        # Get parent node answers
        parent_answers_context = "\n".join(context.get('parent_answers', {}).values())

        # Replace template placeholders
        processed_question = self._replace_template_placeholders(node.question, context)

        # Direct LLM call (no retrieval)
        prompt = f"""Based on the following context, answer the question.

Context from parent nodes:
{parent_answers_context}

Question: {processed_question}

If the context is insufficient, respond: "Unable to answer based on provided information."

Answer directly without retrieval."""

        try:
            response = self.query_router.client.chat.completions.create(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            answer = response.choices[0].message.content.strip()

            execution_time = time.time() - node_start_time

            # Build result (removed non-existent status field)
            from src.core.dag_models import UnifiedNodeResult
            return UnifiedNodeResult(
                node_id=node.id,
                original_question=node.question,
                understood_question=processed_question,
                answer_text=answer,
                # status=NodeStatus.SUCCESS,  # ❌ UnifiedNodeResult does not have status field
                retrieval_contexts=[],  # no retrieval
                channels_used=[],
                execution_time=execution_time,
                predicted_node_type="inference",
                actual_node_type="inference",
                node_type_switched=False,
                switch_reason="Strict mode: no switching allowed",
                is_incremental=False
            )
        except Exception as e:
            output_manager.error(f"Inference node execution failed: {e}")
            raise
