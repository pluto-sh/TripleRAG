"""
Query Understanding and Routing Module
LLM-driven module that decomposes user queries into different retrieval tasks

Added DAG query decomposition functionality
"""
from typing import List, Dict, Any, Optional, Set
import json
from openai import OpenAI
from .models import QueryType, QueryPlan
from config.config import config
from ..utils.llm_normalizers import (
    normalize_sql_output,
    normalize_cypher_output,
    strip_yaml_code_fence,
    repair_linear_dag_yaml,
)
from ..utils.llm_client import build_openai_client
from ..utils.schema_loader import load_sql_schema, load_graph_schema
from ..utils.sql_parsers import parse_columns_map_from_sql_schema_text
from ..utils.output_manager import output_manager
from ..utils.llm_output_cleaner import extract_json_from_llm_output, safe_parse_json

class QueryRouter:
    """Query Router"""
    
    def __init__(self):
        self.client = build_openai_client()
        # load structured database and graph database schemas separately
        self.sql_schema = load_sql_schema()
        self.graph_schema = load_graph_schema()
        # runtime table list (injected by executor/NodeAgent) and canonical column mapping
        self.runtime_sql_tables: List[str] = []
        self.sql_columns_map: Dict[str, Set[str]] = parse_columns_map_from_sql_schema_text(self.sql_schema)
        # record latest LLM raw output (SQL/Cypher) for transparent display
        self.last_raw_sql_output: Optional[str] = None
        self.last_raw_cypher_output: Optional[str] = None
        # record recent multiple attempt raw outputs (for debugging)
        self.last_raw_sql_attempts: List[str] = []
        self.last_raw_cypher_attempts: List[str] = []

    # compatible with old interface to avoid affecting external callers (internally delegates to utils)
    def _load_sql_schema(self) -> str:
        return load_sql_schema()

    def _load_graph_schema(self) -> str:
        return load_graph_schema()
    
    def judge_retrieval_necessity(
        self,
        query: str,
        context: str = "",
        predicted_type: str = "retrieval"
    ) -> Dict[str, Any]:
        """
        Intelligently determine whether data retrieval is needed (for dynamic node type switching)

        Uses different judgment strategies based on predicted type:
        - retrieval node: judge if parent node info is sufficient (skip retrieval if sufficient)
        - inference node: judge if parent node info is sufficient (output incremental retrieval needs if insufficient)

        Args:
            query: node question
            context: parent node context
            predicted_type: predicted node type ("retrieval" or "inference")

        Returns: {
            "need_retrieval": bool,          # whether retrieval is needed
            "reason": str,                   # judgment reason
            "confidence": float,             # confidence (0-1)
            "information_gap": str,          # information gap (only for inference→retrieval)
            "incremental_query": str,        # incremental retrieval query (only for inference→retrieval)
            "target_channels": List[str]     # target retrieval channels (only for inference→retrieval)
        }
        """
        # build different system prompts based on predicted type
        if predicted_type == "retrieval":
            system_prompt = self._build_retrieval_node_judgment_prompt()
        else:  # inference
            system_prompt = self._build_inference_node_judgment_prompt()

        return self._execute_retrieval_judgment(query, context, system_prompt, predicted_type)

    def _build_retrieval_node_judgment_prompt(self) -> str:
        """build judgment prompt for retrieval nodes (simplified version)"""
        return """You are an intelligent path selector for RETRIEVAL nodes.

[Node Context]
This node was predicted as a RETRIEVAL node, which should query databases.

[Your Task]
Judge whether the parent node information is already sufficient to answer the current question.
- If sufficient → Set need_retrieval=false (skip retrieval)
- If insufficient → Set need_retrieval=true (perform retrieval)

[Output Format]
Please strictly follow the JSON format below:
{
    "need_retrieval": true,  // true or false
    "reason": "brief explanation",  // One sentence
    "confidence": 0.8  // 0.0-1.0
}

[Decision Logic]
- If parent node information already contains the answer → need_retrieval=false
- If new data needs to be queried from databases → need_retrieval=true
- If uncertain → conservatively choose true

[Examples]

Example 1:
Query: "Query the industry category of the company in {{p1.answer}}"
Parent info: "Company: SQM, industry: Lithium Mining, location: Chile"
Decision: need_retrieval=false
Reason: "Industry information (Lithium Mining) is already in parent node"

Example 2:
Query: "Query CATL's suppliers"
Parent info: "CATL is a battery manufacturer"
Decision: need_retrieval=true
Reason: "Supplier information not in parent node, database query needed"

Example 3:
Query: "Integrate information from p1 and p2"
Parent info: "p1: Company A data; p2: Company B data"
Decision: need_retrieval=false
Reason: "Pure integration task, no new data needed"
"""

    def _build_inference_node_judgment_prompt(self) -> str:
        """build judgment prompt for inference nodes (enhanced version, supports incremental retrieval)"""
        return """You are an intelligent path selector for INFERENCE nodes.

[Node Context]
This node was predicted as an INFERENCE node, which should perform logical inference based on parent information.

[Your Task]
Judge whether the parent node information is sufficient to answer the current question.

Case 1: Information is sufficient
→ Set need_retrieval=false
→ Only fill in: need_retrieval, reason, confidence

Case 2: Information is insufficient
→ Set need_retrieval=true
→ Must also fill in:
  - information_gap: What specific information is missing?
  - incremental_query: What should be queried to fill the gap? (precise query)
  - target_channels: Which channels to use? ["sql", "graph", "vector"] or subset

[Output Format]
Please strictly follow the JSON format below:
{
    "need_retrieval": true,  // true or false
    "reason": "brief explanation",
    "confidence": 0.8,

    // Only when need_retrieval=true:
    "information_gap": "what is missing",
    "incremental_query": "precise query for the missing info",
    "target_channels": ["sql", "graph", "vector"]  // or subset
}

[Channel Selection Guide]
- "sql": For structured data (quantities, prices, trade records, statistics)
- "graph": For relationships (supply chains, company networks, connections)
- "vector": For semantic content (descriptions, documents, text search)
- Default: ["sql", "graph", "vector"] (use all if uncertain)

[Examples]

Example 1: Information sufficient
Query: "Compare the production capacity of {{p1.answer}} and {{p2.answer}}"
Parent info: "p1: Company A capacity: 500 tons; p2: Company B capacity: 600 tons"
Decision:
{
    "need_retrieval": false,
    "reason": "Both capacities are in parent info, direct comparison possible",
    "confidence": 0.95
}

Example 2: Information insufficient (incremental retrieval needed)
Query: "Compare the carbon production of CATL's largest supplier and BYD's largest supplier"
Parent info: "CATL's largest supplier: Company A, carbon production: 500 tons. BYD's largest supplier: Company B"
Decision:
{
    "need_retrieval": true,
    "reason": "Company B's carbon production is missing",
    "confidence": 0.9,
    "information_gap": "Missing Company B's carbon production data",
    "incremental_query": "Query Company B's carbon production",
    "target_channels": ["sql", "graph"]
}

Example 3: Multiple missing pieces
Query: "Analyze the relationship between companies in {{p1.answer}}"
Parent info: "p1: Company names: A, B, C (no relationship data)"
Decision:
{
    "need_retrieval": true,
    "reason": "Company relationship data is missing",
    "confidence": 0.85,
    "information_gap": "Missing relationship data between companies A, B, C",
    "incremental_query": "Query relationships between companies A, B, and C",
    "target_channels": ["graph"]
}

[Important Notes]
1. Be precise with incremental_query - it should be a clear, specific question
2. Choose target_channels based on the type of missing information
3. If uncertain about channels, use all three: ["sql", "graph", "vector"]
4. information_gap should clearly state what is missing, not just "information insufficient"
"""

    def _execute_retrieval_judgment(
        self,
        query: str,
        context: str,
        system_prompt: str,
        predicted_type: str
    ) -> Dict[str, Any]:
        """execute retrieval judgment LLM call"""

        # Build complete context
        full_context = f"Current question: {query}"
        if context:
            full_context += f"\n\nAdditional context: {context}"

        user_prompt = f"""Please determine whether the following query requires data retrieval:

{full_context}

Please analyze and return the decision in JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1  # low temperature for consistency
                # removed max_tokens limit to ensure complete output
            )

            result = response.choices[0].message.content.strip()

            # clean LLM output, remove possible code block markers
            cleaned_result = strip_yaml_code_fence(result)

            # parse JSON response
            try:
                import json
                decision_data = json.loads(cleaned_result)

                # Validate required fields
                need_retrieval = decision_data.get("need_retrieval", True)
                reason = decision_data.get("reason", "No reason provided")
                confidence = float(decision_data.get("confidence", 0.5))

                # If confidence is too low, conservatively choose retrieval
                if confidence < 0.6:
                    need_retrieval = True
                    reason += " (Low confidence, conservatively choosing retrieval)"

                # build basic return result
                result_dict = {
                    "need_retrieval": need_retrieval,
                    "reason": reason,
                    "confidence": confidence,
                    "information_gap": "",
                    "incremental_query": "",
                    "target_channels": []
                }

                # if inference node and needs retrieval, parse incremental retrieval fields
                if predicted_type == "inference" and need_retrieval:
                    result_dict["information_gap"] = decision_data.get("information_gap", "")
                    result_dict["incremental_query"] = decision_data.get("incremental_query", "")
                    result_dict["target_channels"] = decision_data.get("target_channels", ["sql", "graph", "vector"])

                    # validate incremental retrieval fields
                    if not result_dict["incremental_query"]:
                        output_manager.warning("inference→retrieval switch but no incremental_query provided, using original question")
                        result_dict["incremental_query"] = query
                    if not result_dict["target_channels"]:
                        output_manager.warning("no target_channels specified, defaulting to full retrieval")
                        result_dict["target_channels"] = ["sql", "graph", "vector"]

                return result_dict

            except (json.JSONDecodeError, ValueError) as e:
                # JSON parsing failed, conservatively choose retrieval
                output_manager.warning(f"intelligent judgment JSON parse failed: {e}")
                output_manager.warning(f"raw output: {repr(result)}")
                output_manager.warning(f"cleaned output: {repr(cleaned_result)}")
                return {
                    "need_retrieval": True,
                    "reason": "Decision parsing failed, conservatively choosing retrieval",
                    "confidence": 0.0,
                    "information_gap": "",
                    "incremental_query": "",
                    "target_channels": []
                }

        except Exception as e:
            output_manager.error(f"intelligent judgment failed: {e}, using default need retrieval")
            return {
                "need_retrieval": True,
                "reason": f"Decision process exception: {str(e)}",
                "confidence": 0.0,
                "information_gap": "",
                "incremental_query": "",
                "target_channels": []
            }

    def analyze_query(self, query: str, context: str = None) -> QueryPlan:
        """analyze query and generate query plan"""

        system_prompt = f"""You are an intelligent query router responsible for analyzing user queries and determining which retrieval method to use.

[SQL Database Schema (ONLY reference tables and fields that actually exist in the SQL database!!!)]
{self.sql_schema}

[Graph Database Schema (ONLY reference nodes and relationships that actually exist in the graph database!!!)]
{self.graph_schema}

[Channel Selection Rules]
1.  **Priority Ranking**: Based on the query intent, rank the available channels (`mysql`, `neo4j`, `vector`) by priority.
2.  **MySQL Priority Scenarios**: When the query involves precise structured data (such as IDs, names, numerical values) or requires aggregation statistics (SUM/COUNT/AVG), `mysql` should be the highest priority.
3.  **Neo4j Priority Scenarios**: When the query focuses on relationships between entities (such as suppliers, customers, upstream/downstream) or requires graph traversal, `neo4j` should be the highest priority.
4.  **Vector Priority Scenarios**: When the query is vague, conceptual, requires semantic understanding, or when the needed fields cannot be found in structured or graph databases, `vector` can be the primary channel.
5.  **Single Channel vs Multi-Channel**:
    *   For simple, clear queries, return only one highest priority channel.
    *   For complex queries, return multiple channels by priority, and the system will try them in order.

[Output Format]
Please strictly follow the JSON format below, where `query_types` is a list of strings sorted by descending priority:
{{
    "query_types": ["mysql", "neo4j", "vector"],  // Example of channel list sorted by priority
    "inference": "Your analysis and decision process (1-2 sentences)."
}}
"""

        # Build user prompt, include context information if available
        context_part = f"\n\n[Context Information]\n{context}" if context else ""
        user_prompt = f"User query: {query}{context_part}\n\nPlease analyze this query and generate the query plan."

        try:
            response = self.client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )

            result = response.choices[0].message.content

            # parse JSON response
            try:
                plan_data = json.loads(result)

                # convert query types
                query_types = [QueryType(qt) for qt in plan_data.get("query_types", [])]

                # create query plan - no longer includes specific query statements, let LLM decide autonomously during execution
                query_plan = QueryPlan(
                    query_types=query_types,
                    inference=plan_data.get("inference", "")
                )

                return query_plan

            except json.JSONDecodeError:
                # If JSON parsing fails, return default vector retrieval plan
                return QueryPlan(
                    query_types=[QueryType.VECTOR],
                    inference="JSON parsing failed, using default vector retrieval"
                )

        except Exception as e:
            output_manager.error(f"query routing analysis failed: {e}")
            # Return default vector retrieval plan
            return QueryPlan(
                query_types=[QueryType.VECTOR],
                inference=f"Routing analysis failed: {str(e)}, using default vector retrieval"
            )

    # runtime table list update (called by DAG executor for unified schema)
    def update_runtime_sql_tables(self, tables: List[str]):
        # only keep tables that appear in canonical schema, exclude views and undeclared tables
        try:
            allowed = set(self.sql_columns_map.keys()) if hasattr(self, 'sql_columns_map') else set()
            self.runtime_sql_tables = [t for t in (tables or []) if (not allowed or t in allowed)]
        except Exception:
            self.runtime_sql_tables = tables or []

    def get_sql_columns_map(self) -> Dict[str, Set[str]]:
        """expose canonical column mapping for executor/NodeAgent use"""
        return self.sql_columns_map

    def get_runtime_sql_tables(self) -> List[str]:
        """return runtime table list filtered by canonical schema"""
        return self.runtime_sql_tables


    def decompose_query_to_dag(self, query: str, dag_type: str = "auto", dag_model: Optional[str] = None):
        """
        Let LLM freely decide DAG structure, or use rule-based decomposition (ablation experiment)

        Args:
            query: user query
            dag_type: DAG type (deprecated, kept for parameter compatibility)
            dag_model: optional, specify model for decomposition

        Returns:
            QueryDAG object
        """
        # check if using rule-based decomposition (ablation experiment)
        decomp_method = getattr(config.dag, 'decomposition_method', 'llm')

        if decomp_method == 'rule_based':
            output_manager.info(f"[Ablation Experiment] Using RULE-BASED decomposition")
            return self.decompose_query_with_rules(query, dag_type)
        else:
            output_manager.info(f"QueryRouter: Let LLM freely decide DAG structure")
            # directly call dual-LLM collaborative decomposition, let LLM freely design DAG structure based on query complexity
            return self.decompose_query_with_dual_llm(query, "free", dag_model)

    def decompose_query_with_rules(self, query: str, dag_type: str = "linear"):
        """
        Rule-based query decomposition (ablation experiment baseline)

        Rules:
        1. Split by question marks ("?")
        2. Split by sequential connectors ("then", "next", "after that")
        3. Detect comparison structures ("compare A and B", "A vs B")
        4. Default: linear DAG (p1 → p2 → p3)

        Args:
            query: user query (English)
            dag_type: ignored (kept for compatibility)

        Returns:
            QueryDAG object
        """
        from .dag_models import QueryDAG, DAGNode, DAGEdge

        # Rule 1: split by question marks
        sub_queries = [q.strip() for q in query.split('?') if q.strip()]

        # if no question marks, treat as single query
        if len(sub_queries) == 0:
            sub_queries = [query]

        # Rule 2: detect comparison structure
        comparison_keywords = ['compare', 'versus', 'vs', 'contrast', 'difference between']
        is_comparison = any(kw in query.lower() for kw in comparison_keywords)

        # Rule 3: detect sequential structure
        sequential_keywords = ['then', 'next', 'after that', 'subsequently', 'following']
        is_sequential = any(kw in query.lower() for kw in sequential_keywords)

        # build DAG structure
        dag = QueryDAG(original_query=query)
        nodes = []
        edges = []

        if is_comparison and len(sub_queries) == 1:
            # special handling: "Compare A and B" → p1, p2 (parallel), p3 (comparison)
            # extract entities (simplified heuristic)
            parts = query.lower().split(' and ')
            if len(parts) >= 2:
                # create parallel retrieval nodes
                node1 = DAGNode(
                    id="p1",
                    question=f"Query information about {parts[0].split()[-1]}",
                    parent_ids=[],
                    predicted_node_type="retrieval"
                )
                node2 = DAGNode(
                    id="p2",
                    question=f"Query information about {parts[1].split()[0]}",
                    parent_ids=[],
                    predicted_node_type="retrieval"
                )
                node3 = DAGNode(
                    id="p3",
                    question=f"Compare {{{{p1.answer}}}} and {{{{p2.answer}}}}",
                    parent_ids=["p1", "p2"],
                    predicted_node_type="inference"
                )
                nodes = [node1, node2, node3]
                # add edges: p1 → p3, p2 → p3
                edges.append(DAGEdge(from_node="p1", to_node="p3", dependency_reason="Provide first entity data"))
                edges.append(DAGEdge(from_node="p2", to_node="p3", dependency_reason="Provide second entity data"))
            else:
                # fallback to single node
                nodes = [DAGNode(
                    id="p1",
                    question=query,
                    parent_ids=[],
                    predicted_node_type="retrieval"
                )]
        else:
            # default: linear DAG (p1 → p2 → p3 → ...)
            for i, sub_q in enumerate(sub_queries):
                node_id = f"p{i+1}"

                # if sequential, add template reference to previous node
                if i > 0:
                    question = f"Based on {{{{p{i}.answer}}}}, {sub_q}"
                    parent_ids = [f"p{i}"]
                else:
                    question = sub_q
                    parent_ids = []

                node = DAGNode(
                    id=node_id,
                    question=question,
                    parent_ids=parent_ids,
                    predicted_node_type="retrieval"  # conservative: all as retrieval nodes
                )
                nodes.append(node)

                # add edge: p(i) → p(i+1)
                if i > 0:
                    edges.append(DAGEdge(
                        from_node=f"p{i}",
                        to_node=node_id,
                        dependency_reason="Sequential dependency"
                    ))

        # add nodes to DAG
        for node in nodes:
            dag.add_node(node)

        # fix: add edges to DAG
        for edge in edges:
            dag.add_edge(edge)

        # set metadata
        dag.query = query

        output_manager.info(f"[Rule-Based] generated {len(nodes)} nodes")
        for node in nodes:
            output_manager.info(f"  - {node.id}: {node.question} (dependencies: {node.parent_ids})")
        output_manager.info(f"[Rule-Based] generated {len(edges)} edges")

        return dag

    def decompose_query_with_dual_llm(self, query: str, dag_type: str = "free", dag_model: Optional[str] = None):
        """
        Dual-LLM collaborative DAG decomposition method - let LLM freely design DAG structure

        Args:
            query: user query
            dag_type: DAG type (deprecated, kept for parameter compatibility)
            dag_model: optional model name

        Returns:
            QueryDAG object or None (if decomposition failed)
        """
        from .dag_models import QueryDAG

        max_retries = 3

        for attempt in range(max_retries):
            try:
                output_manager.info(f"QueryRouter dual-LLM collaborative decomposition attempt {attempt + 1}/{max_retries}")

                # Step 1: Planner LLM - decomposition strategy
                planning_result = self._plan_decomposition(query, dag_type)
                if not planning_result:
                    output_manager.error(f"QueryRouter attempt {attempt + 1}: Planner LLM decomposition failed")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        output_manager.error("QueryRouter all retries failed, falling back to original method")
                        return self._fallback_to_original(query, dag_type, dag_model)

                # Step 2: Executor LLM - generate DAG JSON
                dag_json = self._execute_decomposition(query, planning_result, dag_model, attempt+1)
                if not dag_json:
                    output_manager.error(f"QueryRouter attempt {attempt + 1}: Executor LLM generate JSON failed")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        output_manager.error("QueryRouter all retries failed, falling back to original method")
                        return self._fallback_to_original(query, dag_type, dag_model)

                # Step 3: build DAG object
                dag = self._build_dag_from_json(dag_json, query)
                if dag:
                    output_manager.success(f"QueryRouter dual-LLM collaboration success in attempt {attempt + 1} generated DAG")
                    return dag
                else:
                    output_manager.error(f"QueryRouter attempt {attempt + 1}: DAG build failed")
                    if attempt < max_retries - 1:
                        continue

            except Exception as e:
                print(f"[QueryRouter] attempt {attempt + 1}: dual-LLM collaboration exception: {e}")
                if attempt < max_retries - 1:
                    continue

        # all retries failed, fallback to original method
        print("[QueryRouter] dual-LLM collaboration completely failed, falling back to original method")
        return self._fallback_to_original(query, dag_type, dag_model)

    def _plan_decomposition(self, query: str, dag_type: str) -> Optional[Dict]:
        """
        Planner LLM: analyze query and generate decomposition strategy

        Args:
            query: user query
            dag_type: DAG type

        Returns:
            decomposition plan dictionary or None
        """
        system_prompt = """You are a query decomposition planner that breaks down complex queries into subtasks with dependencies.

**Output Format** (Pure JSON):
{
  "inference": "Overall approach in 1-2 sentences",
  "decomposition": [
    {"step": 1, "purpose": "", "depends_on": [], "node_type": "retrieval"},
    {"step": 2, "purpose": "", "depends_on": [], "node_type": "inference"},
    {"step": 3, "purpose": "", "depends_on": [], "node_type": "retrieval"}
  ]
}

**node_type Prediction** (Predictive, not definitive):
- "retrieval" (retrieval node): Steps that LIKELY need database retrieval
  Examples: "Find X", "Query Y", "Retrieve Z"

- "inference" (inference node): Steps that LIKELY only need logical inference
  Examples: "Integrate X and Y", "Calculate Z based on X", "Compare A and B"

**Prediction Criteria**:
1. Is this a fact-lookup task? → "retrieval"
2. Is this an integration/calculation task? → "inference"
3. When uncertain → default to "retrieval"

**Important**: This is a PREDICTION. The system will dynamically adjust during execution.

**Examples**:

Example 1 - Linear Dependency (Both retrieval nodes):
Query: "Who is Tesla's largest supplier? What is the transaction volume between this supplier and BYD?"
{
  "inference": "User explicitly asked two questions, first find supplier, then query transaction volume, both need retrieval",
  "decomposition": [
    {"step": 1, "purpose": "Find the name of Tesla's largest supplier", "depends_on": [], "node_type": "retrieval"},
    {"step": 2, "purpose": "Query the transaction volume between step1 supplier and BYD", "depends_on": ["step1"], "node_type": "retrieval"}
  ]
}

Example 2 - Mixed Retrieval/Inference (1 retrieval, 1 inference):
Query: "What is the transaction situation between BYD's largest supplier and CATL?"
{
  "inference": "First need to find BYD's supplier via retrieval, then query transaction needs retrieval again",
  "decomposition": [
    {"step": 1, "purpose": "Find the name of BYD's largest supplier", "depends_on": [], "node_type": "retrieval"},
    {"step": 2, "purpose": "Query the transaction situation between step1 supplier and CATL", "depends_on": ["step1"], "node_type": "retrieval"}
  ]
}

Example 3 - Complex Inference (Multiple retrieval, one inference):
Query: "What is CATL's market share in China?"
{
  "inference": "Need to retrieve CATL sales and market total, then calculate the ratio via inference",
  "decomposition": [
    {"step": 1, "purpose": "Query CATL's annual sales volume", "depends_on": [], "node_type": "retrieval"},
    {"step": 2, "purpose": "Query China's battery market total sales volume", "depends_on": [], "node_type": "retrieval"},
    {"step": 3, "purpose": "Calculate CATL's market share = step1 / step2", "depends_on": ["step1", "step2"], "node_type": "inference"}
  ]
}

**Key Rules**:
- purpose must contain the specific entity names from the original query
- For multi-parent nodes, depends_on lists all parent steps: ["step1", "step2"]
- **Must explicitly annotate node_type for each step**
- Maximum 6 steps, each purpose no more than 100 characters
- Return only JSON, no code block markers"""

        user_prompt = f"User query: {query}\n\nPlease split the query into several dependent subtasks, output in the above JSON format, return only pure JSON."

        try:
            response = self.client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=config.llm.max_tokens
            )

            result_text = response.choices[0].message.content.strip()

            # use specialized cleaning tool to extract JSON
            planning_result = extract_json_from_llm_output(result_text)

            if planning_result:
                output_manager.debug(f"QueryRouter Planner LLM complete output: {planning_result}")
                output_manager.debug(f"QueryRouter Planner LLM inference: {planning_result.get('inference', 'N/A')}")
                return planning_result
            else:
                # if cleaning failed, log detailed info
                output_manager.error(f"QueryRouter Planner LLM JSON extraction failed")
                print(f"[QueryRouter] Planner raw output: {result_text[:500]}...")  # only print first 500 characters

                # try using more lenient parsing
                planning_result = safe_parse_json(result_text, fallback=None)
                if planning_result:
                    output_manager.warning(f"QueryRouter Planner LLM using lenient parse success")
                    return planning_result
                else:
                    return None
        except Exception as e:
            output_manager.error(f"QueryRouter Planner LLM call failed: {e}")
            return None

    def _execute_decomposition(self, query: str, planning_result: Dict, dag_model: Optional[str] = None, attempt: int = 1) -> Optional[Dict]:
        """
        Executor LLM: generate precise DAG JSON based on planning result

        Args:
            query: original user query
            planning_result: planner's output result
            dag_model: optional model name
            attempt: current attempt number (for logging)

        Returns:
            DAG JSON dictionary or None
        """
        selected_dag_model = dag_model or config.llm.model

        system_prompt = """You are a DAG JSON generator that converts decomposition plans into standard DAG JSON.

**Output Format** (Only this structure):
{
  "dag": {
    "nodes": [
      {"id": "p1", "question": "...", "dependencies": [], "predicted_node_type": "retrieval"},
      {"id": "p2", "question": "...", "dependencies": [], "predicted_node_type": "inference"},
      {"id": "p3", "question": "...", "dependencies": [], "predicted_node_type": "retrieval"}
    ]
  }
}

**Conversion Rules**:
- step N → node id: pN
- purpose → question (must preserve original meaning, add {{pX.answer}} references when needed)
- depends_on → dependencies (convert to node id list)
- **node_type → predicted_node_type (directly inherit from planning result, do not change)**

**Example**:
Decomposition plan: step1 find SQM company(retrieval), step2 query industry based on step1(retrieval), step3 query same industry companies based on step1+step2(retrieval)
Conversion result:
{
  "dag": {
    "nodes": [
      {"id": "p1", "question": "Find the operator company of SQM Atacama facility", "dependencies": [], "predicted_node_type": "retrieval"},
      {"id": "p2", "question": "Based on company:{{p1.answer}} query the industry of that company", "dependencies": ["p1"], "predicted_node_type": "retrieval"},
      {"id": "p3", "question": "Based on company industry:{{p2.answer}} find companies with the same industry, and exclude company:{{p1.answer}}", "dependencies": ["p1", "p2"], "predicted_node_type": "retrieval"}
    ]
  }
}

Return only JSON, no explanatory text."""

        # Format decomposition plan
        decomposition_text = f"**Original Query**: {query}\n\n**Decomposition Plan**:\n"
        decomposition_text += f"Overall approach: {planning_result.get('inference', 'N/A')}\n\n"

        for step_info in planning_result.get('decomposition', []):
            step_num = step_info.get('step')
            purpose = step_info.get('purpose')
            depends_on = step_info.get('depends_on', [])
            node_type = step_info.get('node_type', 'retrieval')  # default to retrieval
            decomposition_text += f"Step {step_num}: {purpose}\n"
            decomposition_text += f"  Dependencies: {', '.join(depends_on) if depends_on else 'None'}\n"
            decomposition_text += f"  Predicted node type: {node_type}\n"  # pass predicted type
            decomposition_text += "\n"

        # Streamlined: don't add extra key dependency text, keep prompt concise

        user_prompt = f"""{decomposition_text}

Please generate JSON containing only dag.nodes, including four fields for each node: id, question, dependencies, predicted_node_type. Return only JSON."""

        try:
            response = self.client.chat.completions.create(
                model=selected_dag_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=config.llm.max_tokens
            )

            result_text = response.choices[0].message.content.strip()

            # use specialized cleaning tool to extract JSON
            dag_json = extract_json_from_llm_output(result_text)

            if dag_json:
                output_manager.success(f"QueryRouter Executor LLM success generate DAG JSON (attempt {attempt}/3)")
                return dag_json
            else:
                # if cleaning failed, log detailed info
                output_manager.error(f"QueryRouter Executor LLM JSON extraction failed (attempt {attempt}/3)")
                print(f"[QueryRouter] raw output first 500 characters: {result_text[:500]}...")  # only print first 500 characters to avoid too long

                # try using more lenient parsing
                dag_json = safe_parse_json(result_text, fallback=None)
                if dag_json:
                    output_manager.warning(f"QueryRouter Executor LLM using lenient parse success (attempt {attempt}/3)")
                    return dag_json
                else:
                    output_manager.error(f"QueryRouter Executor LLM lenient parse also failed (attempt {attempt}/3)")
                    return None
        except Exception as e:
            print(f"[QueryRouter] Executor LLM call failed: {e}")
            return None

    def _build_dag_from_json(self, dag_json: Dict, original_query: str):
        """
        build DAG object from JSON (reuse existing logic)

        Args:
            dag_json: JSON representation of DAG
            original_query: original user query

        Returns:
            QueryDAG object or None
        """
        from .dag_models import QueryDAG, DAGNode, DAGEdge

        try:
            dag = QueryDAG(original_query=original_query)

            # parse node data
            dag_data = dag_json.get('dag', dag_json)
            nodes_data = dag_data.get('nodes', [])

            if not nodes_data:
                print(f"[QueryRouter] node data not found in JSON")
                return None

            # create all nodes
            for node_data in nodes_data:
                node_id = node_data.get('id')
                question = node_data.get('question')
                if not node_id or not question:
                    continue

                # read predicted_node_type, default to retrieval
                predicted_type = node_data.get('predicted_node_type', 'retrieval')

                node = DAGNode(
                    id=node_id,
                    question=question,
                    decomposition_reason=f"dual-LLM collaborative decomposition subtask",
                    parent_ids=node_data.get('dependencies', []).copy(),
                    predicted_node_type=predicted_type,  # set predicted type
                    actual_node_type=predicted_type       # initially same as predicted type
                )

                dag.add_node(node)

            # create dependency relationships - fixed version
            for node_data in nodes_data:
                node_id = node_data.get('id', '')
                dependencies = node_data.get('dependencies', [])

                # ensure dependencies is list format
                if isinstance(dependencies, str):
                    deps_text = dependencies.replace('[', '').replace(']', '')
                    dependencies = [d.strip() for d in deps_text.split(',') if d.strip()]
                elif not isinstance(dependencies, list):
                    dependencies = []

                # create edge for each dependency relationship
                for dep_id in dependencies:
                    if dep_id in dag.nodes and node_id in dag.nodes:
                        edge = DAGEdge(
                            from_node=dep_id,
                            to_node=node_id,
                            dependency_reason=f"dual-LLM collaborative dependency relationship: {dep_id}→{node_id}"
                        )
                        dag.add_edge(edge)
                        print(f"[QueryRouter] create edge: {dep_id} -> {node_id}")
                    else:
                        print(f"[QueryRouter] warning: cannot create edge {dep_id} -> {node_id}, node does not exist")

            # before validating DAG structure, check and fix possible orphan node issues
            root_nodes = dag.get_root_nodes()
            leaf_nodes = dag.get_leaf_nodes()

            # if there are multiple root nodes and multiple leaf nodes, add connections to avoid orphan nodes
            if len(root_nodes) > 1 and len(leaf_nodes) > 1:
                # for each root node, if it's also a leaf node (no children), connect it to next node
                node_ids = list(dag.nodes.keys())
                for i in range(len(node_ids) - 1):
                    current_node = dag.nodes[node_ids[i]]
                    next_node = dag.nodes[node_ids[i + 1]]

                    # if current node has no children and next node is not its child, add dependency relationship
                    if not current_node.children_ids and next_node.id not in current_node.children_ids:
                        # update node parent-child relationships
                        current_node.children_ids.append(next_node.id)
                        next_node.parent_ids.append(current_node.id)

                        # create edge
                        edge = DAGEdge(
                            from_node=current_node.id,
                            to_node=next_node.id,
                            dependency_reason=f"automatically added dependency relationship to avoid orphan nodes: {current_node.id}→{next_node.id}"
                        )
                        dag.edges.append(edge)
                        print(f"[QueryRouter] automatically add edge: {current_node.id} -> {next_node.id}")

            # validate DAG structure
            from src.core.dag_validator import validate_dag
            is_valid, errors, suggestions = validate_dag(dag, strict=False)

            if not is_valid:
                # filter out orphan node errors, as we've already tried to fix them
                filtered_errors = [error for error in errors if "orphan node" not in error.lower()]
                if filtered_errors:
                    output_manager.error(f"QueryRouter DAG validate failed:")
                    for error in filtered_errors:
                        output_manager.error(f"  - {error}")
                    return None
                else:
                    # only orphan node errors, try to continue
                    output_manager.warning("QueryRouter DAG has orphan nodes, but attempted to fix")

            # calculate execution plan
            execution_plan = dag.get_execution_plan()
            output_manager.info(f"QueryRouter dual-LLM collaborative DAG execution plan: {execution_plan}")

            return dag

        except Exception as e:
            output_manager.error(f"QueryRouter DAG build failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _fallback_to_original(self, query: str, dag_type: str, dag_model: Optional[str] = None):
        """
        fallback to original single-LLM decomposition method

        Args:
            query: user query
            dag_type: DAG type
            dag_model: optional model name

        Returns:
            QueryDAG object or None
        """
        return None