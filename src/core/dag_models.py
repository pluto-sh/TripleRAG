"""
DAG Core Data Structures
Supports linear DAG structures with interfaces reserved for future extensions
Added adaptive optimizer related data models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import time


class NodeStatus(Enum):
    """Node Status"""
    PENDING = "pending"      # pending
    RUNNING = "running"      # running
    COMPLETED = "completed"  # completed
    FAILED = "failed"        # failed
    SKIPPED = "skipped"      # skipped


# ========== Adaptive Optimizer Data Models ==========

class NodeExecutionMode(Enum):
    """Node Execution Mode (kept for backward compatibility, actually deprecated)"""
    NORMAL = "normal"      # normal execution (retrieval + generation)
    # SKIP mode removed, now using actual_node_type to control execution behavior


@dataclass
class InsertionPlan:
    """Node Insertion Plan"""
    parent_id: str           # parent node ID
    child_id: str            # child node ID (will be deferred)
    new_node_question: str   # new node question description
    missing_fields: List[str]  # list of missing fields
    reason: str              # insertion reason (for logging)


@dataclass
class OptimizationResult:
    """Optimization result after each layer execution (simplified)"""
    node_execution_modes: Dict[str, NodeExecutionMode] = field(default_factory=dict)
    # final layer insertion related
    needs_final_insertion: bool = False
    final_insertion_reason: str = ""

    def has_mode_changes(self) -> bool:
        """whether there are execution mode changes"""
        return len(self.node_execution_modes) > 0

    def should_apply_optimizations(self) -> bool:
        """whether optimization operations need to be applied"""
        return self.has_mode_changes() or self.needs_final_insertion


@dataclass
class DAGNode:
    """
    DAG Node
    Enhanced: Full support for branching and convergence, multiple parent and child nodes
    Added adaptive optimizer execution mode support
    Added explicit/implicit node type support (for RAGAS evaluation optimization)
    """
    id: str                                    # Node ID, e.g. "p1", "p2"
    question: str                              # sub-question
    parent_ids: List[str] = field(default_factory=list)    # list of parent node IDs
    children_ids: List[str] = field(default_factory=list)  # list of child node IDs

    # execution status
    status: NodeStatus = NodeStatus.PENDING

    # metadata (for explainability)
    decomposition_reason: str = ""             # reason for decomposing this sub-question

    # node type (retrieval/inference classification - dynamic switching)
    predicted_node_type: str = "retrieval"     # "retrieval": LLM predicts retrieval may be needed
                                              # "inference": LLM predicts only inference may be needed
    actual_node_type: str = "retrieval"        # actual type after execution (may differ from predicted)

    # topological information
    topological_rank: int = -1                 # topological rank for execution ordering

    # execution result (filled after execution)
    result: Optional[Dict[str, Any]] = None    # execution result
    execution_time: float = 0.0                # execution time
    error_message: str = ""                    # error message (if failed)

    # adaptive optimizer execution mode (set by optimizer)
    execution_mode: NodeExecutionMode = NodeExecutionMode.NORMAL

    def __post_init__(self):
        """post-initialization validation"""
        if not self.id:
            raise ValueError("Node ID cannot be empty")
        if not self.question:
            raise ValueError("Node question cannot be empty")

    def is_convergence_node(self) -> bool:
        """determine if this is a convergence node (has multiple parents)"""
        return len(self.parent_ids) > 1

    def is_branch_node(self) -> bool:
        """determine if this is a branch node (has multiple children)"""
        return len(self.children_ids) > 1

    def is_root_node(self) -> bool:
        """determine if this is a root node (no parents)"""
        return len(self.parent_ids) == 0

    def is_leaf_node(self) -> bool:
        """determine if this is a leaf node (no children)"""
        return len(self.children_ids) == 0
    
    def mark_running(self):
        """mark as running"""
        self.status = NodeStatus.RUNNING

    def mark_completed(self, result: Dict[str, Any], execution_time: float):
        """mark as completed"""
        self.status = NodeStatus.COMPLETED
        self.result = result
        self.execution_time = execution_time

    def mark_failed(self, error_message: str):
        """mark as failed"""
        self.status = NodeStatus.FAILED
        self.error_message = error_message

    def mark_skipped(self, reason: str = ""):
        """mark as skipped (based on failure strategy)"""
        self.status = NodeStatus.SKIPPED
        if reason:
            self.error_message = reason

    def is_completed(self) -> bool:
        """check if completed"""
        return self.status == NodeStatus.COMPLETED

    def is_failed(self) -> bool:
        """check if failed"""
        return self.status == NodeStatus.FAILED
    
    def can_execute(self, completed_nodes: set) -> bool:
        """
        determine if node can be executed
        condition: all parent nodes must be completed

        Enhanced: supports multiple parent nodes, all must be completed before execution
        """
        if not self.parent_ids:
            return True  # root node can execute

        return all(parent_id in completed_nodes for parent_id in self.parent_ids)


@dataclass
class DAGEdge:
    """DAG Edge (dependency relationship)"""
    from_node: str                             # source node ID
    to_node: str                               # target node ID
    dependency_reason: str = ""                # dependency reason

    def __post_init__(self):
        if not self.from_node or not self.to_node:
            raise ValueError("Edge source and target nodes cannot be empty")
        if self.from_node == self.to_node:
            raise ValueError("Cannot depend on itself")


@dataclass
class QueryDAG:
    """
    Query DAG
    Supports linear structures
    """
    original_query: str                        # original query
    nodes: Dict[str, DAGNode] = field(default_factory=dict)  # node dictionary
    edges: List[DAGEdge] = field(default_factory=list)       # edge list
    
    # topological sorting related
    topological_order: List[str] = field(default_factory=list)  # topological order
    rank_groups: Dict[int, List[str]] = field(default_factory=dict)  # grouped by rank
    
    # metadata
    created_at: float = field(default_factory=time.time)

    def add_node(self, node: DAGNode):
        """add node"""
        if node.id in self.nodes:
            raise ValueError(f"duplicate node ID: {node.id}")
        self.nodes[node.id] = node

    def add_edge(self, edge: DAGEdge):
        """add edge"""
        # validate node existence
        if edge.from_node not in self.nodes:
            raise ValueError(f"source node does not exist: {edge.from_node}")
        if edge.to_node not in self.nodes:
            raise ValueError(f"target node does not exist: {edge.to_node}")

        # update parent-child relationships
        from_node = self.nodes[edge.from_node]
        to_node = self.nodes[edge.to_node]

        if edge.to_node not in from_node.children_ids:
            from_node.children_ids.append(edge.to_node)
        if edge.from_node not in to_node.parent_ids:
            to_node.parent_ids.append(edge.from_node)

        self.edges.append(edge)

    def get_root_nodes(self) -> List[DAGNode]:
        """get root nodes (nodes with no parents)"""
        return [node for node in self.nodes.values() if not node.parent_ids]

    def get_leaf_nodes(self) -> List[DAGNode]:
        """get leaf nodes (nodes with no children)"""
        return [node for node in self.nodes.values() if not node.children_ids]

    def get_node(self, node_id: str) -> Optional[DAGNode]:
        """get node"""
        return self.nodes.get(node_id)

    def get_completed_nodes(self) -> set:
        """get set of all completed node IDs"""
        return {node_id for node_id, node in self.nodes.items() if node.is_completed()}

    def is_linear(self) -> bool:
        """check if structure is linear: each node has at most one parent and one child"""
        for node in self.nodes.values():
            if len(node.parent_ids) > 1 or len(node.children_ids) > 1:
                return False
        return True

    def is_branching(self) -> bool:
        """
        check if structure is branching: supports multiple parents and children
        Enhanced: check if it's a valid branching DAG (non-linear but valid)
        """
        if self.is_linear():
            return False  # linear structure is not branching

        # check if there are multiple parents or children (characteristic of branching DAG)
        has_multiple_parents = any(len(node.parent_ids) > 1 for node in self.nodes.values())
        has_multiple_children = any(len(node.children_ids) > 1 for node in self.nodes.values())

        return has_multiple_parents or has_multiple_children

    def get_dag_type(self) -> str:
        """
        get DAG type
        Returns:
            "linear" | "branching" | "invalid"
        """
        if self.is_linear():
            return "linear"
        elif self.is_branching():
            return "branching"
        else:
            return "invalid"

    def get_linear_order(self) -> List[str]:
        """
        get execution order for linear DAG (deprecated, kept for backward compatibility)
        now unified to use topological_sort
        """
        # for backward compatibility, if it's truly linear, return linear order
        # otherwise use topological sort
        if self.is_linear():
            # traverse from root node
            roots = self.get_root_nodes()
            if not roots:
                return []

            order = []
            current = roots[0]

            while current:
                order.append(current.id)

                # find next node
                if current.children_ids:
                    current = self.nodes[current.children_ids[0]]
                else:
                    break

            return order
        else:
            # non-linear structure uses topological sort
            return self.topological_sort()

    def topological_sort(self, algorithm: str = "kahn") -> List[str]:
        """
        Enhanced: topological sort

        Args:
            algorithm: sorting algorithm ("kahn" | "dfs")

        Returns:
            list of node IDs after topological sort
        """
        if algorithm == "kahn":
            return self._kahn_topological_sort()
        elif algorithm == "dfs":
            return self._dfs_topological_sort()
        else:
            raise ValueError(f"unsupported topological sort algorithm: {algorithm}")

    def _kahn_topological_sort(self) -> List[str]:
        """
        Kahn's algorithm for topological sort
        suitable for DAG execution as it identifies nodes that can be executed in parallel at each level
        """
        from collections import deque

        # calculate in-degree for each node
        in_degree = {node_id: len(node.parent_ids) for node_id, node in self.nodes.items()}

        # find all nodes with in-degree 0 (root nodes)
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])

        result = []

        while queue:
            # take out a node with in-degree 0
            current = queue.popleft()
            result.append(current)

            # decrease in-degree of all child nodes
            current_node = self.nodes[current]
            for child_id in current_node.children_ids:
                in_degree[child_id] -= 1
                # if child node's in-degree becomes 0, add to queue
                if in_degree[child_id] == 0:
                    queue.append(child_id)

        # check if all nodes were processed (cycle detection)
        if len(result) != len(self.nodes):
            remaining = set(self.nodes.keys()) - set(result)
            raise ValueError(f"DAG contains cycle, unprocessed nodes: {remaining}")

        return result

    def _dfs_topological_sort(self) -> List[str]:
        """
        DFS-based topological sort
        """
        visited = set()
        temp_visited = set()
        result = []

        def dfs_visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"DAG contains cycle, involving node: {node_id}")

            if node_id in visited:
                return

            temp_visited.add(node_id)

            # visit all child nodes
            node = self.nodes[node_id]
            for child_id in node.children_ids:
                dfs_visit(child_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            # post-order traversal, so insert at beginning
            result.insert(0, node_id)

        # perform DFS on all unvisited nodes
        for node_id in self.nodes.keys():
            if node_id not in visited:
                dfs_visit(node_id)

        return result

    def get_topological_ranks(self) -> Dict[int, List[str]]:
        """
        Enhanced: get topological rank grouping
        returns nodes grouped by rank, nodes at the same level can be executed in parallel

        Returns:
            {rank: [node_ids]}, lower rank executes earlier
        """
        # use modified Kahn's algorithm to calculate levels
        from collections import deque

        in_degree = {node_id: len(node.parent_ids) for node_id, node in self.nodes.items()}
        ranks = {}
        current_rank = 0

        # first level: all nodes with in-degree 0
        current_level = [node_id for node_id, degree in in_degree.items() if degree == 0]

        while current_level:
            ranks[current_rank] = current_level.copy()

            # update topological rank for these nodes
            for node_id in current_level:
                self.nodes[node_id].topological_rank = current_rank

            next_level = []
            # process all nodes in current level
            for node_id in current_level:
                node = self.nodes[node_id]
                for child_id in node.children_ids:
                    in_degree[child_id] -= 1
                    if in_degree[child_id] == 0:
                        next_level.append(child_id)

            current_level = list(set(next_level))  # deduplicate
            current_rank += 1

        # update DAG's sorting information
        self.rank_groups = ranks

        return ranks

    def get_execution_plan(self) -> Dict[str, Any]:
        """
        Enhanced: generate execution plan
        includes topological sort, level grouping, and other information

        Returns:
            execution plan dictionary
        """
        topo_order = self.topological_sort("kahn")
        rank_groups = self.get_topological_ranks()

        self.topological_order = topo_order

        return {
            "topological_order": topo_order,
            "rank_groups": rank_groups,
            "execution_strategy": "sequential",  # only supports sequential execution
            "total_ranks": len(rank_groups),
            "can_parallelize": any(len(nodes) > 1 for nodes in rank_groups.values()),
            "dag_type": self.get_dag_type(),
            "statistics": self.get_statistics()
        }

    def validate_structure(self) -> tuple[bool, str]:
        """
        Enhanced: validate DAG structure validity (supports linear and branching)
        Returns: (is_valid, error_message)
        """
        if not self.nodes:
            return False, "DAG is empty"

        # use DAG validator for complete validation
        try:
            from src.core.dag_validator import validate_dag
            is_valid, errors, _ = validate_dag(self, strict=True)
            error_msg = "; ".join(errors) if errors else ""
            return is_valid, error_msg
        except ImportError:
            # fallback to basic validation
            return self._basic_structure_validation()

    def _basic_structure_validation(self) -> tuple[bool, str]:
        """basic structure validation (fallback method)"""
        # check root nodes
        roots = self.get_root_nodes()
        if not roots:
            return False, "no root nodes"

        # check connectivity (unified using topological sort)
        try:
            topo_order = self.topological_sort()
            if len(topo_order) != len(self.nodes):
                return False, "unreachable nodes or cycles exist"
        except Exception as e:
            return False, f"structure error: {str(e)}"

        return True, ""

    def validate_linear_structure(self) -> tuple[bool, str]:
        """
        validate linear structure validity (deprecated, no longer enforces linearity)
        now unified to use validate_structure for validation
        """
        # for backward compatibility, delegate to generic structure validation
        return self.validate_structure()

    def get_statistics(self) -> Dict[str, Any]:
        """get DAG statistics"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "dag_type": self.get_dag_type(),  # Enhanced: includes DAG type
            "is_linear": self.is_linear(),
            "is_branching": self.is_branching(),  # Enhanced: added
            "root_count": len(self.get_root_nodes()),
            "leaf_count": len(self.get_leaf_nodes()),
            "completed_nodes": len(self.get_completed_nodes()),
            "pending_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.PENDING]),
            "failed_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.FAILED]),
            # Enhanced: more branching DAG related statistics
            "max_parents": max((len(n.parent_ids) for n in self.nodes.values()), default=0),
            "max_children": max((len(n.children_ids) for n in self.nodes.values()), default=0),
            "convergence_nodes": len([n for n in self.nodes.values() if len(n.parent_ids) > 1]),  # convergence node count
            "branch_nodes": len([n for n in self.nodes.values() if len(n.children_ids) > 1]),  # branch node count
        }

    def __repr__(self) -> str:
        """string representation"""
        stats = self.get_statistics()
        return f"QueryDAG(nodes={stats['total_nodes']}, edges={stats['total_edges']}, linear={stats['is_linear']})"


# ====== Unified Node Execution Result Type ======
@dataclass
class UnifiedNodeResult:
    """
    Unified node execution result (optimized version)
    removed redundant fields, added LLM-understood question and dynamic switching tracking
    """
    node_id: str
    original_question: str      # original question (with template)
    understood_question: str    # specific question understood and expressed by LLM
    answer_text: str

    # metadata for statistics and debugging
    execution_time: float
    channels_used: List[str]

    # dynamic switching tracking fields
    predicted_node_type: str = "retrieval"     # predicted type during DAG construction
    actual_node_type: str = "retrieval"        # actual type during execution
    node_type_switched: bool = False           # whether type switching occurred
    switch_reason: str = ""                    # switching reason explanation

    # incremental retrieval fields
    is_incremental: bool = False                    # whether incremental retrieval was used
    information_gap: str = ""                       # information gap description
    incremental_query: str = ""                     # incremental retrieval query
    incremental_retrieval_failed: bool = False      # whether incremental retrieval failed

    # redundant fields (kept for backward compatibility, but not used in final answer generation)
    channel_results: Dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0  # default value, no longer used
    query_plan: Optional[Dict[str, Any]] = None  # kept for backward compatibility

    # new: store actual retrieval contexts (original retrieval results)
    retrieval_contexts: List[str] = field(default_factory=list)

    # compatibility with old code: provide question and actual_question property mappings
    @property
    def question(self) -> str:
        return self.original_question

    @property
    def actual_question(self) -> str:
        """backward compatibility: map to understood_question"""
        return self.understood_question

    def to_dict(self) -> Dict[str, Any]:
        """convert result to dictionary (optimized version + dynamic switching tracking + incremental retrieval)"""
        return {
            "node_id": self.node_id,
            "original_question": self.original_question,
            "understood_question": self.understood_question,
            "answer_text": self.answer_text,
            "execution_time": self.execution_time,
            "channels_used": self.channels_used,
            # dynamic switching information
            "predicted_node_type": self.predicted_node_type,
            "actual_node_type": self.actual_node_type,
            "node_type_switched": self.node_type_switched,
            "switch_reason": self.switch_reason,
            # incremental retrieval information
            "is_incremental": self.is_incremental,
            "information_gap": self.information_gap,
            "incremental_query": self.incremental_query,
            "incremental_retrieval_failed": self.incremental_retrieval_failed,
            # backward compatibility fields
            "channel_results": self.channel_results,
            "confidence": self.confidence,
            "query_plan": self.query_plan,
            # new: retrieval contexts
            "retrieval_contexts": self.retrieval_contexts
        }


def create_linear_dag(query: str, subproblems: List[Dict[str, str]]) -> QueryDAG:
    """
    helper function to create linear DAG

    Args:
        query: original query
        subproblems: list of sub-questions, format: [
            {"id": "p1", "question": "...", "reason": "..."},
            {"id": "p2", "question": "...", "reason": "..."},
            ...
        ]

    Returns:
        QueryDAG object
    """
    dag = QueryDAG(original_query=query)

    # add all nodes
    for sp in subproblems:
        node = DAGNode(
            id=sp["id"],
            question=sp["question"],
            decomposition_reason=sp.get("reason", "")
        )
        dag.add_node(node)

    # add linear edges (p1→p2→p3→...)
    for i in range(len(subproblems) - 1):
        edge = DAGEdge(
            from_node=subproblems[i]["id"],
            to_node=subproblems[i + 1]["id"],
            dependency_reason=f"{subproblems[i+1]['id']} depends on {subproblems[i]['id']}'s result"
        )
        dag.add_edge(edge)

    return dag
