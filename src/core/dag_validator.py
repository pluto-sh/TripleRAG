"""
DAG Validator
Responsible for validating DAG structure correctness, including cycle detection, connectivity validation, etc.

Features include:
1. Cycle detection (DFS-based cycle detection)
2. Orphan node detection
3. Connectivity validation
4. Branch limit validation
5. Dependency relationship reasonableness check
"""
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from src.core.dag_models import QueryDAG, DAGNode
from config.config import config


class DAGValidationError(Exception):
    """DAG validation error"""
    pass


class DAGValidator:
    """
    DAG Validator

    Provides complete branching DAG validation functionality
    """

    def __init__(self):
        """Initialize validator"""
        # Load validation parameters from config
        dag_config = getattr(config, 'dag', None)
        self.max_nodes = getattr(dag_config, 'max_nodes', 15) if dag_config else 15
        self.max_branches = getattr(dag_config, 'max_branches', 5) if dag_config else 5
        self.max_convergence = getattr(dag_config, 'max_convergence', 3) if dag_config else 3
        self.max_dag_depth = getattr(dag_config, 'max_dag_depth', 6) if dag_config else 6

        validation_config = getattr(dag_config, 'validation', None) if dag_config else None
        self.check_cycles = getattr(validation_config, 'check_cycles', True) if validation_config else True
        self.check_orphans = getattr(validation_config, 'check_orphans', True) if validation_config else True
        self.check_connectivity = getattr(validation_config, 'check_connectivity', True) if validation_config else True
        self.strict_validation = getattr(validation_config, 'strict_validation', True) if validation_config else True

    def validate_dag(self, dag: QueryDAG) -> Tuple[bool, List[str]]:
        """
        Complete DAG structure validation

        Args:
            dag: DAG to validate

        Returns:
            (is_valid, error_list)
        """
        errors = []

        # Basic structure check
        basic_errors = self._validate_basic_structure(dag)
        errors.extend(basic_errors)

        # If basic structure has issues, don't continue with subsequent validation
        if basic_errors and self.strict_validation:
            return False, errors

        # Cycle detection (must be before depth calculation)
        if self.check_cycles:
            cycle_errors = self._check_cycles(dag)
            errors.extend(cycle_errors)
            # If cycles found, don't continue with depth-related validation
            if cycle_errors:
                return False, errors

        # Orphan node detection
        if self.check_orphans:
            orphan_errors = self._check_orphans(dag)
            errors.extend(orphan_errors)

        # Connectivity validation
        if self.check_connectivity:
            connectivity_errors = self._check_connectivity(dag)
            errors.extend(connectivity_errors)

        # Branch limit validation
        limit_errors = self._check_branch_limits(dag)
        errors.extend(limit_errors)

        # Depth check (only when no cycles)
        depth_errors = self._check_dag_depth(dag)
        errors.extend(depth_errors)

        return len(errors) == 0, errors

    def _validate_basic_structure(self, dag: QueryDAG) -> List[str]:
        """Validate basic structure"""
        errors = []

        # Check if empty
        if not dag.nodes:
            errors.append("DAG cannot be empty")
            return errors

        # Check node count limit
        if len(dag.nodes) > self.max_nodes:
            errors.append(f"Node count ({len(dag.nodes)}) exceeds limit ({self.max_nodes})")

        # Check node ID uniqueness
        node_ids = set()
        for node_id in dag.nodes.keys():
            if node_id in node_ids:
                errors.append(f"Duplicate node ID: {node_id}")
            node_ids.add(node_id)

        # Check edge validity
        for edge in dag.edges:
            if edge.from_node not in dag.nodes:
                errors.append(f"Edge references non-existent source node: {edge.from_node}")
            if edge.to_node not in dag.nodes:
                errors.append(f"Edge references non-existent target node: {edge.to_node}")
            if edge.from_node == edge.to_node:
                errors.append(f"Self-loop not allowed: {edge.from_node}")

        # Check parent-child relationship consistency
        for node_id, node in dag.nodes.items():
            # Check parent node relationships
            for parent_id in node.parent_ids:
                if parent_id not in dag.nodes:
                    errors.append(f"Node {node_id} references non-existent parent node: {parent_id}")
                else:
                    parent_node = dag.nodes[parent_id]
                    if node_id not in parent_node.children_ids:
                        errors.append(f"Parent-child relationship inconsistent: {parent_id} -> {node_id}")

            # Check child node relationships
            for child_id in node.children_ids:
                if child_id not in dag.nodes:
                    errors.append(f"Node {node_id} references non-existent child node: {child_id}")
                else:
                    child_node = dag.nodes[child_id]
                    if node_id not in child_node.parent_ids:
                        errors.append(f"Parent-child relationship inconsistent: {node_id} -> {child_id}")

        return errors

    def _check_cycles(self, dag: QueryDAG) -> List[str]:
        """
        Cycle detection - using DFS algorithm
        """
        errors = []

        # Use DFS to detect cycles
        visited = set()
        rec_stack = set()

        def dfs(node_id: str, path: List[str]) -> Optional[List[str]]:
            """DFS cycle detection, returns cycle path (if exists)"""
            if node_id in rec_stack:
                # Found cycle, return cycle path
                cycle_start = path.index(node_id)
                return path[cycle_start:] + [node_id]

            if node_id in visited:
                return None

            visited.add(node_id)
            rec_stack.add(node_id)

            # Visit all child nodes
            node = dag.nodes.get(node_id)
            if node:
                for child_id in node.children_ids:
                    cycle = dfs(child_id, path + [node_id])
                    if cycle:
                        return cycle

            rec_stack.remove(node_id)
            return None

        # Start DFS from each unvisited node
        for node_id in dag.nodes.keys():
            if node_id not in visited:
                cycle = dfs(node_id, [])
                if cycle:
                    cycle_str = " -> ".join(cycle)
                    errors.append(f"Cycle detected: {cycle_str}")

        return errors

    def _check_orphans(self, dag: QueryDAG) -> List[str]:
        """Detect orphan nodes (nodes with neither parents nor children)"""
        errors = []

        # If DAG has only one node, don't consider it an orphan
        if len(dag.nodes) <= 1:
            return errors

        orphans = []
        for node_id, node in dag.nodes.items():
            # Only consider a node as orphan if it has neither parents nor children
            # For multi-root scenarios, allow multiple nodes without parents
            if not node.parent_ids and not node.children_ids:
                orphans.append(node_id)

        if orphans:
            # Decide whether it's an error or warning based on strictness
            if self.strict_validation:
                errors.append(f"Orphan nodes detected: {', '.join(orphans)}")
            else:
                # In non-strict mode, treat orphan nodes as warnings not errors
                # But still return info for upper layer handling
                pass

        return errors

    def _check_connectivity(self, dag: QueryDAG) -> List[str]:
        """Check connectivity - ensure all nodes are reachable from root nodes"""
        errors = []

        if len(dag.nodes) <= 1:
            return errors

        # Get all root nodes
        root_nodes = dag.get_root_nodes()

        if not root_nodes:
            errors.append("No root nodes (all nodes have parents, possible cycle)")
            return errors

        # Allow multiple root nodes, support parallel execution
        # Only check for single entry point in strict_validation mode
        if len(root_nodes) > 1 and self.strict_validation:
            # No longer enforce single entry point, change to warning
            # errors.append(f"Multiple root nodes: {[n.id for n in root_nodes]}, recommend single entry point")
            pass  # Allow multiple root nodes, support parallel structure

        # BFS from all root nodes to check reachability
        reachable = set()
        queue = deque([node.id for node in root_nodes])

        while queue:
            current_id = queue.popleft()
            if current_id in reachable:
                continue

            reachable.add(current_id)
            current_node = dag.nodes[current_id]

            # Add all child nodes to queue
            for child_id in current_node.children_ids:
                if child_id not in reachable:
                    queue.append(child_id)

        # Check unreachable nodes
        unreachable = set(dag.nodes.keys()) - reachable
        if unreachable:
            errors.append(f"Unreachable nodes: {', '.join(unreachable)}")

        return errors

    def _check_branch_limits(self, dag: QueryDAG) -> List[str]:
        """Check branch limits"""
        errors = []

        for node_id, node in dag.nodes.items():
            # Check branch count limit
            if len(node.children_ids) > self.max_branches:
                errors.append(f"Node {node_id} child count ({len(node.children_ids)}) exceeds limit ({self.max_branches})")

            # Check convergence count limit
            if len(node.parent_ids) > self.max_convergence:
                errors.append(f"Node {node_id} parent count ({len(node.parent_ids)}) exceeds limit ({self.max_convergence})")

        return errors

    def _check_dag_depth(self, dag: QueryDAG) -> List[str]:
        """Check DAG depth"""
        errors = []

        # Calculate max depth
        max_depth = self._calculate_max_depth(dag)

        if max_depth > self.max_dag_depth:
            errors.append(f"DAG depth ({max_depth}) exceeds limit ({self.max_dag_depth})")

        return errors

    def _calculate_max_depth(self, dag: QueryDAG) -> int:
        """Calculate maximum depth of DAG"""
        if not dag.nodes:
            return 0

        # Use DFS to calculate depth of each node
        depths = {}

        def calculate_depth(node_id: str) -> int:
            if node_id in depths:
                return depths[node_id]

            node = dag.nodes[node_id]
            if not node.parent_ids:
                # Root node depth is 1
                depths[node_id] = 1
            else:
                # Node depth = max(parent depths) + 1
                parent_depths = [calculate_depth(pid) for pid in node.parent_ids]
                depths[node_id] = max(parent_depths) + 1

            return depths[node_id]

        # Calculate depth for all nodes
        for node_id in dag.nodes.keys():
            calculate_depth(node_id)

        return max(depths.values()) if depths else 0

    def suggest_fixes(self, dag: QueryDAG, errors: List[str]) -> List[str]:
        """
        Provide fix suggestions based on validation errors

        Args:
            dag: DAG object
            errors: Error list

        Returns:
            Fix suggestion list
        """
        suggestions = []

        for error in errors:
            if "Cycle" in error or "cycle" in error.lower():
                suggestions.append("Suggestion: Remove edges causing cycles, or redesign node dependencies")
            elif "Orphan" in error or "orphan" in error.lower():
                suggestions.append("Suggestion: Connect orphan nodes to main DAG structure, or remove unnecessary nodes")
            elif "Unreachable" in error or "unreachable" in error.lower():
                suggestions.append("Suggestion: Add paths from root nodes to unreachable nodes")
            elif "exceeds limit" in error:
                suggestions.append("Suggestion: Reduce node branch count or simplify DAG structure")
            elif "depth" in error and "exceeds" in error:
                suggestions.append("Suggestion: Reduce DAG levels, change some dependencies to parallel")
            elif "Multiple root" in error:
                suggestions.append("Suggestion: Add a virtual root node to unify entry, or explicitly specify main root node")

        return suggestions


def validate_dag(dag: QueryDAG, strict: bool = True) -> Tuple[bool, List[str], List[str]]:
    """
    Convenience function for DAG validation

    Args:
        dag: DAG to validate
        strict: Whether to use strict mode

    Returns:
        (is_valid, error_list, suggestion_list)
    """
    validator = DAGValidator()
    if not strict:
        validator.strict_validation = False

    is_valid, errors = validator.validate_dag(dag)
    suggestions = validator.suggest_fixes(dag, errors) if errors else []

    return is_valid, errors, suggestions
