"""
DAG Executor
Responsible for executing DAG nodes in order

Core Features:
- Sequential execution of linear DAG
- Each node uses LLM to generate answers
- Final answer synthesis generation
- Basic failure handling (retry)
- Memory passing
- Detailed debug output
- Support for dynamic database type selection via configuration file (MySQL/SQLite)
"""
import time
from typing import List, Dict, Any, Optional
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# add project path for importing existing modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.dag_models import QueryDAG, DAGNode, UnifiedNodeResult, NodeStatus
from src.core.memory import Memory
from src.utils.output_manager import output_manager

# import existing Triple RAG components (reuse)
try:
    from src.core.query_router import QueryRouter
    from src.retrievers.sqlite_retriever import MySQLRetrieverAdapter
    from src.retrievers.neo4j_retriever import Neo4jRetriever
    from src.retrievers.vector_retriever import VectorRetriever
    from src.core.fusion_engine import FusionEngine
    from src.core.output_generator import OutputGenerator
    from src.core.models import TripleRAGResponse
    from config.config import config
except ImportError as e:
    print(f"warning: unable to import existing modules: {e}")
    print("Requires existing Triple RAG components for support")


class DAGExecutor:
    """
    DAG Executor

    Core Features:
    - Sequential execution of linear DAG
    - Basic failure handling (retry)
    - Memory passing
    - LLM answer generation
    - Detailed debug output
    - Dynamic database selection (MySQL/SQLite)
    """

    def __init__(self):
        """initialize executor"""
        # reuse existing components
        self.query_router = QueryRouter()

        # Initialize three retrievers (fixed configuration)
        self.mysql_retriever = MySQLRetrieverAdapter(config)
        self.database_type = 'sqlite'
        print("[DAGExecutor] ✓ Using SQLite database (via MySQLRetrieverAdapter)")

        self.neo4j_retriever = Neo4jRetriever()
        print("[DAGExecutor] ✓ Using Neo4j graph database")

        self.vector_retriever = VectorRetriever()
        self.fusion_engine = FusionEngine()
        self.output_generator = OutputGenerator()

        # execution configuration: load from dag_config.yaml (avoid hardcoding), maintain backward compatibility
        try:
            exec_cfg = getattr(getattr(config, 'dag', None), 'execution', None)
            # overall node retry
            if exec_cfg and getattr(exec_cfg, 'max_retry', None) is not None:
                self.max_retry = int(exec_cfg.max_retry)
            else:
                # compatible with old field
                self.max_retry = int(getattr(config, 'dag_retry_times', 2))

            # failure strategy: stop/skip
            self.failure_strategy = (exec_cfg.failure_strategy if exec_cfg and getattr(exec_cfg, 'failure_strategy', None) is not None else 'stop').lower()

            # channel retry (passed to NodeAgent)
            self.channel_retry_max = int(exec_cfg.channel_retry_max) if exec_cfg and getattr(exec_cfg, 'channel_retry_max', None) is not None else 2
        except Exception:
            self.max_retry = int(getattr(config, 'dag_retry_times', 2))
            self.failure_strategy = 'stop'
            self.channel_retry_max = 2

        # introduce NodeAgent, delegate node execution to this component (minimal refactoring)
        try:
            from src.core.node_agent import NodeAgent
            self.node_agent = NodeAgent(
                query_router=self.query_router,
                mysql_retriever=self.mysql_retriever,
                neo4j_retriever=self.neo4j_retriever,
                vector_retriever=self.vector_retriever,
                fusion_engine=self.fusion_engine,
                output_generator=self.output_generator,
                database_type=self.database_type,
                max_retry=self.max_retry,
                channel_retry_max=self.channel_retry_max,
            )
            print("[DAGExecutor] ✓ NodeAgent initialized and taking over node execution")
        except ImportError as e:
            self.node_agent = None
            print(f"[DAGExecutor] ✗ NodeAgent import failed: {e}")

        # initialize adaptive optimizer
        try:
            from src.core.dag_adaptive_optimizer import DAGAdaptiveOptimizer
            self.optimizer = DAGAdaptiveOptimizer(
                llm_client=self.query_router.client,
                query_router=self.query_router
            )
            print("[DAGExecutor] ✓ DAGAdaptiveOptimizer initialized")
        except ImportError as e:
            self.optimizer = None
            print(f"[DAGExecutor] ✗ DAGAdaptiveOptimizer import failed: {e}")
        except Exception as e:
            self.node_agent = None
            import traceback
            print(f"[DAGExecutor] ✗ NodeAgent initialization failed, using built-in execution logic: {e}")
            print("Detailed error info:")
            traceback.print_exc()

        print(f"[DAGExecutor] Initialization completed - Database: {self.database_type.upper()}")

    def execute_dag(self, dag: QueryDAG, original_query: str) -> TripleRAGResponse:
        """
        Enhanced: execute DAG (supports linear and branching DAG)

        Args:
            dag: QueryDAG object
            original_query: original query

        Returns:
            TripleRAGResponse object
        """
        # log user's original query
        from ..utils.output_manager import log_user_query
        log_user_query(original_query)

        print(f"\n{'='*60}")
        print(f"Starting DAG execution")
        print(f"Original query: {original_query}")
        print(f"Node count: {len(dag.nodes)}")
        print(f"Database type: {self.database_type.upper()}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # Enhanced: unified DAG validation, no longer distinguishing linear and branching
        is_valid, error_msg = dag.validate_structure()
        if not is_valid:
            raise ValueError(f"DAG validation failed: {error_msg}")

        # create Memory
        memory = Memory()

        # mount current DAG to router for NodeAgent to access during context building
        try:
            setattr(self.query_router, 'current_dag', dag)
        except Exception:
            pass

        # Enhanced: unified DAG execution strategy, no longer distinguishing linear and branching
        return self._execute_dag(dag, original_query, memory, start_time)

    def _execute_dag(self, dag: QueryDAG, original_query: str, memory: Memory, start_time: float) -> TripleRAGResponse:
        """
        Enhanced: unified DAG execution (supports any complex structure)
        Enhanced: supports intra-layer parallel, inter-layer sequential execution
        Enhanced: integrated adaptive optimizer
        """
        # get execution plan
        execution_plan = dag.get_execution_plan()
        print(f"DAG execution plan generated")

        # Enhanced: execute by layer in parallel
        rank_groups = execution_plan['rank_groups']
        total_ranks = len(rank_groups)

        # use while loop to support dynamic layer changes (after node insertion)
        current_rank = 0
        while current_rank < len(rank_groups):
            same_rank_nodes = rank_groups[current_rank]

            print(f"\n{'─'*60}")
            print(f"Executing layer {current_rank}: {same_rank_nodes}")
            print(f"{'─'*60}")

            if len(same_rank_nodes) == 1:
                # single node, sequential execution
                self._execute_single_node(dag, same_rank_nodes[0], memory)
            else:
                # multiple nodes, parallel execution
                self._execute_parallel_nodes(dag, same_rank_nodes, memory)

            print(f"✓ Rank {current_rank} execution completed")

            # ===== trigger optimizer check =====
            if self.optimizer and self.optimizer.enabled:
                print(f"\n{'='*60}")
                print(f"[Optimizer] Triggering optimization check...")
                print(f"{'='*60}")

                optimization = self.optimizer.optimize_after_layer(
                    current_rank=current_rank,
                    total_ranks=total_ranks,
                    dag=dag,
                    memory=memory
                )

                # node execution mode marking is deprecated, optimizer no longer marks nodes
                # dynamic switching is determined by NodeAgent during execution

                # Handle final layer node insertion
                if optimization.needs_final_insertion:
                    print(f"\n[Optimization] Need supplementary info node")
                    print(f"  Reason: {optimization.final_insertion_reason}")
                    # Note: Node insertion and execution logic not yet implemented

            current_rank += 1

        # generate final answer
        return self._generate_final_response(dag, original_query, memory, start_time)

    def _execute_branching_dag(self, dag: QueryDAG, original_query: str, memory: Memory, start_time: float) -> TripleRAGResponse:
        """
        Enhanced: execute branching DAG
        """
        # generate execution plan
        execution_plan = dag.get_execution_plan()
        print(f"Branching DAG execution plan:")
        print(f"  Topological order: {' → '.join(execution_plan['topological_order'])}")
        print(f"  Layer grouping: {execution_plan['rank_groups']}")
        print(f"  Can parallelize: {execution_plan['can_parallelize']}")
        print()

        # execute nodes in topological order
        topo_order = execution_plan['topological_order']
        for node_id in topo_order:
            self._execute_single_node(dag, node_id, memory)

        # generate final answer
        return self._generate_final_response(dag, original_query, memory, start_time)

    def _execute_single_node(self, dag: QueryDAG, node_id: str, memory: Memory):
        """
        execute single node (generic method)

        Args:
            dag: DAG object
            node_id: node ID to execute
            memory: memory object
        """
        node = dag.get_node(node_id)
        print(f"{'─'*60}")
        print(f"Executing node: {node_id}")
        print(f"Question: {node.question}")
        if node.parent_ids:
            print(f"Dependencies: {' + '.join(node.parent_ids)}")
        print(f"{'─'*60}")

        try:
            # dependency check: ensure all parent nodes are completed
            completed = dag.get_completed_nodes()
            if not node.can_execute(completed):
                missing = set(node.parent_ids) - completed
                raise RuntimeError(f"Parent nodes not all completed, cannot execute {node_id}: missing {missing}")

            # mark as running for external observation
            node.mark_running()

            # execute node (with retry)
            node_result = self._execute_node_with_retry(dag, node, memory)

            # mark node as completed
            node.mark_completed(
                result=node_result.to_dict(),
                execution_time=node_result.execution_time
            )

            # write to Memory
            memory.write_node_result(node_id, node_result)

            print(f"✓ Node {node_id} execution successful")
            print(f"  Time: {node_result.execution_time:.3f}s")
            if node.parent_ids:
                print(f"  Input sources: {', '.join(node.parent_ids)}")
            print()

        except Exception as e:
            print(f"✗ Node {node_id} execution failed: {e}")
            node.mark_failed(str(e))
            # handle based on failure strategy
            if self.failure_strategy == 'skip':
                print(f"→ Failure strategy: skip (skip subsequent nodes depending on this node)")
                self._mark_dependent_nodes_skipped(dag, node_id)
            else:  # stop
                print(f"→ Failure strategy: stop (stop execution)")
                raise RuntimeError(f"Node {node_id} execution failed, stopping DAG execution: {e}")

    def _mark_dependent_nodes_skipped(self, dag: QueryDAG, failed_node_id: str):
        """
        Enhanced: mark all subsequent nodes depending on failed node as skipped

        Args:
            dag: DAG object
            failed_node_id: failed node ID
        """
        # use BFS to find all subsequent nodes depending on this node
        from collections import deque

        to_skip = set()
        queue = deque([failed_node_id])
        visited = set()

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            current_node = dag.get_node(current_id)
            if current_node:
                for child_id in current_node.children_ids:
                    if child_id not in to_skip:
                        to_skip.add(child_id)
                        queue.append(child_id)

        # mark as skipped
        for node_id in to_skip:
            node = dag.get_node(node_id)
            if node and node.status == NodeStatus.PENDING:
                node.mark_skipped(f"Dependent parent node {failed_node_id} execution failed")
                print(f"  → Skipping node {node_id} (depends on failed {failed_node_id})")

    # ========== Adaptive Optimizer Helper Methods (Simplified) ==========
    # _apply_execution_modes is deprecated, optimizer no longer marks execution mode

    def _generate_final_response(self, dag: QueryDAG, original_query: str, memory: Memory, start_time: float) -> TripleRAGResponse:
        """
        generate final response (generic method)
        """
        print(f"{'='*60}")
        print("Generating final answer...")

        # collect all node results
        all_results = memory.get_all_results()
        all_answers = memory.get_all_answers()

        # generate final answer (using OutputGenerator)
        final_answer = self.output_generator.generate_final_answer(
            original_query, all_answers, all_results
        )

        total_time = time.time() - start_time

        # collect all node detailed info for log summary (including node type info)
        all_node_info = {}
        for node_id, result in all_results.items():
            all_node_info[node_id] = {
                'original_question': getattr(result, 'original_question', 'unknown'),
                'understood_question': getattr(result, 'understood_question', 'unknown'),
                'answer_text': getattr(result, 'answer_text', 'unknown'),
                'execution_time': getattr(result, 'execution_time', 0.0),
                'channels_used': getattr(result, 'channels_used', []),
                # add node type info
                'predicted_node_type': getattr(result, 'predicted_node_type', 'unknown'),
                'actual_node_type': getattr(result, 'actual_node_type', 'unknown'),
                'node_type_switched': getattr(result, 'node_type_switched', False),
                'switch_reason': getattr(result, 'switch_reason', '')
            }

        # generate statistics info
        stats = dag.get_statistics()
        stats.update({
            "total_execution_time": total_time,
            "memory_statistics": memory.get_statistics(),
            "execution_strategy": dag.get_dag_type()
        })

        print(f"✓ DAG execution completed")
        print(f"Total time: {total_time:.3f}s")
        print(f"Successful nodes: {stats['completed_nodes']}/{stats['total_nodes']}")
        print(f"{'='*60}\n")

        try:
            import logging
            try:
                ranks = getattr(dag, 'rank_groups', None) or dag.get_topological_ranks()
                if ranks:
                    last_rank = max(ranks.keys())
                    last_nodes = ranks[last_rank]
                else:
                    last_nodes = [n.id for n in dag.get_leaf_nodes()]
            except Exception:
                last_nodes = [n.id for n in dag.get_leaf_nodes()]
            if last_nodes:
                for nid in last_nodes:
                    res = all_results.get(nid)
                    if res and getattr(res, 'answer_text', None):
                        logging.getLogger("triple_rag").info(f"Last-node {nid} answer: {res.answer_text}")
            logging.getLogger("triple_rag").info(f"Final answer: {final_answer}")
        except Exception:
            pass

        # create empty query_plan and fused_results to satisfy TripleRAGResponse requirements
        from src.core.models import QueryPlan, FusedResult, QueryType

        # create basic query plan
        dummy_query_plan = QueryPlan(
            query_types=[QueryType.MYSQL],  # default use MYSQL
            weights={"mysql": 1.0, "sqlite_graph": 0.0, "vector": 0.0},
            inference="Response generated by DAG execution"
        )

        # collect all nodes' original retrieval contexts (only collect retrieval nodes)
        all_retrieval_contexts = []
        retrieval_node_count = 0
        inference_node_count = 0
        switched_count = 0

        for node_id, result in all_results.items():
            node = dag.get_node(node_id)
            if not node:
                continue

            # count node types (based on actual_node_type)
            if node.actual_node_type == "retrieval":
                retrieval_node_count += 1
                # only collect retrieval contexts from retrieval nodes
                if hasattr(result, 'retrieval_contexts') and result.retrieval_contexts:
                    all_retrieval_contexts.extend(result.retrieval_contexts)
            elif node.actual_node_type == "inference":
                inference_node_count += 1

            # count type switches
            if hasattr(result, 'node_type_switched') and result.node_type_switched:
                switched_count += 1

        # calculate prediction accuracy
        total_nodes = len(all_results)
        prediction_accuracy = 1 - (switched_count / total_nodes) if total_nodes > 0 else 1.0

        output_manager.info(f"[DAGExecutor] 📊 Node type statistics:")
        output_manager.info(f"  - Retrieval nodes: {retrieval_node_count}")
        output_manager.info(f"  - Inference nodes: {inference_node_count}")
        output_manager.info(f"  - Type switches: {switched_count}")
        output_manager.info(f"  - Prediction accuracy: {prediction_accuracy:.2%}")
        output_manager.info(f"  - Collected context count: {len(all_retrieval_contexts)}")

        # log final summary (including statistics info)
        from ..utils.output_manager import log_final_dag_summary
        log_final_dag_summary(
            original_query=original_query,
            all_node_info=all_node_info,
            final_answer=final_answer,
            total_time=total_time,
            retrieval_node_count=retrieval_node_count,
            inference_node_count=inference_node_count,
            switched_count=switched_count,
            prediction_accuracy=prediction_accuracy
        )

        # protection logic: if no retrieval nodes or retrieval nodes have no contexts, fallback to collect all nodes
        if not all_retrieval_contexts:
            output_manager.warning(f"[DAGExecutor] Retrieval nodes have no contexts, falling back to collect all node contexts")
            for node_id, result in all_results.items():
                if hasattr(result, 'retrieval_contexts') and result.retrieval_contexts:
                    all_retrieval_contexts.extend(result.retrieval_contexts)
            output_manager.info(f"[DAGExecutor] After fallback, collected context count: {len(all_retrieval_contexts)}")

        # create basic fused result
        success_count = len([r for r in all_results.values() if r.answer_text and r.answer_text.strip()])
        dummy_fused_results = [FusedResult(
            content=final_answer,
            sources=list(all_results.keys()),
            confidence=0.85,
            metadata_summary=f"DAG execution result, successful nodes: {success_count}/{len(all_results)}"
        )]

        return TripleRAGResponse(
            query=original_query,
            answer=final_answer,
            fused_results=dummy_fused_results,
            query_plan=dummy_query_plan,
            total_execution_time=total_time,
            explanation=f"DAG execution completed, processed {len(dag.nodes)} nodes, {success_count} successful",
            final_answer=final_answer,
            retrieved_contexts=all_retrieval_contexts  # add original retrieval contexts
        )

    def _execute_node_with_retry(self, dag: QueryDAG, node: DAGNode, memory: Memory) -> UnifiedNodeResult:
        """execute single node (with retry mechanism) - delegated to NodeAgent"""
        if not self.node_agent:
            # explicitly fail to trigger upper layer TripleRAG's old system fallback
            raise RuntimeError("NodeAgent unavailable, cannot execute node.")
        return self.node_agent.execute_node_with_retry(node, memory, dag=dag)

    def _execute_parallel_nodes(self, dag: QueryDAG, node_ids: List[str], memory: Memory):
        """
        Enhanced: execute multiple nodes at the same layer in parallel

        Args:
            dag: DAG object
            node_ids: list of node IDs to execute in parallel
            memory: memory object (thread-safe)
        """
        print(f"Starting parallel execution of {len(node_ids)} nodes...")

        # check if using SQLite database, if yes, create thread-safe retriever copies
        is_sqlite = self.database_type.lower() == 'sqlite'

        if is_sqlite:
            # for SQLite, we need to create independent retriever instances for each thread
            # this avoids thread safety issues
            print("Detected SQLite database, using thread-safe parallel execution strategy")

            # create independent execution function for each node, ensuring each thread uses independent SQLite connection
            def execute_node_with_thread_safe_retriever(node_id):
                # create independent SQLite retriever instance for each thread
                from src.retrievers.sqlite_retriever import MySQLRetrieverAdapter
                from src.core.node_agent import NodeAgent  # ensure NodeAgent is imported

                try:
                    # create independent SQLite retriever instance, each thread uses its own connection
                    thread_safe_retriever = MySQLRetrieverAdapter()

                    # create temporary NodeAgent instance, using thread-safe retriever
                    temp_node_agent = NodeAgent(
                        query_router=self.query_router,
                        mysql_retriever=thread_safe_retriever,
                        neo4j_retriever=self.neo4j_retriever,
                        vector_retriever=self.vector_retriever,
                        fusion_engine=self.fusion_engine,
                        output_generator=self.output_generator,
                        database_type=self.database_type,
                        max_retry=self.max_retry,
                        channel_retry_max=self.channel_retry_max,
                    )

                    # execute node
                    node = dag.get_node(node_id)
                    print(f"  → Thread {threading.current_thread().name} starting execution of node {node_id}")
                    result = temp_node_agent.execute_node_with_retry(node, memory, dag=dag)

                    # mark node as completed
                    node.mark_completed(
                        result=result.to_dict(),
                        execution_time=result.execution_time
                    )

                    # write to Memory
                    memory.write_node_result(node_id, result)

                    print(f"  ✓ Node {node_id} execution successful in thread {threading.current_thread().name}")
                    return node_id, None  # return node ID and error info
                except Exception as e:
                    print(f"  ✗ Node {node_id} execution failed in thread {threading.current_thread().name}: {e}")
                    return node_id, str(e)  # return node ID and error info
                finally:
                    # ensure closing thread-local SQLite connection
                    if 'thread_safe_retriever' in locals():
                        try:
                            thread_safe_retriever.retriever.thread_safe_retriever.close()
                        except Exception as close_error:
                            print(f"  Error closing SQLite connection for thread {threading.current_thread().name}: {close_error}")

            # use thread pool for parallel execution, limit max concurrency to avoid resource exhaustion
            max_workers = min(len(node_ids), 4)  # max 4 concurrent

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # submit all node execution tasks
                future_to_node = {
                    executor.submit(execute_node_with_thread_safe_retriever, node_id): node_id
                    for node_id in node_ids
                }

                # wait for all tasks to complete
                completed_count = 0
                failed_count = 0

                for future in as_completed(future_to_node):
                    node_id, error = future.result()
                    if error is None:
                        completed_count += 1
                        print(f"  ✓ Node {node_id} parallel execution completed ({completed_count}/{len(node_ids)})")
                    else:
                        failed_count += 1
                        print(f"  ✗ Node {node_id} parallel execution failed: {error}")

                        # mark node as failed
                        node = dag.get_node(node_id)
                        node.mark_failed(error)

                        # handle based on failure strategy
                        if self.failure_strategy == 'stop':
                            # cancel other executing tasks
                            for f in future_to_node:
                                f.cancel()
                            raise RuntimeError(f"Node {node_id} execution failed, stopping parallel execution: {error}")
                        else:  # skip
                            print(f"  → Failure strategy: skip, continuing execution of other nodes")

                print(f"Parallel execution completed: {completed_count} successful, {failed_count} failed")

                # if all nodes failed, raise exception
                if failed_count == len(node_ids):
                    raise RuntimeError(f"All {len(node_ids)} parallel nodes failed execution")
        else:
            # non-SQLite database, use original parallel execution logic
            print("Using standard parallel execution strategy")
            max_workers = min(len(node_ids), 4)  # max 4 concurrent

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # submit all node execution tasks
                future_to_node = {
                    executor.submit(self._execute_single_node_with_error_handler, dag, node_id, memory): node_id
                    for node_id in node_ids
                }

                # wait for all tasks to complete
                completed_count = 0
                failed_count = 0

                for future in as_completed(future_to_node):
                    node_id = future_to_node[future]
                    try:
                        future.result()  # this will raise exception if any
                        completed_count += 1
                        print(f"  ✓ Node {node_id} parallel execution completed ({completed_count}/{len(node_ids)})")
                    except Exception as e:
                        failed_count += 1
                        print(f"  ✗ Node {node_id} parallel execution failed: {e}")

                        # handle based on failure strategy
                        if self.failure_strategy == 'stop':
                            # cancel other executing tasks
                            for f in future_to_node:
                                f.cancel()
                            raise RuntimeError(f"Node {node_id} execution failed, stopping parallel execution: {e}")
                        else:  # skip
                            print(f"  → Failure strategy: skip, continuing execution of other nodes")

                print(f"Parallel execution completed: {completed_count} successful, {failed_count} failed")

                # if all nodes failed, raise exception
                if failed_count == len(node_ids):
                    raise RuntimeError(f"All {len(node_ids)} parallel nodes failed execution")

    def _execute_single_node_with_error_handler(self, dag: QueryDAG, node_id: str, memory: Memory):
        """
        error handling wrapper for executing single node
        used to isolate errors during parallel execution
        """
        try:
            self._execute_single_node(dag, node_id, memory)
        except Exception as e:
            # log error but don't raise, let upper parallel executor handle uniformly
            print(f"Node {node_id} execution error: {e}")
            raise  # re-raise for upper layer handling