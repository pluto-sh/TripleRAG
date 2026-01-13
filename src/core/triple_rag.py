"""
Triple RAG Main System
Integrates query understanding, multi-channel retrieval, fusion engine, and output generator
"""
import time
from typing import List, Dict, Any, Optional
import logging
from .models import TripleRAGResponse, QueryPlan, RetrievalResult, FusedResult, QueryType
from .query_router import QueryRouter
from ..retrievers.sqlite_retriever import MySQLRetrieverAdapter
from ..retrievers.neo4j_retriever import Neo4jRetriever
from ..retrievers.vector_retriever import VectorRetriever
from .fusion_engine import FusionEngine
from .output_generator import OutputGenerator
from config.config import config

class TripleRAG:
    """Triple RAG Main System"""

    def __init__(self):
        """Initialize system components"""
        self.query_router = QueryRouter()

        # LogicRAG mode: only initialize vector retriever
        if hasattr(config, 'logicrag_mode') and config.logicrag_mode:
            print("\n" + "="*60)
            print("LogicRAG mode initialization")
            print("="*60)
            self.mysql_retriever = None
            self.neo4j_retriever = None
            self.vector_retriever = VectorRetriever()
            print("Vector retriever initialized")
            print("SQL/Graph retrievers disabled")
            print("="*60 + "\n")
        else:
            # Initialize three retrievers
            self.mysql_retriever = MySQLRetrieverAdapter(config)
            print("Using SQLite database retriever")

            self.neo4j_retriever = Neo4jRetriever()
            print("Using Neo4j graph database retriever")

            self.vector_retriever = VectorRetriever()
            print("Using vector database retriever")

        self.fusion_engine = FusionEngine()
        self.output_generator = OutputGenerator()

        # ========== DAG Executor Initialization ==========
        from src.core.dag_executor import DAGExecutor
        self.dag_executor = DAGExecutor()
        print("Triple RAG system DAG mode enabled")

        print("Triple RAG system initialization complete")

    def query(self, user_query: str) -> TripleRAGResponse:
        """
        Process user query (DAG mode)

        Args:
            user_query: User query

        Returns:
            TripleRAGResponse: System response
        """
        # Create new log file for each query
        from ..utils.output_manager import setup_logging, get_current_log_file
        from config.config import config
        import datetime
        import os

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = config.paths.get_absolute_path(config.paths.logs_dir)
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"triple_rag_{timestamp}.log")
        setup_logging(log_file)

        print(f"\nProcessing query: {user_query}")
        print(f"Log file: {log_file}")
        logging.getLogger("triple_rag").info(f"User query: {user_query}")
        return self._query_with_dag(user_query)

    def _query_with_dag(self, user_query: str) -> TripleRAGResponse:
        """
        DAG query flow

        Args:
            user_query: User query

        Returns:
            TripleRAGResponse: System response
        """
        start_time = time.time()

        try:
            print("  Step 1: Decomposing query to DAG...")

            # Step 1: Decompose query to DAG (let LLM freely design structure)
            dag = self.query_router.decompose_query_to_dag(user_query)
            if dag is None:
                # Safe fallback: use single-node DAG (directly answer original question)
                from src.core.dag_models import create_linear_dag
                dag = create_linear_dag(user_query, [{"id": "p1", "question": user_query, "reason": "Parsing failed, using single-node fallback"}])
                print("  DAG parsing failed, fallback to single-node DAG")

            print(f"  Decomposition complete: {len(dag.nodes)} nodes")

            # Validate DAG structure (supports any complex structure)
            try:
                # Display DAG basic info
                print(f"  DAG node count: {len(dag.nodes)}")
                print(f"  DAG edge count: {len(dag.edges)}")

                # Display topological execution order
                topo_order = dag.topological_sort()
                print(f"  Topological execution order: {' → '.join(topo_order)}")

            except Exception as e:
                print(f"  DAG structure validation failed ({e}), fallback to single node")
                from src.core.dag_models import create_linear_dag
                dag = create_linear_dag(user_query, [{"id": "p1", "question": user_query, "reason": "Structure validation failed, fallback to single node"}])

            # Step 2: Execute DAG
            print("\n  Step 2: Executing DAG...")
            response = self.dag_executor.execute_dag(dag, user_query)

            return response

        except Exception as e:
            # If DAG execution fails, return error
            total_time = time.time() - start_time
            print(f"DAG execution failed: {e}")
            import traceback
            traceback.print_exc()

            error_plan = QueryPlan(
                query_types=[],
                inference=f"DAG execution failed: {str(e)}"
            )

            error_response = TripleRAGResponse(
                query=user_query,
                answer=f"Sorry, an error occurred while processing your query: {str(e)}",
                fused_results=[],
                query_plan=error_plan,
                total_execution_time=total_time,
                explanation=f"System error, execution time: {total_time:.3f}s"
            )

            return error_response

        finally:
            # Reset logging state after query completion, allow next query to create new log file
            try:
                from ..utils.output_manager import reset_logging
                reset_logging()
            except Exception:
                pass

    def get_formatted_response(self, user_query: str, simple_format: bool = False) -> str:
        """
        Get formatted response text

        Args:
            user_query: User query
            simple_format: Whether to use simple format

        Returns:
            str: Formatted response text
        """
        response = self.query(user_query)

        if simple_format:
            return self.output_generator.format_simple_response(response)
        else:
            return self.output_generator.format_detailed_response(response)

    def get_json_response(self, user_query: str) -> Dict[str, Any]:
        """
        Get JSON format response

        Args:
            user_query: User query

        Returns:
            Dict[str, Any]: JSON format response
        """
        response = self.query(user_query)
        return self.output_generator.export_to_json(response)

    def initialize_sample_data(self):
        """Initialize sample data"""
        print("Initializing sample data...")

        try:
            # Initialize vector database samples
            self.vector_retriever.load_sample_documents()
            print("Vector database sample data loaded")

        except Exception as e:
            print(f"Sample data initialization failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        status = {
            "mode": "dag",
            "mysql_connected": False,
            "sqlite_graph_connected": False,
            "vector_db_ready": False,
            "total_documents": 0,
            "dag_executor_available": self.dag_executor is not None
        }

        try:
            # Check MySQL connection
            mysql_info = self.mysql_retriever.get_database_info()
            status["mysql_connected"] = len(mysql_info) > 0
        except:
            pass

        try:
            # Check graph database connection
            graph_info = self.neo4j_retriever.get_graph_schema()
            status["graph_connected"] = len(graph_info) > 0
        except:
            pass

        try:
            # Check vector database
            vector_info = self.vector_retriever.get_collection_info()
            status["vector_db_ready"] = True
            status["total_documents"] = vector_info.get("count", 0)
        except:
            pass

        return status

    def close(self):
        """Close system connections"""
        try:
            self.mysql_retriever.close()
            print("MySQL connection closed")
        except:
            pass

        try:
            self.neo4j_retriever.close()
            print("Graph database connection closed")
        except:
            pass

        print("Triple RAG system closed")

# Convenience functions
def create_triple_rag() -> TripleRAG:
    """Create Triple RAG instance"""
    return TripleRAG()

def quick_query(query: str, simple_format: bool = True) -> str:
    """Quick query function"""
    rag = create_triple_rag()
    try:
        result = rag.get_formatted_response(query, simple_format)
        return result
    finally:
        rag.close()
