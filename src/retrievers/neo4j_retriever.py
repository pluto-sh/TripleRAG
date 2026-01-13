"""
Neo4j Knowledge Graph Retrieval Channel
Constructs knowledge graph queries based on user queries, leverages graph database to query knowledge graph, and returns structured results
"""
from neo4j import GraphDatabase
import time
import json
import os
from typing import List, Dict, Any, Optional
from ..core.models import RetrievalResult, RetrievalMetadata
from config.config import config
from openai import OpenAI
from ..utils.output_manager import output_manager

class Neo4jRetriever:
    """Neo4j Retriever"""

    def __init__(self):
        self.driver = None
        # Runtime hints (e.g., edge constraint dispatch inputs) for execution phase reference
        self._runtime_hints: Dict[str, Any] = {}

        # Initialize LLM client for intelligent query generation
        self.llm_client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key
        )

        self.connect()

    def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                config.neo4j.uri,
                auth=(config.neo4j.user, config.neo4j.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            output_manager.debug("Neo4j connection successful")
        except Exception as e:
            output_manager.error(f"Neo4j connection failed: {e}")
            self.driver = None
    
    def execute_query(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[RetrievalResult]:
        """Execute Cypher query and return results"""
        if not self.driver:
            output_manager.warning("Neo4j connection unavailable")
            return []

        start_time = time.time()
        results = []

        try:
            # Log output: Actual Cypher statement executed
            output_manager.debug(f"[CYPHER Execution] {cypher_query}")

            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                records = list(result)

                execution_time = time.time() - start_time

                # Create retrieval metadata
                metadata = RetrievalMetadata(
                    query_statement=cypher_query,
                    execution_time=execution_time,
                    result_count=len(records),
                    source_type="neo4j"
                )

                # Convert results to RetrievalResult format
                for i, record in enumerate(records):
                    content = self._format_record_content(record)

                    result_obj = RetrievalResult(
                        content=content,
                        score=1.0,  # Neo4j results default score is 1.0
                        metadata=metadata,
                        source_id=f"neo4j_record_{i}",
                        source="neo4j"
                    )
                    results.append(result_obj)

                output_manager.debug(f"Neo4j query executed successfully, returned {len(results)} results, took {execution_time:.3f} seconds")

        except Exception as e:
            output_manager.error(f"Neo4j query execution failed: {e}")
            execution_time = time.time() - start_time

            # Record metadata even if failed
            metadata = RetrievalMetadata(
                query_statement=cypher_query,
                execution_time=execution_time,
                result_count=0,
                source_type="neo4j"
            )

            # Return error information as result
            error_result = RetrievalResult(
                content=f"Neo4j query execution failed: {str(e)}",
                score=0.0,
                metadata=metadata,
                source_id="neo4j_error",
                source="neo4j"
            )
            results.append(error_result)

        return results

    def set_runtime_hints(self, hints: Dict[str, Any]):
        """Set runtime hints (such as edge_inputs), can be used for parameterization or log display.
        Currently does not directly rewrite Cypher, only records for future extension.
        """
        try:
            if isinstance(hints, dict):
                self._runtime_hints = hints
        except Exception:
            self._runtime_hints = {}
    
    def _format_record_content(self, record) -> str:
        """Format record content as text"""
        content_parts = []

        for key, value in record.items():
            if hasattr(value, 'labels') and hasattr(value, 'items'):
                # This is a node
                labels = list(value.labels)
                properties = dict(value.items())
                content_parts.append(f"Node({':'.join(labels)}): {properties}")
            elif hasattr(value, 'type') and hasattr(value, 'items'):
                # This is a relationship
                rel_type = value.type
                properties = dict(value.items())
                content_parts.append(f"Relationship[{rel_type}]: {properties}")
            else:
                # Regular value
                content_parts.append(f"{key}: {value}")

        return " | ".join(content_parts)

    def search_entities(self, entity_name: str, limit: int = None) -> List[RetrievalResult]:
        """Search entities"""
        if not limit:
            limit = config.max_neo4j_results

        cypher_query = """
        MATCH (n)
        WHERE toLower(toString(n.name)) CONTAINS toLower($entity_name)
           OR toLower(toString(n.title)) CONTAINS toLower($entity_name)
        RETURN n, labels(n) as node_labels
        LIMIT $limit
        """

        parameters = {
            "entity_name": entity_name,
            "limit": limit
        }

        return self.execute_query(cypher_query, parameters)
    
    def search_relationships(self, entity1: str, entity2: str = None, relationship_type: str = None, limit: int = None) -> List[RetrievalResult]:
        """Search relationships"""
        if not limit:
            limit = config.max_neo4j_results

        if entity2:
            # Search relationships between two entities
            cypher_query = """
            MATCH (a)-[r]-(b)
            WHERE (toLower(toString(a.name)) CONTAINS toLower($entity1) OR toLower(toString(a.title)) CONTAINS toLower($entity1))
              AND (toLower(toString(b.name)) CONTAINS toLower($entity2) OR toLower(toString(b.title)) CONTAINS toLower($entity2))
            """
            if relationship_type:
                cypher_query += f" AND type(r) = $relationship_type"

            cypher_query += """
            RETURN a, r, b, type(r) as rel_type
            LIMIT $limit
            """

            parameters = {
                "entity1": entity1,
                "entity2": entity2,
                "limit": limit
            }
            if relationship_type:
                parameters["relationship_type"] = relationship_type
        else:
            # Search all relationships of one entity
            cypher_query = """
            MATCH (a)-[r]-(b)
            WHERE toLower(toString(a.name)) CONTAINS toLower($entity1)
               OR toLower(toString(a.title)) CONTAINS toLower($entity1)
            """
            if relationship_type:
                cypher_query += f" AND type(r) = $relationship_type"

            cypher_query += """
            RETURN a, r, b, type(r) as rel_type
            LIMIT $limit
            """

            parameters = {
                "entity1": entity1,
                "limit": limit
            }
            if relationship_type:
                parameters["relationship_type"] = relationship_type

        return self.execute_query(cypher_query, parameters)

    def search_paths(self, start_entity: str, end_entity: str, max_depth: int = 3, limit: int = None) -> List[RetrievalResult]:
        """Search paths"""
        if not limit:
            limit = config.max_neo4j_results

        cypher_query = f"""
        MATCH path = (start)-[*1..{max_depth}]-(end)
        WHERE (toLower(toString(start.name)) CONTAINS toLower($start_entity) OR toLower(toString(start.title)) CONTAINS toLower($start_entity))
          AND (toLower(toString(end.name)) CONTAINS toLower($end_entity) OR toLower(toString(end.title)) CONTAINS toLower($end_entity))
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT $limit
        """

        parameters = {
            "start_entity": start_entity,
            "end_entity": end_entity,
            "limit": limit
        }

        return self.execute_query(cypher_query, parameters)
    
    def get_graph_schema(self) -> Dict[str, Any]:
        """Get graph database schema information"""
        if not self.driver:
            return {}

        schema_info = {}

        try:
            with self.driver.session() as session:
                # Get node labels
                result = session.run("CALL db.labels()")
                labels = [record["label"] for record in result]
                schema_info["node_labels"] = labels

                # Get relationship types
                result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in result]
                schema_info["relationship_types"] = rel_types

                # Get property keys
                result = session.run("CALL db.propertyKeys()")
                prop_keys = [record["propertyKey"] for record in result]
                schema_info["property_keys"] = prop_keys

                # Get node and relationship counts
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()["node_count"]
                schema_info["node_count"] = node_count

                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                schema_info["relationship_count"] = rel_count

        except Exception as e:
            output_manager.error(f"Failed to get graph schema information: {e}")

        return schema_info


    def get_graph_schema_summary(self) -> str:
        """Get graph database schema summary information"""
        try:
            # Read schema from config file
            schema_path = config.paths.get_absolute_path(config.paths.graph_schema_path)
            if os.path.exists(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Dynamically get schema information
                schema_info = self.get_graph_schema()
                summary = f"""
Node labels: {', '.join(schema_info.get('node_labels', []))}
Relationship types: {', '.join(schema_info.get('relationship_types', []))}
Node count: {schema_info.get('node_count', 0)}
Relationship count: {schema_info.get('relationship_count', 0)}
"""
                return summary
        except Exception as e:
            output_manager.error(f"Failed to get graph schema summary: {e}")
            return "Graph database schema information unavailable"

    def generate_query_or_placeholder(self, question: str, parent_context: str = "") -> str:
        """
        Use LLM to determine if question involves relationship query, and generate Cypher query or return placeholder

        Args:
            question: User question
            parent_context: Parent node context (optional)

        Returns:
            Cypher query statement or placeholder identifier
        """
        if not self.llm_client:
            output_manager.warning("LLM client not initialized, returning placeholder")
            return "GRAPH_PLACEHOLDER_NO_LLM_CLIENT"

        try:
            # Get graph schema information
            schema_info = self.get_graph_schema_summary()

            prompt = f"""You are a Neo4j Cypher query generation expert. Please determine whether a graph database query is needed based on the question, and if so, generate an appropriate Cypher query statement.

Question: {question}
Parent node context: {parent_context}
Graph data schema: {schema_info}

Tasks:
1. **First determine if the question is related to the graph database**:
   - If the question can be answered by querying graph database related information, generate a Cypher query
   - If the question is clearly unrelated to the data in the graph database, please return: GRAPH_PLACEHOLDER_NOT_RELEVANT
   - If uncertain whether the question is related to the graph database, tend to generate a Cypher query

2. **If you need to generate a Cypher query**:
   - Must only use node and relationship types that actually exist in the database
   - If the parent node context contains specific information, please use these specific values directly in the Cypher query to replace template placeholders (such as {{{{p1.answer}}}})
   - Add appropriate LIMIT 5 to limit the number of results

Output format: Directly output the Cypher statement or GRAPH_PLACEHOLDER_NOT_RELEVANT, do not output any additional information.
"""

            response = self.llm_client.chat.completions.create(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = response.choices[0].message.content.strip()

            # Clean possible code block markers
            if result.startswith("```cypher"):
                result = result[9:]
            elif result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            return result

        except Exception as e:
            output_manager.error(f"Cypher query generation failed: {e}")
            return "GRAPH_PLACEHOLDER_GENERATION_ERROR"

    def execute_query_with_retry(self, query_or_placeholder: str, parent_context: str = "", result_limit: int = 5) -> List[RetrievalResult]:
        """
        Execute Cypher query, supports retry mechanism

        Args:
            query_or_placeholder: Cypher query statement or placeholder
            parent_context: Parent node context (for providing more information during retry)
            result_limit: Result count limit

        Returns:
            List of retrieval results
        """
        # Check if it's a placeholder
        if query_or_placeholder.startswith("GRAPH_PLACEHOLDER"):
            output_manager.info(f"🕸️ Graph channel intelligent skip: {query_or_placeholder}")
            return self._create_placeholder_result(query_or_placeholder)

        try:
            # First attempt: Execute query
            # Add LIMIT restriction
            if 'LIMIT' not in query_or_placeholder.upper():
                query_or_placeholder = f"{query_or_placeholder.rstrip(';')} LIMIT {result_limit}"

            results = self.execute_query(query_or_placeholder)

            # Check if result contains error
            if results and len(results) == 1 and "Neo4j query execution failed" in results[0].content:
                raise Exception(results[0].content)

            return results

        except Exception as e:
            output_manager.error(f"❌ Cypher query first execution failed: {e}")

            # Check retry configuration
            try:
                retry_enabled = bool(getattr(getattr(config, 'dag', None), 'execution', None).channel_retry_enabled)
            except Exception:
                retry_enabled = True

            if not retry_enabled:
                return self._create_placeholder_result("GRAPH_PLACEHOLDER_EXECUTION_FAILED_NO_RETRY")

            # Second attempt: Pass in error hint
            enhanced_context = f"{parent_context}\n\n[Error hint] Cypher query execution error: {str(e)}"

            try:
                fixed_query = self.generate_query_or_placeholder(
                    query_or_placeholder,  # Use original query as question context
                    enhanced_context
                )

                if fixed_query.startswith("GRAPH_PLACEHOLDER"):
                    output_manager.info(f"🕸️ Graph channel retry abandoned: {fixed_query}")
                    return self._create_placeholder_result(fixed_query)

                # Add LIMIT restriction
                if 'LIMIT' not in fixed_query.upper():
                    fixed_query = f"{fixed_query.rstrip(';')} LIMIT {result_limit}"

                results = self.execute_query(fixed_query)
                output_manager.info(f"✅ Cypher query second attempt successful")
                return results

            except Exception as retry_error:
                output_manager.error(f"❌ Cypher query second attempt also failed: {retry_error}")
                return self._create_placeholder_result("GRAPH_PLACEHOLDER_EXECUTION_FAILED")

    def _create_placeholder_result(self, placeholder_message: str) -> List[RetrievalResult]:
        """Create placeholder result"""
        placeholder_result = RetrievalResult(
            content=placeholder_message,  # Placeholder identifier
            score=0.0,
            metadata=RetrievalMetadata(
                query_statement="",
                execution_time=0.001,
                result_count=0,
                source_type="neo4j"
            ),
            source_id="placeholder",
            source="neo4j"
        )
        return [placeholder_result]

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            output_manager.debug("Neo4j connection closed")