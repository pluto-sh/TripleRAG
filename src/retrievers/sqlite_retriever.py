"""
SQLite Retriever
Performs supply chain data retrieval using SQLite database
"""
import json
import sqlite3
import time
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from ..core.models import RetrievalResult, RetrievalMetadata
from config.config import config

from ..utils.thread_safe_sqlite import ThreadSafeSQLiteRetriever
from openai import OpenAI
from ..utils.output_manager import output_manager

SQLITE_DB_PATH = config.sqlite.database_path

class SQLiteConfig:
    """SQLite Configuration Class"""
    def __init__(self, database_path: str = SQLITE_DB_PATH, max_results: int = 50):
        self.database_path = database_path
        self.max_results = max_results

class SQLiteRetriever:
    """SQLite Database Retriever (Thread-Safe Version)"""

    def __init__(self, config: SQLiteConfig):
        self.config = config
        # Use thread-safe SQLite retriever
        self.thread_safe_retriever = ThreadSafeSQLiteRetriever(config.database_path)

    def connect(self):
        """Connect to SQLite database (thread-safe version does not require explicit connection)"""
        # Thread-safe retriever automatically manages connections
        pass

    def close(self):
        """Close database connection (thread-safe version does not require explicit closing)"""
        # Thread-safe retriever automatically manages connections
        pass

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _format_results(self, results: List[Dict[str, Any]], query: str, search_type: str = "keyword") -> List[RetrievalResult]:
        """Format query results"""
        formatted_results = []

        for i, row in enumerate(results):
            # row is already in dictionary form
            row_dict = row

            # Create content text
            content_parts = []
            for key, value in row_dict.items():
                if value is not None and str(value).strip():
                    content_parts.append(f"{key}: {value}")

            content = "\n".join(content_parts)

            # Create metadata - use correct RetrievalMetadata structure
            metadata = RetrievalMetadata(
                query_statement=query,
                execution_time=0.0,  
                result_count=len(results),
                source_type="mysql"  
            )

            result = RetrievalResult(
                content=content,
                score=1.0 - (i * 0.01),  
                metadata=metadata,
                source_id=f"sqlite_{search_type}_{i}",
                source="mysql"  
            )

            formatted_results.append(result)

        return formatted_results

    def execute_sql(self, sql_query: str, limit: int = None) -> List[RetrievalResult]:
        """Execute raw SQL query (core functionality, let LLM directly output SQL query)"""
        if limit is None:
            limit = self.config.max_results

        try:
            # Add LIMIT restriction (if not in SQL)
            if 'LIMIT' not in sql_query.upper():
                sql_query = f"{sql_query.rstrip(';')} LIMIT {limit}"

            # Log output: actual SQL statement executed
            output_manager.debug(f"[SQLExecute] {sql_query}")

            # Use thread-safe retriever to execute query
            results = self.thread_safe_retriever.execute_sql(sql_query)
            
            # Use query description as search keyword
            query_desc = sql_query[:50].replace('\n', ' ')
            return self._format_results(results, query_desc, "direct_sql")
            
        except Exception as e:
            output_manager.error(f"SQL query execution failed: {e}")
            # Return structured error result (transparent), no keyword fallback
            metadata = RetrievalMetadata(
                query_statement=sql_query,
                execution_time=0.0,
                result_count=0,
                source_type="mysql"
            )
            error_result = RetrievalResult(
                content=f"SQL execution error: {str(e)}",
                score=0.0,
                metadata=metadata,
                source_id="sqlite_sql_error",
                source="mysql"
            )
            return [error_result]
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        try:
            # Use thread-safe retriever to get database info
            info = self.thread_safe_retriever.get_database_info()
            
            # Get material type statistics from config file instead of querying database
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'sql_schema_hotpotqa.md')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    schema_content = f.read()
                
                # Simple parsing of material type information
                material_types = []
                if 'material_types' in schema_content:
                    lines = schema_content.split('\n')
                    in_material_types = False
                    for line in lines:
                        if 'material_types' in line and ('table' in line or 'CREATE TABLE' in line):
                            in_material_types = True
                        elif in_material_types and line.strip().startswith('material_name'):
                            # Extract material type information
                            material_types.append(line.strip())
                        elif in_material_types and line.strip().startswith(')'):
                            break
                
                info['material_statistics'] = {mt.split('(')[0].strip(): 0 for mt in material_types}
            else:
                info['material_statistics'] = {}
            
            return info
            
        except Exception as e:
            output_manager.error(f"Failed to get database info: {e}")
            return {}
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables in database"""
        try:
            # Use thread-safe retriever to get database info
            info = self.thread_safe_retriever.get_database_info()
            return info.get('tables', [])
        except Exception as e:
            output_manager.error(f"Failed to get table list: {e}")
            return []


# Create adapter class for compatibility
class MySQLRetrieverAdapter:
    """MySQL Retriever Adapter, wraps SQLite retriever as MySQL retriever interface (thread-safe version)"""

    def __init__(self, config_obj=None):
        # Use SQLite database path from config
        sqlite_config = SQLiteConfig(
            database_path=config.sqlite.database_path,
            max_results=getattr(config_obj, 'max_mysql_results', 50) if config_obj else 50
        )
        self.retriever = SQLiteRetriever(sqlite_config)

        # Initialize LLM client for intelligent judgment and SQL generation
        self.llm_client = OpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key
        )
    
    def search(self, query: str, limit: int = None) -> List[RetrievalResult]:
        """General search method"""
        return self.retriever.execute_sql(query, limit)

    def get_table_schema_summary(self) -> str:
        """Get database table schema summary information"""
        try:
            # Read schema from config file
            schema_path = config.paths.get_absolute_path(config.paths.sql_schema_path)
            if os.path.exists(schema_path):
                with open(schema_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # If config file doesn't exist, return basic info
                return """
                Table Overview: companies, production_facilities, annual_production

                Table Fields:
                - companies: id, company_name, country_id, industry, company_type, website
                - production_facilities: id, facility_name, facility_type, status, material_type_id, city, state_province, country_id, operator_company_id, primary_owner_company_id, product_type, feedstock
                - annual_production: id, facility_id, year, production_volume, is_projected
                """
        except Exception as e:
            output_manager.error(f"Failed to get table schema summary: {e}")
            return "Database table structure information unavailable"

    def generate_sql_or_placeholder(self, question: str, parent_context: str = "") -> str:
        """
        Use LLM to determine if question is suitable for SQL query, and generate SQL or return placeholder

        Args:
            question: User question
            parent_context: Parent node context (optional)

        Returns:
            SQL query statement or placeholder identifier
        """
        if not self.llm_client:
            output_manager.warning("LLM client not initialized, returning placeholder")
            return "SQL_PLACEHOLDER_NO_LLM_CLIENT"

        try:
            # Get database table structure information
            schema_info = self.get_table_schema_summary()

            prompt = f"""You are a SQL query generation expert. Please determine whether a SQL query is needed based on the question, and if so, generate an appropriate SQL query statement.

Question: {question}
Parent node context: {parent_context}
Database table structure: {schema_info}

Tasks:
1. **First determine if the question is related to the SQL database**:
   - If the question can be answered by querying SQL database related information, generate a SQL query
   - If the question is clearly unrelated to the data in the SQL database, please return: SQL_PLACEHOLDER_NOT_RELEVANT
   - If uncertain whether the question is related to the SQL database, tend to generate a SQL query

2. **If you need to generate a SQL query**:
   - Must only use table names and fields that actually exist in the database
   - If the parent node context contains specific information, please use these specific values directly in the SQL query to replace template placeholders (such as {{{{p1.answer}}}})
   - Add appropriate LIMIT 5 to limit the number of results

Output format: Directly output the SQL statement or SQL_PLACEHOLDER_NOT_RELEVANT, do not output any additional information.
"""

            # Call LLM to generate SQL 
            response = self.llm_client.chat.completions.create(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )

            result = response.choices[0].message.content.strip()

            # Clean possible code block markers
            if result.startswith("```sql"):
                result = result[6:]
            if result.startswith("```"):
                result = result[3:]
            if result.endswith("```"):
                result = result[:-3]
            result = result.strip()

            return result

        except Exception as e:
            output_manager.error(f"SQLGeneration failed: {e}")
            return "SQL_PLACEHOLDER_GENERATION_ERROR"

    def execute_sql_with_retry(self, sql_or_placeholder: str, parent_context: str = "", result_limit: int = 5) -> List[RetrievalResult]:
        """
        ExecuteSQLquery，Support retry mechanism

        Args:
            sql_or_placeholder: SQLQuery statement or placeholder
            parent_context: Parent node context
            result_limit: result count limit

        Returns:
            Retrieval result list
        """
        # Check if it's a placeholder
        if sql_or_placeholder.startswith("SQL_PLACEHOLDER"):
            output_manager.info(f"📋 SQLchannel intelligent skip: {sql_or_placeholder}")
            return self._create_placeholder_result(sql_or_placeholder)

        try:
            # First attempt：ExecuteSQLquery
            results = self.retriever.execute_sql(sql_or_placeholder, result_limit)

            # Check if result contains error
            if results and len(results) == 1 and "SQL execution error" in results[0].content:
                raise Exception(results[0].content)

            return results

        except Exception as e:
            output_manager.error(f"❌ SQLfirstExecutefailed: {e}")
            try:
                retry_enabled = bool(getattr(getattr(config, 'dag', None), 'execution', None).channel_retry_enabled)
            except Exception:
                retry_enabled = True
            if not retry_enabled:
                return self._create_placeholder_result("SQL_PLACEHOLDER_EXECUTION_FAILED_NO_RETRY")

            # use error analyzer to generate friendlyError hint
            try:
                from src.utils.sql_error_analyzer import parse_sqlite_error_message
                error_hint = parse_sqlite_error_message(str(e))
            except ImportError:
                error_hint = f"SQL execution error: {str(e)}"

            # Second attempt：pass inError hint
            enhanced_context = f"{parent_context}\n\n[Error hint] {error_hint}"

            try:
                fixed_sql = self.generate_sql_or_placeholder(
                    sql_or_placeholder,  # use rawSQLAs question context
                    enhanced_context
                )

                if fixed_sql.startswith("SQL_PLACEHOLDER"):
                    output_manager.info(f"📋SQL channel retries: {fixed_sql}")
                    return self._create_placeholder_result(fixed_sql)

                results = self.retriever.execute_sql(fixed_sql, result_limit)
                output_manager.info(f"✅ SQLSecond attempt successful")
                return results

            except Exception as retry_error:
                output_manager.error(f"❌ SQLsecond attempt也failed: {retry_error}")
                return self._create_placeholder_result("SQL_PLACEHOLDER_EXECUTION_FAILED")

    def _create_placeholder_result(self, placeholder_message: str) -> List[RetrievalResult]:
        """create placeholder result"""
        placeholder_result = RetrievalResult(
            content=placeholder_message,  
            score=0.0,
            metadata=RetrievalMetadata(
                query_statement="",
                execution_time=0.001,
                result_count=0,
                source_type="mysql"
            ),
            source_id="placeholder",
            source="mysql"
        )
        return [placeholder_result]

    
    def execute_sql(self, sql_query: str, limit: int = None) -> List[RetrievalResult]:
        """ExecuterawSQLquery（core functionality）"""
        return self.retriever.execute_sql(sql_query, limit)

    def retrieve(self, sql_query: str, limit: int = None) -> List[RetrievalResult]:
        """Compatibility method：adapt legacy code calls，Forward toexecute_sql"""
        return self.execute_sql(sql_query, limit)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        return self.retriever.get_database_info()