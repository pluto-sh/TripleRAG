"""
Thread-Safe SQLite Connection Manager

Solves SQLite thread safety issues in multi-threaded environments, ensuring each thread uses an independent connection.
"""
import sqlite3
import threading
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import os


class ThreadSafeSQLiteManager:
    """
    Thread-Safe SQLite Connection Manager

    Maintains independent SQLite connections for each thread, avoiding "SQLite objects created in a thread
    can only be used in that same thread" errors.
    """

    def __init__(self, db_path: str):
        """
        Initialize thread-safe SQLite manager

        Args:
            db_path: SQLite database file path
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()

        # Ensure database file exists
        if not os.path.exists(db_path):
            # Create database file
            conn = sqlite3.connect(db_path)
            conn.close()
            print(f"[ThreadSafeSQLiteManager] Created new database: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get SQLite connection for current thread

        If current thread doesn't have a connection yet, create a new one

        Returns:
            SQLite connection for current thread
        """
        if not hasattr(self._local, 'connection'):
            # Create new connection for current thread
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False  # Allow cross-thread connection usage
            )
            # Set row factory to return query results as dictionaries
            self._local.connection.row_factory = sqlite3.Row
            # Enable WAL mode to improve concurrent performance
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            # Set busy timeout
            self._local.connection.execute("PRAGMA busy_timeout=30000")
            print(f"[ThreadSafeSQLiteManager] Created new connection for thread {threading.current_thread().name}")

        return self._local.connection
    
    @contextmanager
    def get_cursor(self):
        """
        Context manager to get database cursor for current thread

        Usage:
        with manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM table")
            results = cursor.fetchall()

        Yields:
            Database cursor for current thread
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()

    def execute_sql(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results

        Args:
            sql: SQL query statement
            params: Query parameters

        Returns:
            List of query results, each result is a dictionary
        """
        with self.get_cursor() as cursor:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            # Convert Row objects to dictionaries
            results = [dict(row) for row in cursor.fetchall()]
            return results

    def execute_sql_with_error_handling(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query with error handling

        Args:
            sql: SQL query statement
            params: Query parameters

        Returns:
            List of query results, each result is a dictionary

        Raises:
            RuntimeError: If SQL execution fails
        """
        try:
            return self.execute_sql(sql, params)
        except sqlite3.Error as e:
            error_msg = f"SQLite execution error: {str(e)}"
            print(f"[ThreadSafeSQLiteManager] {error_msg}")
            raise RuntimeError(error_msg)

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information

        Returns:
            Dictionary containing database table information
        """
        try:
            # Get all table names
            tables_sql = "SELECT name FROM sqlite_master WHERE type='table'"
            tables_result = self.execute_sql(tables_sql)
            table_names = [row['name'] for row in tables_result]

            # Get structure information for each table
            table_info = {}
            for table_name in table_names:
                pragma_sql = f"PRAGMA table_info({table_name})"
                columns_result = self.execute_sql(pragma_sql)
                table_info[table_name] = columns_result

            return {
                'tables': table_names,
                'table_info': table_info
            }
        except Exception as e:
            print(f"[ThreadSafeSQLiteManager] Failed to get database information: {e}")
            return {'tables': [], 'table_info': {}}

    def close_connection(self):
        """Close connection for current thread"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')
            print(f"[ThreadSafeSQLiteManager] Closed connection for thread {threading.current_thread().name}")

    def close_all_connections(self):
        """
        Close connections for all threads

        Note: This method can only close thread connections that are currently accessible,
        cannot close connections for other threads that have been created but are not currently accessible
        """
        self.close_connection()
        print("[ThreadSafeSQLiteManager] Closed current thread connection")


class ThreadSafeSQLiteRetriever:
    """
    Thread-Safe SQLite Retriever

    Uses ThreadSafeSQLiteManager to ensure safety in multi-threaded environments
    """

    def __init__(self, db_path: str):
        """
        Initialize thread-safe SQLite retriever

        Args:
            db_path: SQLite database file path
        """
        self.db_path = db_path
        self.manager = ThreadSafeSQLiteManager(db_path)
        print(f"[ThreadSafeSQLiteRetriever] Initialization complete, database: {db_path}")

    def execute_sql(self, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute SQL query

        Args:
            sql: SQL query statement
            params: Query parameters

        Returns:
            List of query results, each result is a dictionary
        """
        return self.manager.execute_sql_with_error_handling(sql, params)

    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information

        Returns:
            Dictionary containing database table information
        """
        return self.manager.get_database_info()

    def close(self):
        """Close retriever"""
        self.manager.close_connection()
        print("[ThreadSafeSQLiteRetriever] Closed")