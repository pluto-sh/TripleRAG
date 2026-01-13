from typing import Optional
import os


def load_sql_schema(path: str = "config/sql_schema_hotpotqa.md") -> str:
    """Load structured database schema (SQL).

    Maintains behavior consistent with existing implementation:
    - Returns fixed message when file is missing
    - Returns message with error details for other exceptions
    """
    try:
        # Get current file directory, then build path relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        full_path = os.path.join(project_root, path)

        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "SQL schema file not found."
    except Exception as e:
        return f"Error loading SQL schema: {e}"


def load_graph_schema(path: str = "config/graph_schema_hotpotqa.md") -> str:
    """Load graph database schema (Neo4j).

    Maintains behavior consistent with existing implementation.
    """
    try:
        # Get current file directory, then build path relative to project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        full_path = os.path.join(project_root, path)

        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Graph database schema file not found."
    except Exception as e:
        return f"Error loading graph database schema: {e}"