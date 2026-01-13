"""
SQL Error Analysis Tool (No LLM)

Provides two types of analysis:
- Pre-check errors (basic validity, missing tables)
- Runtime errors (parsing error strings from database drivers)

Outputs concise human-readable hints for feedback in nodes and guiding retries.
"""
from typing import Set
import re


def build_precheck_error_message(sql: str, invalid_basic: bool, missing_tables: Set[str]) -> str:
    """Construct SQL pre-check failure hint message.

    Args:
        sql: SQL to be executed (may be empty)
        invalid_basic: Whether basic validity is not met (starting verb, prefix, code block, etc.)
        missing_tables: Set of referenced non-existent tables

    Returns:
        Concise error hint text
    """
    reasons = []
    if not sql:
        reasons.append("SQL generation is empty")
    if invalid_basic:
        reasons.append("SQL does not start with common verb or contains invalid prefix/code block")
    if missing_tables:
        reasons.append("References non-existent tables: " + ", ".join(sorted(missing_tables)))

    if not reasons:
        return "SQL pre-check failed: Unknown reason"

    return "SQL pre-check failed: " + "; ".join(reasons)


def build_precheck_error_message_extended(sql: str, invalid_basic: bool, missing_tables: Set[str], invalid_columns: Set[str]) -> str:
    """Construct SQL pre-check failure hint including column-level validation.

    Args:
        sql: SQL to be executed (may be empty)
        invalid_basic: Whether basic validity is not met (starting verb, prefix, code block, etc.)
        missing_tables: Set of referenced non-existent tables
        invalid_columns: Set of referenced non-existent columns (format: table.column)

    Returns:
        Concise error hint text
    """
    reasons = []
    if not sql:
        reasons.append("SQL generation is empty")
    if invalid_basic:
        reasons.append("SQL does not start with common verb or contains invalid prefix/code block")
    if missing_tables:
        reasons.append("References non-existent tables: " + ", ".join(sorted(missing_tables)))
    if invalid_columns:
        reasons.append("References non-existent columns: " + ", ".join(sorted(invalid_columns)))

    if not reasons:
        return "SQL pre-check failed: Unknown reason"
    return "SQL pre-check failed: " + "; ".join(reasons)


def parse_mysql_error_message(error_text: str) -> str:
    """Parse MySQL runtime error information, output more friendly hints.

    Common error patterns:
    - Table 'xxx' doesn't exist / ERROR 1146 (42S02)
    - Unknown column 'col' in 'field list' / ERROR 1054 (42S22)
    - You have an error in your SQL syntax ...
    - No database selected / Unknown database 'xxx'
    - Access denied for user 'xxx'
    """
    text = (error_text or "").strip()

    # Extract core error part (allow prefix like "MySQL query execution failed: ")
    m = re.search(r"MySQL查询执行失败:\s*(.*)", text)
    core = m.group(1).strip() if m else text

    # Match by category
    if re.search(r"ERROR\s*1146|doesn't\s+exist", core, re.IGNORECASE):
        # Table doesn't exist
        tm = re.search(r"Table\s+'([^']+)'\s+doesn't\s+exist", core, re.IGNORECASE)
        tbl = tm.group(1) if tm else None
        return f"References non-existent table{f' `{tbl}`' if tbl else ''}, please check table name or runtime table list."

    if re.search(r"ERROR\s*1054|Unknown\s+column", core, re.IGNORECASE):
        cm = re.search(r"Unknown\s+column\s+'([^']+)'", core, re.IGNORECASE)
        col = cm.group(1) if cm else None
        return f"References non-existent column{f' `{col}`' if col else ''}, please check field name or select valid fields."

    if re.search(r"error\s+in\s+your\s+SQL\s+syntax", core, re.IGNORECASE):
        return "SQL syntax error, please check keyword order, quote and bracket pairing, alias and function usage."

    if re.search(r"No\s+database\s+selected|Unknown\s+database", core, re.IGNORECASE):
        return "Database selection error, please confirm connected database matches target database."

    if re.search(r"Access\s+denied\s+for\s+user", core, re.IGNORECASE):
        return "Access permission error, please check database user permissions and connection credentials."

    # Fallback to core text
    return core or "Unknown SQL execution error"

def parse_sqlite_error_message(error_text: str) -> str:
    """Parse SQLite runtime error information, output more friendly hints.
    Common errors:
    - no such table: xxx
    - no such column: xxx
    - near '...': syntax error
    - database is locked
    - permission denied
    """
    text = (error_text or "").strip()
    m = re.search(r"SQL执行错误:\s*(.*)", text)
    core = m.group(1).strip() if m else text

    if re.search(r"no\s+such\s+table", core, re.IGNORECASE):
        tm = re.search(r"no\s+such\s+table:\s*([^\s]+)", core, re.IGNORECASE)
        tbl = tm.group(1) if tm else None
        return f"References non-existent table{f' `{tbl}`' if tbl else ''}, please check table name or runtime table list."

    if re.search(r"no\s+such\s+column", core, re.IGNORECASE):
        cm = re.search(r"no\s+such\s+column:\s*([^\s]+)", core, re.IGNORECASE)
        col = cm.group(1) if cm else None
        return f"References non-existent column{f' `{col}`' if col else ''}, please check field name or select valid fields."

    if re.search(r"near\s+'[^']*':\s*syntax\s+error", core, re.IGNORECASE) or re.search(r"syntax\s+error", core, re.IGNORECASE):
        return "SQL syntax error, please check keyword order, quote and bracket pairing, alias and function usage."

    if re.search(r"database\s+is\s+locked", core, re.IGNORECASE):
        return "Database is locked, please retry later or check concurrent transactions."

    if re.search(r"permission\s+denied", core, re.IGNORECASE):
        return "Access permission error, please check database file permissions and path."

    return core or "Unknown SQL execution error"


def parse_neo4j_error_message(error_text: str) -> str:
    """Parse Neo4j/Cypher runtime error information, output more friendly hints.

    Common error patterns (based on typical fragments from Neo4j driver exceptions):
    - SyntaxError: Invalid input '...' /mismatched input ... expecting ...
    - Neo.ClientError.Schema.EntityNotFound: Node(Labels) or Relationship type not found
    - Neo.ClientError.Statement.ParameterMissing: Expected parameter(s) ...
    - Neo.ClientError.Security.Unauthorized / Forbidden
    - Neo.TransientError.Transaction.LockClientStopped / Database is locked
    """
    text = (error_text or "").strip()
    # Allow error prefix like "Neo4j query execution failed: " or "Graph query execution failed: "
    m = re.search(r"(?:Neo4j查询执行失败|图查询执行失败):\s*(.*)", text)
    core = m.group(1).strip() if m else text

    # Syntax error
    if re.search(r"SyntaxError|invalid\s+input|mismatched\s+input|expecting", core, re.IGNORECASE):
        return "Query syntax error, please check keyword order, bracket/quote pairing, alias and pipe(WITH) usage."

    # Entity/label/relationship type doesn't exist
    if re.search(r"EntityNotFound|Node.*not\s+found|Relationship.*not\s+found|label\s+\w+\s+not\s+found|Unknown\s+label", core, re.IGNORECASE):
        return "References non-existent label or relationship type, please check label/relationship name matches graph schema."

    # Parameter missing
    if re.search(r"ParameterMissing|Expected\s+parameter|parameter\s+\w+\s+is\s+missing", core, re.IGNORECASE):
        return "Missing query parameter, please ensure all $parameters are provided or use literals for testing."

    # Permission related
    if re.search(r"Unauthorized|Forbidden|Access\s+denied", core, re.IGNORECASE):
        return "Access permission error, please check Neo4j user role and credentials."

    # Lock or transaction error
    if re.search(r"LockClientStopped|database\s+is\s+locked|TransientError", core, re.IGNORECASE):
        return "Graph database locked or transaction issue, please retry later or reduce concurrency."

    # Fallback
    return core or "Unknown Cypher execution error"

def parse_sqlite_graph_error_message(error_text: str) -> str:
    """Parse SQLite graph database error message, extract friendly error hints.

    Common error patterns:
    - no such table: xxx
    - no such column: xxx
    - syntax error near '...'
    - database is locked
    - constraint failed
    - table xxx already exists
    """
    text = (error_text or "").strip()
    m = re.search(r"(?:SQLite查询执行失败|SQL执行错误):\s*(.*)", text)
    core = m.group(1).strip() if m else text

    # Table doesn't exist
    if re.search(r"no\s+such\s+table", core, re.IGNORECASE):
        tm = re.search(r"no\s+such\s+table:\s*([^\s]+)", core, re.IGNORECASE)
        tbl = tm.group(1) if tm else None
        return f"Data table doesn't exist{f' `{tbl}`' if tbl else ''}, please check table name is correct."

    # Column doesn't exist
    if re.search(r"no\s+such\s+column", core, re.IGNORECASE):
        cm = re.search(r"no\s+such\s+column:\s*([^\s]+)", core, re.IGNORECASE)
        col = cm.group(1) if cm else None
        return f"Column name doesn't exist{f' `{col}`' if col else ''}, please check column name is correct."

    # Syntax error
    if re.search(r"syntax\s+error", core, re.IGNORECASE):
        return "SQL syntax error, please check query statement format, keyword order and bracket pairing."

    # Table already exists
    if re.search(r"table\s+[^\s]+\s+already\s+exists", core, re.IGNORECASE):
        return "Table already exists, please avoid duplicate creation."

    # Constraint violation
    if re.search(r"constraint\s+failed", core, re.IGNORECASE):
        return "Constraint violation, please ensure data meets constraint conditions."

    # Database locked
    if re.search(r"database\s+is\s+locked", core, re.IGNORECASE):
        return "Database is locked, please retry later."

    # Operation successful
    if re.search(r"not\s+an\s+error", core, re.IGNORECASE):
        return "Operation completed successfully."

    # Fallback
    return core or "Unknown SQLite graph database query error"


def build_cypher_precheck_error_message(cypher: str, invalid_basic: bool, invalid_entities: Set[str]) -> str:
    """Construct Cypher pre-check failure hint message.

    Args:
        cypher: Cypher text to be executed
        invalid_basic: Whether basic validity is not met (starting keyword, code block, etc.)
        invalid_entities: Set of labels or relationship types that don't exist in schema

    Returns:
        Concise error hint text
    """
    reasons = []
    if not cypher:
        reasons.append("Cypher generation is empty")
    if invalid_basic:
        reasons.append("Cypher does not start with common keyword or contains invalid prefix/code block")
    if invalid_entities:
        reasons.append("References non-existent labels/relationship types: " + ", ".join(sorted(invalid_entities)))
    if not reasons:
        return "Cypher pre-check failed: Unknown reason"
    return "Cypher pre-check failed: " + "; ".join(reasons)