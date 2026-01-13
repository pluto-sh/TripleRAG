import re
from typing import Set, Dict, List, Tuple


def extract_table_candidates(sql: str) -> Set[str]:
    """Roughly extract possible table name set from SQL.

    Strategy consistent with existing implementation in DAGExecutor:
    - Only extract simple identifiers after FROM/JOIN, don't handle complex cases like subqueries/aliases
    - Filter SQLite system tables (sqlite_sequence/sqlite_schema/sqlite_master)
    """
    candidates: Set[str] = set()
    if not sql:
        return candidates
    for pattern in [r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)", r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)"]:
        for m in re.finditer(pattern, sql, flags=re.IGNORECASE):
            candidates.add(m.group(1))
    ignore = {"sqlite_sequence", "sqlite_schema", "sqlite_master"}
    return {c for c in candidates if c not in ignore}


def detect_missing_tables(sql: str, available_tables: Set[str]) -> Set[str]:
    """Detect table names referenced in SQL but not in available set."""
    candidates = extract_table_candidates(sql)
    return {c for c in candidates if c not in (available_tables or set())}


# ========== Column-level parsing and pre-check helpers ==========

def parse_columns_map_from_sql_schema_text(schema_text: str) -> Dict[str, Set[str]]:
    """Parse {table: {columns,...}} mapping from sql_schema.md text.

    Expected format (example):
    - companies: id, company_name, country_id
    - production_facilities: id, facility_name, ...

    Returns empty mapping if parsing fails.
    """
    columns_map: Dict[str, Set[str]] = {}
    if not schema_text:
        return columns_map
    for line in schema_text.splitlines():
        line = line.strip()
        # Match "- table_name: col1, col2, col3"
        m = re.match(r"^-\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$", line)
        if not m:
            continue
        table = m.group(1)
        cols_part = m.group(2)
        cols = [c.strip() for c in cols_part.split(',') if c.strip()]
        if table and cols:
            columns_map[table] = set(cols)
    return columns_map


def extract_table_aliases(sql: str) -> Dict[str, str]:
    """Extract alias mapping {alias: table} from SQL (supports simple forms of FROM/JOIN).

    Supports:
    - FROM table AS t
    - FROM table t
    - JOIN table AS x
    - JOIN table x
    Does not handle subqueries and complex expressions.
    """
    aliases: Dict[str, str] = {}
    if not sql:
        return aliases
    # FROM clause
    for pattern in [
        r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    ]:
        for m in re.finditer(pattern, sql, flags=re.IGNORECASE):
            table, alias = m.group(1), m.group(2)
            aliases[alias] = table
    # JOIN clause
    for pattern in [
        r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
    ]:
        for m in re.finditer(pattern, sql, flags=re.IGNORECASE):
            table, alias = m.group(1), m.group(2)
            aliases[alias] = table
    return aliases


def extract_column_refs(sql: str) -> List[Tuple[str, str]]:
    """Extract list of qualified column references [(qualifier, column)].

    Recognizes pattern: identifier.identifier (e.g., t.col or table.col).
    """
    refs: List[Tuple[str, str]] = []
    if not sql:
        return refs
    for m in re.finditer(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\b", sql):
        refs.append((m.group(1), m.group(2)))
    return refs


def extract_unqualified_select_columns(sql: str) -> Set[str]:
    """Try to extract set of unqualified column names from SELECT clause.

    Only works in simple SELECT ... FROM scenarios; ignores functions/asterisks and complex expressions.
    """
    result: Set[str] = set()
    if not sql:
        return result
    m = re.search(r"\bSELECT\b(.*?)\bFROM\b", sql, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return result
    select_part = m.group(1)
    # Roughly split columns (by comma), remove function calls and AS aliases
    items = [x.strip() for x in select_part.split(',') if x.strip()]
    for item in items:
        # Remove possible AS alias part
        item = re.sub(r"\bAS\b\s+[a-zA-Z_][a-zA-Z0-9_]*", "", item, flags=re.IGNORECASE)
        # Skip functions/asterisks/qualified columns
        if '(' in item or ')' in item or item == '*' or '.' in item:
            continue
        # Simple identifier matching
        m2 = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)$", item)
        if m2:
            result.add(m2.group(1))
    return result


def detect_invalid_columns(sql: str, columns_map: Dict[str, Set[str]], available_tables: Set[str]) -> Set[str]:
    """Detect column references not in canonical schema.

    Returns format: {"table.column", ...}

    Rules:
    - If qualifier.column appears: if qualifier is alias, map to its table; if table name, use directly.
    - Only validate when qualifier's corresponding table is in available_tables (avoid cross-database misjudgment).
    - For unqualified columns with only one candidate table, attribute column to that table for validation.
    """
    invalid: Set[str] = set()
    if not sql or not columns_map:
        return invalid
    aliases = extract_table_aliases(sql)
    # 1) Handle qualified columns
    for qual, col in extract_column_refs(sql):
        table = None
        if qual in aliases:
            table = aliases[qual]
        elif qual in columns_map:
            table = qual
        if table and (not available_tables or table in available_tables):
            allowed = columns_map.get(table, set())
            if col not in allowed:
                invalid.add(f"{table}.{col}")
    # 2) Handle unqualified columns (only when there's one candidate table)
    candidates = list(extract_table_candidates(sql))
    if len(candidates) == 1:
        only_table = candidates[0]
        if (not available_tables) or (only_table in available_tables):
            allowed = columns_map.get(only_table, set())
            for col in extract_unqualified_select_columns(sql):
                if col not in allowed:
                    invalid.add(f"{only_table}.{col}")
    return invalid