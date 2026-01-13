"""
LLM Output Cleaning and Normalization Tool

Provides unified output normalization for SQL and Cypher, removing code blocks, prefix noise, explanatory text and extracting main statements.
"""

import re
from typing import Optional
import re


def normalize_sql_output(sql: Optional[str]) -> Optional[str]:
    """Normalize LLM-generated SQL text, remove code block markers and noise, and extract valid SQL"""
    if not sql:
        return sql
    s = sql.strip()

    # Remove outer quotes (if entire string is surrounded by quotes)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    # If empty or only quotes after removing quotes, return None
    if not s or s.strip("'\" ") == "":
        return None

    # Remove Markdown code block markers
    s = re.sub(r"```(?:sql)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()
    # Remove common prefixes (with optional colon/Chinese colon)
    prefixes = [
        "SQL", "sql", "SQL语句", "Sql", "SQL Query", "查询", "生成SQL", "生成 SQL",
        "查询语句", "以下是SQL", "Here is the SQL", "Here is an SQL query"
    ]
    for p in prefixes:
        pattern = rf"^\s*{re.escape(p)}\s*[:：]?\s*"
        s = re.sub(pattern, "", s, flags=re.IGNORECASE)
    # Handle prefix 'sql ' without colon
    s = re.sub(r"^\s*sql\s+", "", s, flags=re.IGNORECASE)
    # Extract main SQL starting from keywords (SELECT/WITH/INSERT/UPDATE/DELETE)
    m = re.search(r"\b(SELECT|WITH|INSERT|UPDATE|DELETE)\b.*", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(0).strip()
    # Truncate at first semicolon (if explanatory text exists)
    if ";" in s:
        s = s.split(";", 1)[0].strip() + ";"
    # Remove trailing Chinese period
    if s.endswith("。"):
        s = s[:-1].rstrip()
    # Final cleanup of residual backticks/code blocks
    s = s.replace("```", "").strip()

    # Final check: if result is empty or only whitespace, return None
    if not s or s.strip() == "":
        return None

    return s


def normalize_cypher_output(cypher: Optional[str]) -> Optional[str]:
    """Normalize LLM-generated Cypher text, remove code block markers and noise, and extract valid Cypher"""
    if not cypher:
        return cypher
    s = cypher.strip()

    # Remove outer quotes (if entire string is surrounded by quotes)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]

    # If empty or only quotes after removing quotes, return None
    if not s or s.strip("'\" ") == "":
        return None

    s = re.sub(r"```(?:cypher)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()
    prefixes = [
        "Cypher", "cypher", "生成Cypher", "生成 Cypher", "查询", "以下是Cypher", "Here is the Cypher"
    ]
    for p in prefixes:
        s = re.sub(rf"^\s*{re.escape(p)}\s*[:：]?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*cypher\s+", "", s, flags=re.IGNORECASE)
    # Extract main Cypher starting from MATCH/RETURN/CREATE
    m = re.search(r"\b(MATCH|RETURN|CREATE|MERGE|WITH)\b.*", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(0).strip()
    if s.endswith("。"):
        s = s[:-1].rstrip()
    s = s.replace("```", "").strip()

    # Final check: if result is empty or only whitespace, return None
    if not s or s.strip() == "":
        return None

    return s


def strip_code_fence(text: Optional[str]) -> Optional[str]:
    """
    Remove Markdown code block markers, supports yaml and json formats:
    - Supports ```yaml, ```json, ``` etc.
    - Automatically extracts code block content
    - Returns as-is in other cases
    """
    if not text:
        return text
    s = text.strip()

    # Check if starts with code block
    if s.startswith("```"):
        # Find first line end position
        first_line_end = s.find('\n')
        if first_line_end == -1:
            # No newline, might be single line, directly remove ```
            if s.endswith("```"):
                return s[3:-3].strip()
            else:
                return s[3:].strip()

        # Has newline, remove first line and trailing ```
        content = s[first_line_end+1:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        return content

    return text


def strip_yaml_code_fence(text: Optional[str]) -> Optional[str]:
    """
    Backward compatible function, now calls the more generic strip_code_fence

    Original functionality:
    Remove Markdown code block markers starting with "```yaml", maintaining consistency with existing logic:
    - Only truncates when string starts with "```yaml";
    - Directly removes first 7 characters and trailing 3 backticks;
    - Returns as-is in other cases, no additional processing.

    Now updated to support multiple code block formats.
    """
    return strip_code_fence(text)


def is_basic_sql_valid(sql: Optional[str]) -> bool:
    """Lightweight SQL validity check, consistent with DAGExecutor internal implementation.
    Rules:
    - Non-empty;
    - Does not contain code block backticks;
    - Does not start with 'sql' prefix;
    - Starts with common verbs (SELECT/WITH/INSERT/UPDATE/DELETE).
    """
    if not sql:
        return False
    s = sql.strip()
    if "```" in s:
        return False
    if re.match(r"^\s*sql\b", s, flags=re.IGNORECASE):
        return False
    return re.match(r"^\s*(SELECT|WITH|INSERT|UPDATE|DELETE)\b", s, flags=re.IGNORECASE) is not None


def is_basic_cypher_valid(cypher: Optional[str]) -> bool:
    """Lightweight Cypher validity check.
    Rules:
    - Non-empty;
    - Does not contain code block backticks;
    - Does not start with 'cypher' prefix;
    - Starts with common keywords (MATCH/RETURN/CREATE/MERGE/WITH).
    """
    if not cypher:
        return False
    s = cypher.strip()
    if "```" in s:
        return False
    if re.match(r"^\s*cypher\b", s, flags=re.IGNORECASE):
        return False
    return re.match(r"^\s*(MATCH|RETURN|CREATE|MERGE|WITH)\b", s, flags=re.IGNORECASE) is not None


def repair_linear_dag_yaml(text: Optional[str]) -> Optional[str]:
    """
    Lightweight repair of linear DAG YAML text, handling the following common issues:
    - Code block markers (```yaml ... ```)
    - Dependency line "dependencies: [ ..." not closed or empty content
    - Missing "dag:" root key, supplement minimal header

    This repair is heuristic, aiming to improve parsing success rate; if complete repair is not possible, caller should provide fallback.
    """
    if text is None:
        return text
    s = strip_yaml_code_fence(text) or ""
    lines = s.splitlines()
    fixed_lines = []
    for line in lines:
        m = re.match(r"^(\s*dependencies:\s*)(.*)$", line)
        if m:
            prefix = m.group(1)
            rest = (m.group(2) or "").strip()
            # Already closed, keep as-is
            if rest.endswith(']'):
                fixed_lines.append(line)
                continue
            # Normalize to flow-style list
            content = rest
            if content.startswith('['):
                content = content[1:].strip()
            # Remove quotes and extra whitespace, split by comma/space
            content = content.replace('"', '').replace("'", '')
            tokens = [t.strip() for t in re.split(r"[,\s]+", content) if t.strip()]
            fixed = prefix + (f"[{', '.join(tokens)}]" if tokens else "[]")
            fixed_lines.append(fixed)
        else:
            fixed_lines.append(line)
    s2 = "\n".join(fixed_lines)
    # If missing dag: root key and nodes: exists, supplement minimal header
    if "dag:" not in s2 and re.search(r"^\s*nodes:\s*$", s2, flags=re.MULTILINE):
        s2 = "dag:\n  id: query_dag\n" + s2
    return s2