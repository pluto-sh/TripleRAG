"""
Streamlined Output and Log Management Tool
Separates console output and detailed logging
"""

import logging
import sys
import os
from typing import Optional
from datetime import datetime
import threading

# Global variable to store current query's log file
_current_log_file = None
_log_lock = threading.Lock()

def setup_logging(log_file: str = None):
    """Set up logging system - supports creating separate log file for each query"""
    global _current_log_file

    with _log_lock:
        # If there's already a current log file and no new file specified, reuse existing one
        if _current_log_file is not None and log_file is None:
            return logging.getLogger("triple_rag")

        # If new log file is specified, or need to create new one
        if log_file is None:
            # If no log file specified, create timestamped log file
            from config.config import config
            logs_dir = config.paths.get_absolute_path(config.paths.logs_dir)
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"triple_rag_{timestamp}.log")

        _current_log_file = log_file

    # Create logger
    logger = logging.getLogger("triple_rag")
    logger.setLevel(logging.DEBUG)

    # Clear previous handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler - records all detailed information
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Console handler - only records important information
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(message)s'
    )

    # Set formatters
    file_handler.setFormatter(detailed_formatter)
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def get_current_log_file() -> str:
    """Get current query's log file path"""
    with _log_lock:
        return _current_log_file

def reset_logging():
    """Reset logging state, allow creating new log file"""
    global _current_log_file
    with _log_lock:
        _current_log_file = None

class OutputManager:
    """Output Manager - separates streamlined output and detailed logging"""

    def __init__(self):
        self.logger = None

    def _ensure_logger(self):
        """Ensure logger is initialized"""
        if self.logger is None:
            self.logger = setup_logging()
        return self.logger

    def info(self, message: str, detail: Optional[str] = None):
        """Output important information to console, detailed information to log"""
        print(message)
        if detail:
            self._ensure_logger().debug(detail)

    def success(self, message: str, detail: Optional[str] = None):
        """Output success information"""
        print(f"✓ {message}")
        if detail:
            self._ensure_logger().debug(detail)

    def error(self, message: str, detail: Optional[str] = None):
        """Output error information"""
        print(f"✗ {message}")
        if detail:
            self._ensure_logger().error(detail)

    def warning(self, message: str, detail: Optional[str] = None):
        """Output warning information"""
        print(f"⚠️ {message}")
        if detail:
            self._ensure_logger().warning(detail)

    def debug(self, message: str):
        """Output debug information to log"""
        self._ensure_logger().debug(message)

# Global output manager instance
output_manager = OutputManager()

def log_node_execution(node_id: str, question: str, execution_time: float, result_count: int, channels: list):
    """Log node execution information"""
    channels_str = ", ".join(channels)
    print(f"✓ Node {node_id} executed successfully")

    detail_info = f"""
Node Execution Details:
- Node ID: {node_id}
- Question: {question}
- Execution Time: {execution_time:.3f}s
- Result Count: {result_count}
- Channels Used: {channels_str}
- Timestamp: {datetime.now().isoformat()}
"""
    output_manager.debug(detail_info)

def log_user_query(original_query: str):
    """Log user's original query"""
    print(f"🔍 User Query: {original_query}")

    detail_info = f"""
User Query Details:
- Original Query: {original_query}
- Query Time: {datetime.now().isoformat()}
- Query Status: Processing started
"""
    output_manager.debug(detail_info)

def log_node_detailed_execution(
    node_id: str,
    original_question: str,
    understood_question: str,
    answer_text: str,
    execution_time: float,
    channels: list,
    predicted_node_type: str = None,
    actual_node_type: str = None,
    node_type_switched: bool = False,
    switch_reason: str = ""
):
    """Log detailed node execution information (including question understanding, answer, and node type)"""
    channels_str = ", ".join(channels)

    # Build type display information
    type_display = ""
    if predicted_node_type and actual_node_type:
        if node_type_switched:
            type_display = f" [Type Switch: {predicted_node_type}→{actual_node_type}]"
        else:
            type_display = f" [Type: {actual_node_type}]"

    print(f"✓ Node {node_id} executed successfully{type_display}")

    # Console output simplified information
    answer_preview = answer_text[:100] + "..." if len(answer_text) > 100 else answer_text
    print(f"  Question: {understood_question}")
    print(f"  Answer: {answer_preview}")
    print(f"  Time: {execution_time:.3f}s")

    # Detailed information logged to file
    type_info_lines = []
    if predicted_node_type:
        type_info_lines.append(f"- Predicted Node Type: {predicted_node_type}")
    if actual_node_type:
        type_info_lines.append(f"- Actual Node Type: {actual_node_type}")
    if node_type_switched:
        type_info_lines.append(f"- Type Switch: Yes")
        if switch_reason:
            type_info_lines.append(f"- Switch Reason: {switch_reason}")
    else:
        type_info_lines.append(f"- Type Switch: No")

    type_info_str = "\n".join(type_info_lines) if type_info_lines else ""

    detail_info = f"""
================================================================
Detailed Node Execution Record:
================================================================
- Node ID: {node_id}
- Original Question: {original_question}
- LLM Understood Question: {understood_question}
- Complete Answer: {answer_text}
- Execution Time: {execution_time:.3f}s
- Channels Used: {channels_str}
{type_info_str}
- Completion Time: {datetime.now().isoformat()}
================================================================
"""
    output_manager.debug(detail_info)

def log_final_dag_summary(
    original_query: str,
    all_node_info: dict,
    final_answer: str,
    total_time: float,
    retrieval_node_count: int = None,
    inference_node_count: int = None,
    switched_count: int = None,
    prediction_accuracy: float = None
):
    """Log final DAG execution summary (including node type statistics)"""
    print(f"\n🎯 DAG Execution Complete Summary")
    print(f"Total Time: {total_time:.1f}s")

    # If statistics available, display simplified version on console
    if retrieval_node_count is not None and inference_node_count is not None:
        print(f"Node Distribution: Retrieval Nodes={retrieval_node_count}, Inference Nodes={inference_node_count}")
        if prediction_accuracy is not None:
            print(f"Prediction Accuracy: {prediction_accuracy:.1%}")

    # Console output simplified summary
    final_answer_preview = final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
    print(f"Final Answer: {final_answer_preview}")

    # Detailed summary logged to file
    summary_parts = [
        "=" * 80,
        "DAG Execution Final Summary",
        "=" * 80,
        f"User Original Question: {original_query}",
        "",
        "Node Processing Details:"
    ]

    for node_id, node_info in all_node_info.items():
        # Build node type display
        predicted_type = node_info.get('predicted_node_type', 'Unknown')
        actual_type = node_info.get('actual_node_type', 'Unknown')
        switched = node_info.get('node_type_switched', False)
        switch_reason = node_info.get('switch_reason', '')

        type_display = f"   Node Type: {actual_type}"
        if switched:
            type_display += f" (Predicted: {predicted_type}, Switched)"
            if switch_reason:
                type_display += f"\n   Switch Reason: {switch_reason}"
        else:
            type_display += f" (Predicted: {predicted_type}, No Switch)"

        summary_parts.extend([
            f"",
            f"📌 Node {node_id}:",
            f"   Original Question: {node_info.get('original_question', 'Unknown')}",
            f"   LLM Understood Question: {node_info.get('understood_question', 'Unknown')}",
            f"   Node Answer: {node_info.get('answer_text', 'Unknown')}",
            f"   Execution Time: {node_info.get('execution_time', 0):.3f}s",
            f"   Channels Used: {', '.join(node_info.get('channels_used', []))}",
            type_display
        ])

    # Add statistical summary
    if retrieval_node_count is not None and inference_node_count is not None:
        total_nodes = retrieval_node_count + inference_node_count
        summary_parts.extend([
            "",
            "📊 Node Type Statistics:",
            f"   Total Nodes: {total_nodes}",
            f"   Retrieval Nodes: {retrieval_node_count}",
            f"   Inference Nodes: {inference_node_count}",
        ])

        if switched_count is not None:
            summary_parts.append(f"   Type Switches: {switched_count}")

        if prediction_accuracy is not None:
            summary_parts.append(f"   Prediction Accuracy: {prediction_accuracy:.2%}")

    summary_parts.extend([
        "",
        f"🎯 Final Answer: {final_answer}",
        f"⏱️  Total Execution Time: {total_time:.3f}s",
        f"📅 Completion Time: {datetime.now().isoformat()}",
        "=" * 80
    ])

    detail_summary = "\n".join(summary_parts)
    output_manager.debug(detail_summary)

def log_sql_generation(sql_query: str, success: bool, error_msg: Optional[str] = None):
    """Log SQL generation information"""
    if success:
        print(f"  SQL Generation: ✓")
        output_manager.debug(f"Successfully generated SQL:\n{sql_query}")
    else:
        print(f"  SQL Generation: ✗ {error_msg or 'Failed'}")
        output_manager.debug(f"SQL generation failed: {error_msg}")

def log_dag_execution(total_nodes: int, successful_nodes: int, total_time: float):
    """Log DAG execution summary"""
    success_rate = (successful_nodes / total_nodes * 100) if total_nodes > 0 else 0

    print(f"\n🎯 DAG Execution Complete: {successful_nodes}/{total_nodes} nodes successful ({success_rate:.0f}%) | Total Time: {total_time:.1f}s")

    detail_info = f"""
DAG Execution Summary:
- Total Nodes: {total_nodes}
- Successful Nodes: {successful_nodes}
- Failed Nodes: {total_nodes - successful_nodes}
- Success Rate: {success_rate:.1f}%
- Total Time: {total_time:.3f}s
- Completion Time: {datetime.now().isoformat()}
"""
    output_manager.debug(detail_info)