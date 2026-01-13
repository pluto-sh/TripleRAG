"""
Triple RAG - Batch Processing Entry Point

This is a simplified version that only supports batch query processing.
For interactive mode or other features, please refer to the full implementation.
"""
import sys
import os

# Ensure project root directory is in search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time
from src.core.triple_rag import TripleRAG

def batch_query_mode(input_file, output_file):
    """
    Batch query processing mode

    Args:
        input_file: Input queries file (JSON format)
        output_file: Output results file (JSONL format)
    """
    rag = TripleRAG()

    try:
        # Read query file (JSON format)
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Detect file format: LogicRAG format vs traditional format
        is_logicrag_format = False
        qa_pairs = []

        if isinstance(data, list):
            # LogicRAG format: directly a list
            is_logicrag_format = True
            qa_pairs = data
            print(f"[Batch Mode] Detected LogicRAG format data")
        elif isinstance(data, dict) and 'qa_pairs' in data:
            # Traditional Triple RAG format: with qa_pairs key
            qa_pairs = data.get('qa_pairs', [])
            print(f"[Batch Mode] Detected Triple RAG format data")
        else:
            print(f"[Batch Mode] Unknown data format, trying to process as list")
            qa_pairs = data if isinstance(data, list) else []

        results = []

        print(f"Starting to process {len(qa_pairs)} queries...")

        for i, qa_pair in enumerate(qa_pairs, 1):
            # Extract query and ground_truth (compatible with both formats)
            if is_logicrag_format:
                query = qa_pair.get('question', '')
                ground_truth = qa_pair.get('answer', '')
                query_id = qa_pair.get('_id', str(i - 1))
            else:
                query = qa_pair.get('question', '')
                ground_truth = qa_pair.get('answer', '')
                query_id = i - 1

            print(f"\nProcessing query {i}/{len(qa_pairs)}: {query[:50]}...")

            try:
                start_time = time.time()
                response = rag.get_json_response(query)
                execution_time = time.time() - start_time

                # Get configuration information
                from config.config import config as global_config
                dag_method = getattr(global_config.dag, 'decomposition_method', 'llm') if hasattr(global_config.dag, 'decomposition_method') else 'llm'
                node_switching = 'disabled' if hasattr(global_config.dag, 'dynamic_type_switching') and hasattr(global_config.dag.dynamic_type_switching, 'enabled') and not global_config.dag.dynamic_type_switching.enabled else 'enabled'
                fusion_mode = getattr(global_config.dag.fusion, 'mode', 'full') if hasattr(global_config.dag, 'fusion') else 'full'

                results.append({
                    "query_id": query_id,
                    "question": query,
                    "ground_truth": ground_truth,
                    "system_answer": response.get('answer', ''),
                    "retrieval_contexts": response.get('retrieved_contexts', []),
                    "execution_time": execution_time,
                    "config": {
                        "dag_method": dag_method,
                        "node_switching": node_switching,
                        "fusion_mode": fusion_mode
                    },
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "query_id": query_id,
                    "question": query,
                    "ground_truth": ground_truth,
                    "system_answer": "",
                    "error": str(e),
                    "status": "error"
                })

        # Output results (JSONL format - one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"\nResults saved to: {output_file}")

    finally:
        rag.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Triple RAG System - Batch Processing Mode',
        epilog='Example: python main.py --input dataset/queries.json --output results.jsonl'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input queries file (JSON format)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output results file (JSONL format)')

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    batch_query_mode(args.input, args.output)

if __name__ == "__main__":
    main()
