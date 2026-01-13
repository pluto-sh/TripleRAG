# Triple RAG

**An Industrial-Grade Multi-Source Heterogeneous Data Intelligent Q&A System**

Triple RAG implements complex query decomposition and parallel processing through DAG (Directed Acyclic Graph) orchestration, supporting multi-modal fusion retrieval across structured data (SQLite/MySQL), graph data (Neo4j), and semantic data (Vector DB).
The core codes and data related to the paper have been made open source. Due to time constraints, some parts are still under review and will be made open source as soon as possible.

---

## 🌟 Key Features

- **DAG-Based Query Orchestration**: Automatically decomposes complex queries into executable DAG structures with parallel node execution
- **Dynamic Node Type Switching**: Intelligently switches between retrieval and inference modes based on information sufficiency
- **Incremental Retrieval Optimization**: Only queries missing information instead of full retrieval, achieving 50-70% performance improvement
- **Multi-Source Data Fusion**: Seamlessly integrates SQL databases, graph databases, and vector databases
- **Adaptive Optimization**: Smart node skipping and automatic completion checking

---

## 📋 System Requirements

- Python 3.8+
- SQLite (or MySQL)
- Neo4j 3.5+
- ChromaDB
- OpenAI-compatible LLM API (e.g., vLLM, DeepSeek, OpenAI)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Triple_RAG.git
cd Triple_RAG

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Update database paths and LLM settings in `config/config.py`:

```python
@dataclass
class LLMConfig:
    base_url: str = 'http://localhost:8000/v1'  # Your LLM service URL
    model: str = 'Qwen2.5-7B-Instruct'

@dataclass
class SQLiteConfig:
    database_path: str = './dataset/databases/your_database.db'

@dataclass
class Neo4jConfig:
    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'your_password'
```

### 3. Prepare Dataset

Place your datasets in the `dataset/` directory. See `dataset/DATA_FORMAT.md` for detailed format specifications.

**Required files**:
- `dataset/queries.json` - Input queries
- `dataset/databases/*.db` - SQLite database
- Neo4j graph database (configured connection)
- `dataset/vector_db/bench/` - ChromaDB collection

### 4. Run Batch Processing

```bash
python main.py --input dataset/queries.json --output results.jsonl
```

**Output format** (JSONL):
```json
{"query_id": 0, "question": "...", "system_answer": "...", "status": "success"}
{"query_id": 1, "question": "...", "system_answer": "...", "status": "success"}
```

---

## 🏗️ System Architecture

Triple RAG adopts a 4-layer architecture:

![System Architecture](./figure/figure1.png)

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: DAG Orchestration Layer                           │
│  - Query decomposition & DAG generation                     │
│  - Topological sorting & parallel scheduling                │
│  - Adaptive optimization                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Node Execution Layer                              │
│  - Dynamic node type switching (retrieval ↔ inference)      │
│  - Incremental retrieval optimization                       │
│  - Answer generation & formatting                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Data Fusion Layer                                 │
│  - Multi-modal data fusion & deduplication                  │
│  - Runtime context & memory management                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Storage/Retrieval Layer                           │
│  - SQLite/MySQL adapter                                     │
│  - Neo4j graph adapter                                      │
│  - Vector database retriever (ChromaDB)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Triple_RAG_main/
├── README.md              # This file
├── LICENSE                # MIT License
├── requirements.txt       # Python dependencies
├── main.py               # Entry point (batch processing)
├── dataset/              # Dataset directory (prepare your own data)
│   └── DATA_FORMAT.md    # Data format specification
├── config/               # Configuration files
│   ├── config.py         # Main configuration
│   ├── dag_config.yaml   # DAG execution config
│   ├── sql_schema.md     # SQL schema example
│   └── graph_schema.md   # Graph schema example
└── src/                  # Core source code
    ├── core/             # DAG orchestration layer
    ├── retrievers/       # Data retrieval layer
    └── utils/            # Utility functions
```

---

## 🔧 Configuration

### DAG Configuration

Edit `config/dag_config.yaml` to customize DAG execution behavior:

```yaml
dag:
  enabled: true
  max_nodes: 10
  execution:
    max_retry: 3
    node_timeout: 60
    failure_strategy: "stop"
  adaptive_optimizer:
    enabled: true
    skip_enabled: true
    skip_confidence_threshold: 0.7
```

### Database Configuration

Update `config/config.py` with your database credentials and paths.

---

## 💡 Core Innovations

### 1. DAG-Based Query Decomposition

Complex queries are automatically decomposed into a DAG structure where:
- Nodes represent sub-queries (retrieval or inference)
- Edges represent dependencies between nodes
- Parallel execution where possible

### 2. Dynamic Node Type Switching

**For Retrieval Nodes**:
- Check if parent information is sufficient
- If YES → Switch to inference mode (skip retrieval)
- If NO → Perform full retrieval

**For Inference Nodes**:
- Check if parent information is sufficient
- If YES → Direct answer generation
- If NO → Switch to retrieval mode with **incremental retrieval**

### 3. Incremental Retrieval

Instead of full retrieval, the system:
1. Identifies information gaps
2. Generates precise queries for missing information
3. Selectively activates retrieval channels (SQL/Graph/Vector)
4. Merges parent info + incremental results

**Benefits**: 50-70% faster, more precise, reduced redundancy

### 4. Multi-Source Data Fusion

Intelligently fuses results from:
- **SQL databases**: Structured transactional data
- **Graph databases**: Entity relationships and knowledge graphs
- **Vector databases**: Semantic document search

### 5. Adaptive Optimization

- **Node Skipping**: Automatically skips redundant retrieval
- **Completion Checking**: Validates final answer completeness
- **Auto-insertion**: Adds nodes if answer is incomplete

---

## 📊 Example Use Cases

### Multi-Hop Reasoning
```
Query: "Compare CATL's and BYD's largest supplier carbon production"

DAG Execution:
Node 1 (retrieval): Query CATL's largest supplier and carbon production
  → Result: "CATL supplier: Company A, carbon: 500 tons"

Node 2 (inference→retrieval): Compare with BYD
  → Incremental retrieval: Only query BYD data
  → Merge: CATL + BYD → Comparison result
```

### Complex Supply Chain Analysis
```
Query: "Which provinces in China have battery companies? What are their main
        supply materials? Do these provinces have supply relationships?"

DAG Execution:
Node 1: Query provinces with battery companies (SQL + Graph)
Node 2: Query main supply materials (SQL)
Node 3: Analyze inter-province supply relationships (Graph)
Node 4: Synthesize final answer (Inference)
```

---

## 🛠️ Development

### Code Structure

- **`src/core/dag_models.py`**: DAG data structures
- **`src/core/query_router.py`**: Query decomposition engine
- **`src/core/dag_executor.py`**: DAG execution engine
- **`src/core/node_agent.py`**: Node execution logic
- **`src/core/fusion_engine.py`**: Multi-source data fusion
- **`src/core/memory.py`**: Context management

### Adding Custom Retrievers

1. Create a new retriever in `src/retrievers/`
2. Implement the `retrieve(query: str) -> List[Dict]` method
3. Register in `src/core/triple_rag.py`

---

## 📝 Citation

If you use Triple RAG in your research, please cite:

```bibtex
@article{triplerag2025,
  title={Triple RAG: An Industrial-Grade Multi-Source Heterogeneous Data Intelligent Q\&A System},
  author={TripleRAG Development Team},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- Built with [OpenAI API](https://openai.com/api/)
- Graph database powered by [Neo4j](https://neo4j.com/)
- Vector search powered by [ChromaDB](https://www.trychroma.com/)

---

