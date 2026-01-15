# Dataset Directory

This directory contains the data required by the Triple RAG system. Due to file size limitations, the complete multimodal benchmark dataset is available on Hugging Face.

## 📊 Data Availability

### GitHub Repository (Limited)
This repository contains only the **QAPair** dataset:
- **ChainHopQA.json** - Question-answer pairs for ChainHopQA benchmark
- **ReconHotPotQA.json** - Question-answer pairs for HotpotQA (Reconstruction) benchmark
- **ChainHopQA_full.json** - Extended ChainHopQA dataset
- **ReconHotPotQA_full.json** - Extended HotpotQA dataset

### Complete Dataset (Hugging Face)
The full multimodal benchmark data is available at: [https://huggingface.co/datasets/oo123123/TripleRAG](https://huggingface.co/datasets/oo123123/TripleRAG)

**Complete dataset includes:**
- **SQLite Databases**: Structured transactional data with enterprise trade information
- **Neo4j Graph Databases**: Knowledge graphs with entities and relationships
  - ChainHopQA graph database
  - HotpotQA (Reconstruction) graph database
- **Vector Databases**: ChromaDB collections with document embeddings
  - ChainHopQA vector collection
  - HotpotQA (Reconstruction) vector collection
- **Complete QAPairs**: Full question-answer pairs for both benchmarks

## 📁 Directory Structure

```
dataset/
├── README.md                      # This file
├── QAPair/                        # Question-answer pairs
│   ├── ChainHopQA.json           # ChainHopQA benchmark Q&A pairs
│   ├── ChainHopQA_full.json      # Extended ChainHopQA dataset
│   ├── ReconHotPotQA.json        # HotpotQA (Reconstruction) Q&A pairs
│   └── ReconHotPotQA_full.json   # Extended HotpotQA dataset
├── sqlite/                        # SQLite database files (Available at HuggingFace)
│   ├── ChainHopQA.db             # ChainHopQA SQLite database
│   └── ReconHotPotQA.db          # HotpotQA SQLite database
├── neo4j/                         # Neo4j graph databases (Available at HuggingFace)
│   ├── ChainHopQA/               # ChainHopQA graph database
│   └── ReconHotPotQA/            # HotpotQA graph database
└── vector_db/                     # Vector database (ChromaDB) (Available at HuggingFace)
    ├── README.md                 # Vector DB setup instructions
    ├── ChainHopQA/               # ChainHopQA vector collection
    └── ReconHotPotQA/            # HotpotQA vector collection
```

## 🚀 Quick Start

### Step 1: Prepare Your Data

The dataset is organized by benchmark type (ChainHopQA and ReconHotPotQA), each containing:

1. **Question-Answer Pairs (QAPair)**: Available in GitHub repository
2. **Structured Data (SQLite)**: Relational databases with enterprise trade data
3. **Graph Data (Neo4j)**: Knowledge graphs with entities and relationships
4. **Unstructured Data (Vector DB)**: Document embeddings for semantic search

### Step 2: Import Databases

#### SQLite Database
```bash
# Verify ChainHopQA database
sqlite3 dataset/sqlite/ChainHopQA.db ".tables"

# Verify ReconHotPotQA database  
sqlite3 dataset/sqlite/ReconHotPotQA.db ".tables"

# Example query to check data
sqlite3 dataset/sqlite/ChainHopQA.db "SELECT COUNT(*) FROM EnterpriseTradeData;"
```

#### Neo4j Graph Database
```bash
# Start Neo4j service
neo4j start

# Graph databases are pre-configured for both benchmarks:
# - dataset/neo4j/ChainHopQA/ - ChainHopQA graph database
# - dataset/neo4j/ReconHotPotQA/ - ReconHotPotQA graph database

# Configure connection in config/config.py or .env file
# See config/graph_schema.md for schema details
```

#### Vector Database
```python
# Vector databases are pre-built for both benchmarks:
# - dataset/vector_db/ChainHopQA/ - ChainHopQA vector collection
# - dataset/vector_db/ReconHotPotQA/ - ReconHotPotQA vector collection

# Example usage:
import chromadb

# For ChainHopQA
client = chromadb.PersistentClient(path="./dataset/vector_db/ChainHopQA")
collection = client.get_collection(name="documents")

# For ReconHotPotQA  
client = chromadb.PersistentClient(path="./dataset/vector_db/ReconHotPotQA")
collection = client.get_collection(name="documents")
```

### Step 3: Prepare Query File

Create a `queries.json` file with your questions:

```json
[
  {
    "question": "What are the main battery suppliers in China?",
    "answer": "CATL and BYD are the main suppliers..."
  }
]
```

### Step 4: Run the System

```bash
# Single query mode
python main.py --mode single --query "Your question here"

# Batch query mode
python main.py --mode batch --input dataset/queries.json --output results.jsonl
```

## 📋 Data Requirements

### SQLite Database
- **Table**: `EnterpriseTradeData`
- **Fields**: id, enterprise, country, trade_type, material, material_quantity, unit_price, trade_date
- **Schema**: See `config/sql_schema.md`

### Neo4j Graph Database
- **Node Types**: Product, Organization, Location, Company, Person, Event
- **Relationships**: supplies, produces, located_in, owns, purchases, etc.
- **Schema**: See `config/graph_schema.md`

### Vector Database
- **Format**: ChromaDB collection
- **Collection Name**: `documents`
- **Embedding Model**: `qwen3-embedding` (configurable)

## 📖 Documentation

- **DATA_FORMAT.md**: Comprehensive data format specification
- **databases/README.md**: SQLite database setup guide
- **vector_db/README.md**: Vector database setup guide

## ⚙️ Configuration

All paths are configured in `config/config.py`:

```python
# SQLite databases
database_path: str = './dataset/sqlite/ChainHopQA.db'  # or ReconHotPotQA.db

# Vector databases
persist_directory: str = './dataset/vector_db/ChainHopQA'  # or ReconHotPotQA

# Neo4j (configured separately)
uri: str = 'bolt://localhost:7687'
```

**Note**: You need to switch between ChainHopQA and ReconHotPotQA datasets based on your benchmark choice.

You can also use environment variables via `.env` file (see `.env.example`).

## ✅ Verification Checklist

Before running the system, ensure:

- [ ] QAPair files exist in `dataset/QAPair/`
- [ ] SQLite database files exist in `dataset/sqlite/`
- [ ] SQLite databases have the correct schema
- [ ] Neo4j graph databases exist in `dataset/neo4j/`
- [ ] Neo4j service is running (if using graph data)
- [ ] Neo4j connection credentials are configured
- [ ] Vector databases exist in `dataset/vector_db/`
- [ ] Query file is prepared in the correct format
- [ ] All paths in `config/config.py` are correctly set for your chosen benchmark

## 🔧 Troubleshooting

### Database Not Found
```
Error: FileNotFoundError: database file not found
```
**Solution**: Ensure your database files are in `dataset/sqlite/` with the correct names (ChainHopQA.db or ReconHotPotQA.db).

### Neo4j Connection Failed
```
Error: Neo4j connection error
```
**Solution**:
1. Check if Neo4j service is running: `neo4j status`
2. Verify connection settings in `.env` or `config/config.py`

### Vector DB Empty
```
Error: Collection 'documents' not found
```
**Solution**: 
1. Check if vector databases exist in `dataset/vector_db/ChainHopQA/` or `dataset/vector_db/ReconHotPotQA/`
2. Ensure you're using the correct path for your chosen benchmark

## 📞 Support

For more information:
- Main README: `../README.md`
- Configuration Guide: `../config/config.py`
- Schema Documentation: `../config/sql_schema.md` and `../config/graph_schema.md`

---

**Note**: This directory structure is designed to be flexible. You can adapt it to your specific domain and data sources.
