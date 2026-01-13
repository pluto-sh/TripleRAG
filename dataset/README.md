# Dataset Directory

This directory contains all the data required by the Triple RAG system.

## 📁 Directory Structure

```
dataset/
├── README.md                      # This file
├── DATA_FORMAT.md                 # Detailed data format specification
├── queries_example.json           # Example query input file
├── databases/                     # SQLite database files
│   ├── README.md                 # Database setup instructions
│   └── battery_supply_chain.db   # [TO BE ADDED] Main database file
└── vector_db/                     # Vector database (ChromaDB)
    ├── README.md                 # Vector DB setup instructions
    └── bench/                    # ChromaDB persistent storage
        └── .gitkeep              # Placeholder for git
```

## 🚀 Quick Start

### Step 1: Prepare Your Data

You need to prepare three types of data:

1. **Structured Data (SQLite)**: Relational database with enterprise trade data
2. **Graph Data (Neo4j)**: Knowledge graph with entities and relationships
3. **Unstructured Data (Vector DB)**: Document embeddings for semantic search

### Step 2: Import Databases

#### SQLite Database
```bash
# Copy your database file
cp /path/to/your/database.db dataset/databases/battery_supply_chain.db

# Verify
sqlite3 dataset/databases/battery_supply_chain.db "SELECT COUNT(*) FROM EnterpriseTradeData;"
```

#### Neo4j Graph Database
```bash
# Start Neo4j service
neo4j start

# Import your graph data (see config/graph_schema.md for schema)
# Configure connection in config/config.py or .env file
```

#### Vector Database
```python
# Build vector database from documents
import chromadb

client = chromadb.PersistentClient(path="./dataset/vector_db/bench")
collection = client.create_collection(name="documents")

# Add your documents
collection.add(
    documents=["doc1", "doc2", ...],
    ids=["id1", "id2", ...]
)
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

See `queries_example.json` for more examples.

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
- **queries_example.json**: Example query input formats

## ⚙️ Configuration

All paths are configured in `config/config.py`:

```python
# SQLite
database_path: str = './dataset/databases/battery_supply_chain.db'

# Vector DB
persist_directory: str = './dataset/vector_db/bench'

# Neo4j (configured separately)
uri: str = 'bolt://localhost:7687'
```

You can also use environment variables via `.env` file (see `.env.example`).

## ✅ Verification Checklist

Before running the system, ensure:

- [ ] SQLite database file exists in `dataset/databases/`
- [ ] SQLite database has the correct schema
- [ ] Neo4j service is running (if using graph data)
- [ ] Neo4j connection credentials are configured
- [ ] Vector database is built in `dataset/vector_db/bench/`
- [ ] Query file is prepared in the correct format
- [ ] All paths in `config/config.py` are correct

## 🔧 Troubleshooting

### Database Not Found
```
Error: FileNotFoundError: database file not found
```
**Solution**: Ensure your database file is in `dataset/databases/` with the correct name.

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
**Solution**: Build your vector database first (see `vector_db/README.md`).

## 📞 Support

For more information:
- Main README: `../README.md`
- Configuration Guide: `../config/config.py`
- Schema Documentation: `../config/sql_schema.md` and `../config/graph_schema.md`

---

**Note**: This directory structure is designed to be flexible. You can adapt it to your specific domain and data sources.
