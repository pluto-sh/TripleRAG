"""
Triple RAG Configuration File
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Get project root directory
def get_project_root() -> Path:
    """Get project root directory (where config/ folder is located)"""
    return Path(__file__).parent.parent

@dataclass
class PathConfig:
    """Path Configuration - All paths are relative to project root"""
    # Project root directory
    project_root: str = field(default_factory=lambda: str(get_project_root()))

    # Log directory
    logs_dir: str = './logs'

    # Schema file paths
    sql_schema_path: str = 'config/sql_schema_hotpotqa.md'
    graph_schema_path: str = 'config/graph_schema_hotpotqa.md'

    def get_absolute_path(self, relative_path: str) -> str:
        """Convert relative path to absolute path"""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(self.project_root, relative_path)

@dataclass
class LLMConfig:
    """LLM Configuration"""
    # vLLM service configuration
    base_url: str = 'http://localhost:8000/v1'
    api_key: str = 'EMPTY'  # vLLM does not require API key
    model: str = 'Qwen2.5-7B-Instruct'
    max_tokens: int = 4096
    temperature: float = 0.0

@dataclass
class MySQLConfig:
    """MySQL Configuration"""
    host: str = 'localhost'
    port: int = 3306
    user: str = 'root'
    password: str = 'your_password'  # Please change to your MySQL password
    database: str = 'triple_rag_db'

@dataclass
class SQLiteConfig:
    """SQLite Configuration"""
    database_path: str = './dataset/databases/battery_supply_chain.db'
    max_results: int = 50

@dataclass
class Neo4jConfig:
    """Neo4j Configuration"""
    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'your_password'  # Please change to your Neo4j password

@dataclass
class VectorDBConfig:
    """Vector Database Configuration"""
    persist_directory: str = './dataset/vector_db/bench'
    collection_name: str = 'documents'
    embedding_model: str = 'qwen3-embedding'
    max_results: int = 5

# ========== DAG Configuration Classes ==========

@dataclass
class DAGAdaptiveOptimizerConfig:
    """DAG Adaptive Optimizer Configuration"""
    # Master switch
    enabled: bool = False

    # Node skipping configuration
    skip_enabled: bool = True
    skip_confidence_threshold: float = 0.7

    # Final layer completion check
    final_completion_check_enabled: bool = True
    final_insertion_max: int = 1
    final_completion_threshold: float = 0.85
    final_completion_llm_temperature: float = 0.1

@dataclass
class DAGExecutionConfig:
    """DAG Execution Configuration"""
    max_retry: int = 3
    node_timeout: int = 60
    failure_strategy: str = "stop"
    # Retry count for each retrieval channel (MySQL/Neo4j/Vector)
    channel_retry_max: int = 2
    channel_retry_enabled: bool = True

@dataclass
class DAGDecompositionConfig:
    """DAG Decomposition Configuration"""
    dag_model: Optional[str] = None

@dataclass
class DAGConfig:
    """DAG Mode Configuration"""
    enabled: bool = False
    dag_type: str = "linear"
    max_nodes: int = 10
    execution: DAGExecutionConfig = field(default_factory=DAGExecutionConfig)
    decomposition: DAGDecompositionConfig = field(default_factory=DAGDecompositionConfig)
    adaptive_optimizer: DAGAdaptiveOptimizerConfig = field(default_factory=DAGAdaptiveOptimizerConfig)

@dataclass
class TripleRAGConfig:
    """Triple RAG Main Configuration"""
    # Path configuration
    paths: PathConfig = field(default_factory=PathConfig)

    # Component configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)

    # DAG configuration
    dag: DAGConfig = field(default_factory=DAGConfig)

    # DAG mode switch (backward compatibility)
    use_dag: bool = False

    # Test mode configuration
    test_mode: bool = True  # Enable test format output ([answer1],[answer2],...)

    # LogicRAG mode configuration
    logicrag_mode: bool = False  # Enable LogicRAG dataset mode (vector channel only)
    logicrag_dataset: Optional[str] = None  # LogicRAG dataset name (hotpotqa, 2wikimultihopqa, musique)

    # Retrieval weight configuration
    mysql_weight: float = 0.3
    neo4j_weight: float = 0.3
    vector_weight: float = 0.4

    # Retrieval result limits
    max_mysql_results: int = 10
    max_neo4j_results: int = 10
    max_vector_results: int = 5

    def __post_init__(self):
        """Post-initialization: load dag_config.yaml"""
        self._load_dag_config()

        # Sync use_dag and dag.enabled
        self.use_dag = self.dag.enabled

        # LogicRAG mode setup
        if self.logicrag_mode:
            self._setup_logicrag_mode()

        print(f"[Configuration] DAG mode: {self.use_dag}, LogicRAG mode: {self.logicrag_mode}")

    def _load_dag_config(self):
        """Load DAG configuration from dag_config.yaml"""
        # Find YAML file
        config_dir = Path(__file__).parent
        yaml_path = config_dir / 'dag_config.yaml'

        if not yaml_path.exists():
            print(f"ℹ️  dag_config.yaml not found, using default configuration: {yaml_path}")
            return

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            print(f"✓ Successfully loaded dag_config.yaml")

            # Load DAG configuration
            if 'dag' in yaml_config:
                dag_dict = yaml_config['dag']

                # Update basic configuration
                self.dag.enabled = dag_dict.get('enabled', False)
                self.dag.max_nodes = dag_dict.get('max_nodes', 10)

                # Update execution configuration
                if 'execution' in dag_dict:
                    exec_dict = dag_dict['execution']
                    self.dag.execution.max_retry = exec_dict.get('max_retry', 3)
                    self.dag.execution.node_timeout = exec_dict.get('node_timeout', 60)
                    self.dag.execution.failure_strategy = exec_dict.get('failure_strategy', 'stop')
                    self.dag.execution.channel_retry_max = exec_dict.get('channel_retry_max', 2)
                    self.dag.execution.channel_retry_enabled = bool(exec_dict.get('channel_retry_enabled', True))

                # Update decomposition configuration
                if 'decomposition' in dag_dict:
                    decomp_dict = dag_dict['decomposition'] or {}
                    if isinstance(decomp_dict, dict):
                        dag_model = decomp_dict.get('dag_model')
                        if dag_model is not None:
                            self.dag.decomposition.dag_model = dag_model

                # Load adaptive optimizer configuration
                if 'adaptive_optimizer' in dag_dict:
                    opt_dict = dag_dict['adaptive_optimizer'] or {}
                    if isinstance(opt_dict, dict):
                        # Master switch
                        self.dag.adaptive_optimizer.enabled = bool(opt_dict.get('enabled', False))

                        # Skip configuration
                        self.dag.adaptive_optimizer.skip_enabled = bool(opt_dict.get('skip_enabled', True))
                        self.dag.adaptive_optimizer.skip_confidence_threshold = float(
                            opt_dict.get('skip_confidence_threshold', 0.7)
                        )

                        # Final layer completion check configuration
                        self.dag.adaptive_optimizer.final_completion_check_enabled = bool(
                            opt_dict.get('final_completion_check_enabled', True)
                        )
                        self.dag.adaptive_optimizer.final_insertion_max = int(
                            opt_dict.get('final_insertion_max', 1)
                        )
                        self.dag.adaptive_optimizer.final_completion_threshold = float(
                            opt_dict.get('final_completion_threshold', 0.85)
                        )
                        self.dag.adaptive_optimizer.final_completion_llm_temperature = float(
                            opt_dict.get('final_completion_llm_temperature', 0.1)
                        )

                        print(f"✓ Adaptive optimizer configuration loaded: enabled={self.dag.adaptive_optimizer.enabled}")

                print(f"✓ DAG configuration loaded: enabled={self.dag.enabled}")
            else:
                print(f"⚠️  No 'dag' configuration found in dag_config.yaml")

            # Compatibility override: apply overrides from compatibility node
            try:
                compat = yaml_config.get('compatibility', {})
                if isinstance(compat, dict) and compat:
                    # Override database type
                    dbt = compat.get('database_type')
                    if isinstance(dbt, str) and dbt:
                        self.database_type = dbt
                    # Override Neo4j type
                    neo = compat.get('use_simple_neo4j')
                    if neo is not None:
                        self.use_simple_neo4j = bool(neo)
                    # Override weights
                    weights = compat.get('weights', {})
                    if isinstance(weights, dict) and weights:
                        self.mysql_weight = float(weights.get('mysql', self.mysql_weight))
                        self.neo4j_weight = float(weights.get('neo4j', self.neo4j_weight))
                        self.vector_weight = float(weights.get('vector', self.vector_weight))
                    # Override result limits
                    maxres = compat.get('max_results', {})
                    if isinstance(maxres, dict) and maxres:
                        self.max_mysql_results = int(maxres.get('mysql', self.max_mysql_results))
                        self.max_neo4j_results = int(maxres.get('neo4j', self.max_neo4j_results))
                        self.max_vector_results = int(maxres.get('vector', self.max_vector_results))
                    print(f"✓ Compatibility overrides applied: database_type={self.database_type}, use_simple_neo4j={self.use_simple_neo4j}")
            except Exception as e:
                print(f"⚠️  Failed to apply compatibility overrides: {e}")

        except FileNotFoundError:
            print(f"⚠️  dag_config.yaml file not found: {yaml_path}")
        except yaml.YAMLError as e:
            print(f"✗ YAML parsing error: {e}")
        except Exception as e:
            print(f"✗ Failed to load dag_config.yaml: {e}")
            import traceback
            traceback.print_exc()

    def _setup_logicrag_mode(self):
        """Setup LogicRAG mode: use vector channel only, disable SQL and Graph"""
        print(f"\n{'='*60}")
        print(f"🔧 Enabling LogicRAG mode")
        print(f"{'='*60}")

        # Disable SQL and Graph channels (set weights to 0)
        self.mysql_weight = 0.0
        self.neo4j_weight = 0.0
        self.vector_weight = 1.0

        print(f"✓ Retrieval channel configuration: SQL=0.0, Graph=0.0, Vector=1.0")

        # If dataset is specified, update vector DB path
        if self.logicrag_dataset:
            try:
                from config.dataset_config import LogicRAGDatasets
                dataset = LogicRAGDatasets.get_dataset(self.logicrag_dataset)

                if dataset:
                    self.vector_db.persist_directory = dataset.vector_db_path
                    print(f"✓ Vector DB path: {dataset.vector_db_path}")
                    print(f"✓ Dataset: {self.logicrag_dataset} ({dataset.description})")
                else:
                    print(f"⚠️  Dataset '{self.logicrag_dataset}' not found, using default configuration")

            except Exception as e:
                print(f"⚠️  Failed to load dataset configuration: {e}")

        print(f"{'='*60}\n")

# Global configuration instance
config = TripleRAGConfig()
