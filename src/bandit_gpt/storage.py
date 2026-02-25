"""
Context Persistence (SQLite)

Production-ready storage backends for context vectors in the BanditRouter.

Problem: deque(maxlen=10_000) at 100 QPS fills in 100 seconds.
Human feedback (RLHF) arriving >100s later is lost.

Solution: Pluggable storage backend with production-ready defaults:
- EphemeralContextStore: RAM-based (testing/demos)
- SqliteContextStore: Disk-persisted (production default, zero dependencies)
- Extensible to Redis, S3, etc. without changing router logic
"""

from __future__ import annotations

import logging
import sqlite3
import pickle
import time
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ContextStore(ABC):
    """
    Abstract base class for context vector storage.
    
    Problem: deque(maxlen=10_000) at 100 QPS fills in 100 seconds.
    Human feedback (RLHF) arriving >100s later is lost.
    
    Solution: Pluggable storage backend with production-ready defaults:
    - EphemeralContextStore: RAM-based (testing/demos)
    - SqliteContextStore: Disk-persisted (production default, zero dependencies)
    - Extensible to Redis, S3, etc. without changing router logic
    """
    
    @abstractmethod
    def save_context(self, request_id: str, context: np.ndarray, model_id: str) -> None:
        """Save context vector for later retrieval."""
        pass
    
    @abstractmethod
    def get_context(self, request_id: str) -> Tuple[np.ndarray | None, str | None]:
        """Retrieve (context, model_id) for feedback processing. Returns (None, None) if expired/missing."""
        pass
    
    @abstractmethod
    def prune(self) -> int:
        """Remove expired entries. Returns count of pruned items."""
        pass


class EphemeralContextStore(ContextStore):
    """
    RAM-based context store using bounded deque.
    
    **Use Case**: Testing, demos, or latency-critical deployments where
    feedback arrives within seconds (automated metrics only).
    
    **Limitations**:
    - Fixed capacity (default 10k entries)
    - Lost on restart
    - Unsuitable for RLHF (human feedback arrives hours/days later)
    """
    
    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._store: deque = deque(maxlen=max_size)
        self._index: Dict[str, Tuple[np.ndarray, str]] = {}
    
    def save_context(self, request_id: str, context: np.ndarray, model_id: str) -> None:
        # Evict oldest if at capacity
        if len(self._store) >= self.max_size and request_id not in self._index:
            oldest_id = self._store.popleft()
            self._index.pop(oldest_id, None)
        
        # Store new entry
        if request_id not in self._index:
            self._store.append(request_id)
        self._index[request_id] = (context, model_id)
    
    def get_context(self, request_id: str) -> Tuple[np.ndarray | None, str | None]:
        if request_id in self._index:
            return self._index[request_id]
        return None, None
    
    def prune(self) -> int:
        """No-op for ephemeral store (automatic eviction via deque)."""
        return 0


class SqliteContextStore(ContextStore):
    """
    Production-ready context store using SQLite (zero external dependencies).
    
    **Advantages**:
    - Handles millions of entries
    - Persists across restarts
    - Supports delayed feedback (RLHF, days later)
    - WAL mode for high concurrency (10k+ writes/sec)
    - Automatic TTL-based expiration
    
    **Storage**: ~1KB per context (384-dim embedding + metadata)
    - 1M entries ≈ 1GB disk
    - 10M entries ≈ 10GB disk (weeks of 100 QPS traffic)
    
    **Production Deployment**:
    - Call `prune()` daily via cron/scheduler
    - Default TTL: 7 days (sufficient for most RLHF workflows)
    - For longer retention, increase ttl_seconds or backup to S3
    """
    
    def __init__(self, db_path: str | Path = "data/router_context.db", ttl_seconds: int = 86400 * 7):
        """
        Initialize SQLite context store with intelligent path resolution.
        
        **Lazy Initialization**: Database file is NOT created until first use.
        This prevents cluttering ~/.bandit_gpt/ for users who only route without feedback.
        
        **Path Resolution Strategy**:
        - Absolute path: Use as-is
        - Relative path in dev mode: Resolve to repo's data/ directory
        - Relative path in library mode: Resolve to ~/.bandit_gpt/
        
        Args:
            db_path: Database file path (default: "data/router_context.db")
            ttl_seconds: Time-to-live for context entries (default: 7 days)
        """
        path = Path(db_path)
        
        if path.is_absolute():
            # User provided absolute path - use as-is
            resolved_path = path
        else:
            # Smart resolution: detect dev mode vs library install
            resolved_path = self._resolve_db_path(path)
        
        self.db_path = str(resolved_path)
        self.ttl = ttl_seconds
        self.write_timeout = 5.0  # seconds
        self.read_timeout = 1.0   # seconds
        self._initialized = False  # Lazy initialization flag
    
    def _ensure_initialized(self):
        """
        Lazy initialization: Create database only on first use.
        
        This prevents creating ~/.bandit_gpt/router_context.db for users who
        never use context storage (e.g., routing-only without feedback).
        """
        if not self._initialized:
            # Ensure parent directory exists before creating DB
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
            self._initialized = True
    
    @staticmethod
    def _resolve_db_path(relative_path: Path) -> Path:
        """
        Resolve database path intelligently based on install mode.
        
        **Detection Logic**:
        1. Check if running from development repo (has .git or pyproject.toml nearby)
        2. If dev mode: Use repo's data/ directory
        3. If library mode: Use ~/.bandit_gpt/
        
        This prevents issues with pip installs where site-packages may be read-only.
        
        Args:
            relative_path: Relative path like "data/router_context.db"
        
        Returns:
            Resolved absolute path
        """
        # Get the package installation directory
        package_dir = Path(__file__).parent.parent.parent  # banditGPT repo root or site-packages parent
        
        # Check for development mode indicators
        is_dev_mode = (
            (package_dir / ".git").exists() or  # Git repo
            (package_dir / "pyproject.toml").exists() or  # Python project
            (package_dir / "setup.py").exists() or  # Legacy setup
            (package_dir / "README.md").exists()  # Project readme
        )
        
        if is_dev_mode:
            resolved = package_dir / relative_path
            logger.info("Dev mode detected: Using repo database at %s", resolved)
        else:
            db_filename = relative_path.name if relative_path.name else "router_context.db"
            resolved = Path.home() / ".bandit_gpt" / db_filename
            logger.info("Library mode detected: Using user database at %s", resolved)
        
        return resolved
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # WAL mode: enables concurrent reads during writes (critical for production)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")  # Faster writes, still safe
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_log (
                    request_id TEXT PRIMARY KEY,
                    context_blob BLOB NOT NULL,
                    model_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            # Index for TTL-based pruning
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON context_log(created_at)")
    
    def save_context(self, request_id: str, context: np.ndarray, model_id: str) -> None:
        """Save context with robust error handling (never crashes router)."""
        self._ensure_initialized()  # Lazy init: create DB on first save
        try:
            blob = pickle.dumps(context, protocol=pickle.HIGHEST_PROTOCOL)
            with sqlite3.connect(self.db_path, timeout=self.write_timeout) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO context_log (request_id, context_blob, model_id, created_at) VALUES (?, ?, ?, ?)",
                    (request_id, blob, model_id, time.time())
                )
                conn.commit()
        except sqlite3.OperationalError as e:
            logger.error("Failed to persist context %s: %s", request_id, e)
        except Exception as e:
            logger.error("Unexpected error persisting context %s: %s", request_id, e)
    
    def get_context(self, request_id: str) -> Tuple[np.ndarray | None, str | None]:
        """Retrieve context with timeout and error handling."""
        self._ensure_initialized()  # Lazy init: create DB if needed for retrieval
        try:
            with sqlite3.connect(self.db_path, timeout=self.read_timeout) as conn:
                cursor = conn.execute(
                    "SELECT context_blob, model_id FROM context_log WHERE request_id = ?",
                    (request_id,)
                )
                row = cursor.fetchone()
                if row:
                    context = pickle.loads(row[0])
                    return context, row[1]
            return None, None
        except sqlite3.OperationalError:
            logger.warning("Database locked when retrieving context %s", request_id)
            return None, None
        except Exception as e:
            logger.error("Error retrieving context %s: %s", request_id, e)
            return None, None
    
    def prune(self, force: bool = False) -> int:
        """Remove entries older than TTL. Returns count of deleted rows."""
        self._ensure_initialized()  # Lazy init: create DB if pruning non-existent DB
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                if force:
                    # Delete everything (for testing/reset)
                    cursor = conn.execute("DELETE FROM context_log")
                else:
                    # Delete records older than TTL
                    cutoff = time.time() - self.ttl
                    cursor = conn.execute("DELETE FROM context_log WHERE created_at < ?", (cutoff,))
                
                deleted = cursor.rowcount
                conn.commit()
                
                if deleted > 0:
                    logger.info("Pruned %d old contexts from store", deleted)
                
                return deleted
        except Exception as e:
            logger.error("Error pruning contexts: %s", e)
            return 0
    
    def stats(self) -> dict:
        """Get database statistics for monitoring/debugging."""
        self._ensure_initialized()  # Lazy init: create DB if getting stats
        try:
            with sqlite3.connect(self.db_path, timeout=1.0) as conn:
                # Get counts and timestamps
                row = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM context_log
                """).fetchone()
                
                total, oldest, newest = row
                
                # Get database file size
                from pathlib import Path
                db_size_mb = Path(self.db_path).stat().st_size / (1024 * 1024)
                
                return {
                    "total_contexts": total,
                    "oldest_timestamp": oldest,
                    "newest_timestamp": newest,
                    "db_size_mb": round(db_size_mb, 2),
                    "ttl_days": self.ttl / 86400
                }
        except Exception as e:
            logger.error("Error getting stats: %s", e)
            return {}


# ---------------------------------------------------------------------------
# Checkpoint Manager: Automatic State Persistence
# ---------------------------------------------------------------------------

class CheckpointManager:
    """
    Professional state persistence manager for BanditRouter.
    
    **Architecture**:
    - RAM is fast: A_inv, b, theta stay in memory during runtime
    - Disk is safe: Periodic saves to checkpoint file for crash recovery
    
    **The Professional Implementation**:
    - Atomic writes (temp file + rename) prevent corruption
    - Handles registry drift (new models added since last checkpoint)
    - Separates State (learned) from Config (immutable)
    - Enables seamless "Magic Resume" across restarts
    
    **Usage**:
    ```python
    # At startup
    router = BanditRouter.create(...)
    checkpointer = CheckpointManager()
    
    if checkpointer.load(router):
        print("Resumed from checkpoint")
    else:
        print("Cold start - using procedural warmup")
    
    # During runtime (on shutdown or periodic background task)
    checkpointer.save(router)
    ```
    
    **What Gets Saved**:
    - Learned state: A_inv, b, theta, last_update, t
    - Model registry snapshot (for drift detection)
    - Timestamp (for age tracking)
    
    **What Does NOT Get Saved**:
    - Config (models.json, anchors, features) - comes from code
    - Context store (SQLite already handles this separately)
    - Encoder model (loaded from HuggingFace on startup)
    """
    
    def __init__(self, directory: str | Path = "~/.bandit_gpt/checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            directory: Where to store checkpoint files (default: ~/.bandit_gpt/checkpoints)
        """
        self.directory = Path(directory).expanduser()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.filepath = self.directory / "router_state.pkl"
    
    def save(self, router_instance) -> None:
        """
        Save router's learned state to disk (atomic write).
        
        **Atomic Write Pattern**:
        1. Write to temp file
        2. Rename temp → final (atomic operation)
        3. Prevents corruption if crash happens during write
        
        Args:
            router_instance: BanditRouter instance to save
        """
        # Extract minimal state (only learned parameters)
        state = {
            "t": router_instance.bandit.t,
            "A": {k: v for k, v in router_instance.bandit.A.items()},
            "A_inv": {k: v for k, v in router_instance.bandit.A_inv.items()},
            "b": {k: v for k, v in router_instance.bandit.b.items()},
            "last_update": {k: v for k, v in router_instance.bandit.last_update.items()},
            "models": router_instance.bandit.models,  # For drift detection
            "timestamp": time.time()
        }
        
        # Atomic write (write temp then rename) to prevent corruption
        temp_path = self.filepath.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Atomic rename (crashes before this = old file intact, crashes after = new file intact)
        temp_path.replace(self.filepath)
        
        logger.info("Checkpoint saved to %s", self.filepath)
    
    def load(self, router_instance) -> bool:
        """
        Restore router's learned state from disk.
        
        **Registry Drift Handling**:
        If models.json changed since checkpoint:
        - Keep learned state for existing models
        - Initialize new models with priors + warmup
        - Warn about removed models
        
        Args:
            router_instance: BanditRouter instance to hydrate
        
        Returns:
            True if checkpoint loaded successfully, False for cold start
        """
        if not self.filepath.exists():
            return False
        
        try:
            with open(self.filepath, "rb") as f:
                state = pickle.load(f)
            
            # Check age
            age_seconds = time.time() - state["timestamp"]
            age_str = self._format_age(age_seconds)
            
            # Restore core state
            router_instance.bandit.t = state["t"]
            router_instance.bandit.last_update = state["last_update"]
            
            # Handle registry drift
            saved_models = set(state["models"])
            current_models = set(router_instance.bandit.models)
            
            if saved_models == current_models:
                # Perfect match - restore everything
                router_instance.bandit.A = state["A"]
                router_instance.bandit.A_inv = state["A_inv"]
                router_instance.bandit.b = state["b"]
                logger.info("Checkpoint loaded (%s old, %d models)", age_str, len(saved_models))
            else:
                # Registry drift - merge intelligently
                added_models = current_models - saved_models
                removed_models = saved_models - current_models
                
                # Restore state for models that still exist
                for model in current_models:
                    if model in saved_models:
                        router_instance.bandit.A[model] = state["A"][model]
                        router_instance.bandit.A_inv[model] = state["A_inv"][model]
                        router_instance.bandit.b[model] = state["b"][model]
                
                # Log drift
                drift_msg = []
                if added_models:
                    drift_msg.append(f"+{len(added_models)} new")
                if removed_models:
                    drift_msg.append(f"-{len(removed_models)} removed")
                
                logger.warning("Registry drift detected (%s)", ", ".join(drift_msg))
                logger.info(
                    "Checkpoint loaded (%s old). Kept: %d models",
                    age_str, len(current_models & saved_models),
                )
                if added_models:
                    sample = ", ".join(list(added_models)[:3])
                    suffix = "..." if len(added_models) > 3 else ""
                    logger.info("New (cold-start): %s%s", sample, suffix)
            
            return True
            
        except Exception as e:
            logger.error("Failed to load checkpoint: %s. Starting fresh.", e)
            return False
    
    def _format_age(self, seconds: float) -> str:
        """Format age in human-readable format."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds/60)}m"
        elif seconds < 86400:
            return f"{int(seconds/3600)}h"
        else:
            return f"{int(seconds/86400)}d"
    
    def delete(self) -> bool:
        """
        Delete checkpoint file (for clean restart).
        
        Returns:
            True if file was deleted, False if it didn't exist
        """
        if self.filepath.exists():
            self.filepath.unlink()
            logger.info("Checkpoint deleted: %s", self.filepath)
            return True
        return False
