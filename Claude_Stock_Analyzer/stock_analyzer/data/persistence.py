"""
Database Persistence Layer
===========================
SQLite-based persistence for:
- Analysis results
- Backtest results
- Model metrics
- Trading signals history
- Portfolio snapshots
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from core.config import DATA_DIR

logger = logging.getLogger("stock_analyzer.persistence")

# Database file location
DB_PATH = DATA_DIR / "stock_analyzer.db"


@dataclass
class AnalysisRecord:
    """Record of a stock analysis."""
    id: Optional[int]
    symbol: str
    timestamp: datetime
    signal: str
    confidence: float
    rsi: Optional[float]
    macd: Optional[float]
    price: float
    indicators: Dict[str, Any]


@dataclass
class BacktestRecord:
    """Record of a backtest result."""
    id: Optional[int]
    symbol: str
    timestamp: datetime
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    parameters: Dict[str, Any]


@dataclass
class SignalRecord:
    """Record of a trading signal."""
    id: Optional[int]
    symbol: str
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    price: float
    model_prediction: Optional[float]
    was_executed: bool = False
    outcome: Optional[float] = None


class DatabaseManager:
    """
    SQLite database manager for persisting analysis data.

    Uses connection pooling and context managers for safety.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self) -> None:
        """Create database and tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    rsi REAL,
                    macd REAL,
                    price REAL,
                    indicators TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Backtest results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    start_date DATETIME,
                    end_date DATETIME,
                    initial_capital REAL,
                    final_value REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    parameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL,
                    price REAL,
                    model_prediction REAL,
                    was_executed INTEGER DEFAULT 0,
                    outcome REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    symbol TEXT,
                    timestamp DATETIME NOT NULL,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    feature_importance TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Portfolio snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_value REAL,
                    cash REAL,
                    positions TEXT,
                    daily_return REAL,
                    cumulative_return REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_symbol ON analyses(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtests_symbol ON backtests(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # Analysis methods
    def save_analysis(self, record: AnalysisRecord) -> int:
        """Save an analysis result."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analyses (symbol, timestamp, signal, confidence, rsi, macd, price, indicators)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.symbol,
                record.timestamp,
                record.signal,
                record.confidence,
                record.rsi,
                record.macd,
                record.price,
                json.dumps(record.indicators)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_analyses(
        self,
        symbol: Optional[str] = None,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AnalysisRecord]:
        """Get analysis records with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM analyses WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                AnalysisRecord(
                    id=row['id'],
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    signal=row['signal'],
                    confidence=row['confidence'],
                    rsi=row['rsi'],
                    macd=row['macd'],
                    price=row['price'],
                    indicators=json.loads(row['indicators']) if row['indicators'] else {}
                )
                for row in rows
            ]

    # Backtest methods
    def save_backtest(self, record: BacktestRecord) -> int:
        """Save a backtest result."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtests
                (symbol, timestamp, start_date, end_date, initial_capital, final_value,
                 total_return, sharpe_ratio, max_drawdown, total_trades, win_rate, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.symbol,
                record.timestamp,
                record.start_date,
                record.end_date,
                record.initial_capital,
                record.final_value,
                record.total_return,
                record.sharpe_ratio,
                record.max_drawdown,
                record.total_trades,
                record.win_rate,
                json.dumps(record.parameters)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_backtests(
        self,
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> List[BacktestRecord]:
        """Get backtest records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if symbol:
                cursor.execute(
                    "SELECT * FROM backtests WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM backtests ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )

            rows = cursor.fetchall()

            return [
                BacktestRecord(
                    id=row['id'],
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    start_date=row['start_date'],
                    end_date=row['end_date'],
                    initial_capital=row['initial_capital'],
                    final_value=row['final_value'],
                    total_return=row['total_return'],
                    sharpe_ratio=row['sharpe_ratio'],
                    max_drawdown=row['max_drawdown'],
                    total_trades=row['total_trades'],
                    win_rate=row['win_rate'],
                    parameters=json.loads(row['parameters']) if row['parameters'] else {}
                )
                for row in rows
            ]

    def get_best_backtests(self, metric: str = 'sharpe_ratio', limit: int = 10) -> List[BacktestRecord]:
        """Get top backtests by a specific metric."""
        valid_metrics = ['sharpe_ratio', 'total_return', 'win_rate']
        if metric not in valid_metrics:
            metric = 'sharpe_ratio'

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM backtests ORDER BY {metric} DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()

            return [
                BacktestRecord(
                    id=row['id'],
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    start_date=row['start_date'],
                    end_date=row['end_date'],
                    initial_capital=row['initial_capital'],
                    final_value=row['final_value'],
                    total_return=row['total_return'],
                    sharpe_ratio=row['sharpe_ratio'],
                    max_drawdown=row['max_drawdown'],
                    total_trades=row['total_trades'],
                    win_rate=row['win_rate'],
                    parameters=json.loads(row['parameters']) if row['parameters'] else {}
                )
                for row in rows
            ]

    # Signal methods
    def save_signal(self, record: SignalRecord) -> int:
        """Save a trading signal."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO signals
                (symbol, timestamp, signal_type, confidence, price, model_prediction, was_executed, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.symbol,
                record.timestamp,
                record.signal_type,
                record.confidence,
                record.price,
                record.model_prediction,
                1 if record.was_executed else 0,
                record.outcome
            ))
            conn.commit()
            return cursor.lastrowid

    def get_signals(
        self,
        symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        limit: int = 100
    ) -> List[SignalRecord]:
        """Get signal records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM signals WHERE 1=1"
            params = []

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if signal_type:
                query += " AND signal_type = ?"
                params.append(signal_type)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [
                SignalRecord(
                    id=row['id'],
                    symbol=row['symbol'],
                    timestamp=row['timestamp'],
                    signal_type=row['signal_type'],
                    confidence=row['confidence'],
                    price=row['price'],
                    model_prediction=row['model_prediction'],
                    was_executed=bool(row['was_executed']),
                    outcome=row['outcome']
                )
                for row in rows
            ]

    def update_signal_outcome(self, signal_id: int, outcome: float, was_executed: bool = True) -> None:
        """Update a signal with its outcome."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE signals SET outcome = ?, was_executed = ? WHERE id = ?
            """, (outcome, 1 if was_executed else 0, signal_id))
            conn.commit()

    # Model metrics methods
    def save_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
        symbol: Optional[str] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> int:
        """Save model performance metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_metrics
                (model_name, symbol, timestamp, accuracy, precision_score, recall, f1_score, auc_roc, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                symbol,
                datetime.now(),
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1'),
                metrics.get('auc_roc'),
                json.dumps(feature_importance) if feature_importance else None
            ))
            conn.commit()
            return cursor.lastrowid

    # Portfolio methods
    def save_portfolio_snapshot(
        self,
        total_value: float,
        cash: float,
        positions: Dict[str, Any],
        daily_return: float,
        cumulative_return: float
    ) -> int:
        """Save a portfolio snapshot."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO portfolio_snapshots
                (timestamp, total_value, cash, positions, daily_return, cumulative_return)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                total_value,
                cash,
                json.dumps(positions),
                daily_return,
                cumulative_return
            ))
            conn.commit()
            return cursor.lastrowid

    def get_portfolio_history(self, limit: int = 252) -> List[Dict[str, Any]]:
        """Get portfolio history (default: 1 year of trading days)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()

            return [
                {
                    'timestamp': row['timestamp'],
                    'total_value': row['total_value'],
                    'cash': row['cash'],
                    'positions': json.loads(row['positions']) if row['positions'] else {},
                    'daily_return': row['daily_return'],
                    'cumulative_return': row['cumulative_return']
                }
                for row in rows
            ]

    # Utility methods
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {}

            cursor.execute("SELECT COUNT(*) FROM analyses")
            stats['total_analyses'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM backtests")
            stats['total_backtests'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM signals")
            stats['total_signals'] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM analyses")
            stats['unique_symbols_analyzed'] = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(sharpe_ratio) FROM backtests WHERE sharpe_ratio IS NOT NULL")
            result = cursor.fetchone()[0]
            stats['avg_sharpe_ratio'] = result if result else 0

            return stats

    def cleanup_old_records(self, days: int = 365) -> int:
        """Delete records older than specified days."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
            cutoff_dt = datetime.fromtimestamp(cutoff)

            total_deleted = 0

            for table in ['analyses', 'backtests', 'signals', 'model_metrics', 'portfolio_snapshots']:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_dt,))
                total_deleted += cursor.rowcount

            conn.commit()
            logger.info(f"Cleaned up {total_deleted} old records")
            return total_deleted


# Global database instance
db = DatabaseManager()
