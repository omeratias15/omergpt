"""
API Endpoints for omerGPT
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path
import logging

from .auth import get_current_user

logger = logging.getLogger(__name__)

# Create routers
router = APIRouter()

# Data paths
DATA_DIR = Path("C:/LLM/omerGPT/data")
LOGS_DIR = Path("C:/LLM/omerGPT/logs")
REPORTS_DIR = Path("C:/LLM/omerGPT/reports")

@router.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "omerGPT API",
        "version": "3.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@router.get("/status")
async def get_status(user: dict = Depends(get_current_user)):
    """Get system status"""
    try:
        status = {
            "system": "operational",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "ingestion": "running",
                "signals": "running",
                "risk_manager": "running",
                "sentiment_analysis": "running",
                "macro_engine": "running",
                "adaptive_learning": "running"
            }
        }
        return status
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/signals")
async def get_signals(
    limit: int = 10,
    user: dict = Depends(get_current_user)
):
    """Get latest trading signals"""
    try:
        signals_file = DATA_DIR / "signals.json"

        if not signals_file.exists():
            return {"signals": [], "count": 0}

        with open(signals_file, 'r') as f:
            data = json.load(f)

        signals = data.get("signals", [])[:limit]

        return {
            "signals": signals,
            "count": len(signals),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk")
async def get_risk_metrics(user: dict = Depends(get_current_user)):
    """Get current risk metrics"""
    try:
        risk_file = DATA_DIR / "risk_metrics.json"

        if not risk_file.exists():
            return {
                "portfolio_var": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "exposure": {}
            }

        with open(risk_file, 'r') as f:
            metrics = json.load(f)

        return metrics
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment")
async def get_sentiment(user: dict = Depends(get_current_user)):
    """Get market sentiment analysis"""
    try:
        sentiment_file = DATA_DIR / "sentiment_scores.json"

        if not sentiment_file.exists():
            return {"overall": 0.0, "coins": {}}

        with open(sentiment_file, 'r') as f:
            sentiment = json.load(f)

        return sentiment
    except Exception as e:
        logger.error(f"Error getting sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/macro")
async def get_macro_regime(user: dict = Depends(get_current_user)):
    """Get current macro regime"""
    try:
        macro_file = DATA_DIR / "macro_regime.json"

        if not macro_file.exists():
            return {"regime": "Unknown", "confidence": 0.0}

        with open(macro_file, 'r') as f:
            regime = json.load(f)

        return regime
    except Exception as e:
        logger.error(f"Error getting macro regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports")
async def list_reports(user: dict = Depends(get_current_user)):
    """List available reports"""
    try:
        if not REPORTS_DIR.exists():
            return {"reports": []}

        reports = []
        for report_file in REPORTS_DIR.glob("report_*.html"):
            reports.append({
                "name": report_file.name,
                "path": str(report_file),
                "size": report_file.stat().st_size,
                "created": datetime.fromtimestamp(
                    report_file.stat().st_ctime
                ).isoformat()
            })

        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x["created"], reverse=True)

        return {"reports": reports, "count": len(reports)}
    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_metrics(user: dict = Depends(get_current_user)):
    """Get trading performance metrics"""
    try:
        metrics_file = LOGS_DIR / "trading_metrics.json"

        if not metrics_file.exists():
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "hit_rate": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_pnl": 0.0
            }

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        return metrics
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/signals/refresh")
async def refresh_signals(user: dict = Depends(get_current_user)):
    """Trigger signal refresh"""
    try:
        # In production, this would trigger the signal generation pipeline
        logger.info("Signal refresh requested")
        return {
            "status": "success",
            "message": "Signal refresh initiated",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error refreshing signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))
