"""
Dashboard Updater for omerGPT
Updates Streamlit dashboard with real-time performance data
"""
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

class DashboardUpdater:
    """Update dashboard data for Streamlit"""

    def __init__(self):
        self.data_dir = Path("C:/LLM/omerGPT/data")
        self.dashboard_data_file = self.data_dir / "dashboard_data.json"
        self.performance_metrics_file = self.data_dir / "performance_metrics.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_performance_metrics(self) -> List[Dict]:
        """Load performance metrics"""
        try:
            if self.performance_metrics_file.exists():
                with open(self.performance_metrics_file, 'r') as f:
                    data = json.load(f)
                return data.get("metrics", [])
            return []
        except Exception as e:
            logger.error(f"Error loading performance metrics: {e}")
            return []

    def aggregate_metrics(self, metrics: List[Dict], window_minutes: int = 60) -> Dict:
        """Aggregate metrics over time window"""
        if not metrics:
            return {}

        cutoff = datetime.now() - timedelta(minutes=window_minutes)

        # Filter recent metrics
        recent_metrics = [
            m for m in metrics
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]

        if not recent_metrics:
            recent_metrics = metrics[-10:]  # Use last 10 if no recent data

        # Calculate averages
        cpu_values = [m["cpu"]["cpu_percent"] for m in recent_metrics]
        mem_values = [m["memory"]["memory_percent"] for m in recent_metrics]

        aggregated = {
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "memory_avg": sum(mem_values) / len(mem_values),
            "memory_max": max(mem_values),
            "sample_count": len(recent_metrics),
            "time_window_minutes": window_minutes
        }

        # GPU metrics if available
        gpu_metrics = []
        for m in recent_metrics:
            if m.get("gpu"):
                gpu_metrics.extend(m["gpu"])

        if gpu_metrics:
            gpu_loads = [g["load"] for g in gpu_metrics]
            gpu_mem_utils = [g["memory_util"] for g in gpu_metrics]

            aggregated["gpu_avg_load"] = sum(gpu_loads) / len(gpu_loads)
            aggregated["gpu_max_load"] = max(gpu_loads)
            aggregated["gpu_avg_memory"] = sum(gpu_mem_utils) / len(gpu_mem_utils)
            aggregated["gpu_max_memory"] = max(gpu_mem_utils)

        return aggregated

    def prepare_chart_data(self, metrics: List[Dict], limit: int = 100) -> Dict:
        """Prepare data for charts"""
        if not metrics:
            return {}

        # Take last N metrics
        recent = metrics[-limit:]

        # Extract time series data
        timestamps = [m["timestamp"] for m in recent]
        cpu_values = [m["cpu"]["cpu_percent"] for m in recent]
        mem_values = [m["memory"]["memory_percent"] for m in recent]

        chart_data = {
            "timestamps": timestamps,
            "cpu": cpu_values,
            "memory": mem_values
        }

        # GPU data if available
        if recent[0].get("gpu"):
            gpu_data = {
                "gpu_load": [],
                "gpu_memory": []
            }

            for m in recent:
                if m.get("gpu") and len(m["gpu"]) > 0:
                    gpu_data["gpu_load"].append(m["gpu"][0]["load"])
                    gpu_data["gpu_memory"].append(m["gpu"][0]["memory_util"])
                else:
                    gpu_data["gpu_load"].append(0)
                    gpu_data["gpu_memory"].append(0)

            chart_data["gpu"] = gpu_data

        return chart_data

    def get_system_health(self, aggregated: Dict) -> str:
        """Determine system health status"""
        if not aggregated:
            return "Unknown"

        cpu = aggregated.get("cpu_avg", 0)
        memory = aggregated.get("memory_avg", 0)

        if cpu > 90 or memory > 90:
            return "Critical"
        elif cpu > 75 or memory > 75:
            return "Warning"
        elif cpu > 50 or memory > 50:
            return "Good"
        else:
            return "Excellent"

    def update_dashboard_data(self):
        """Update dashboard data file"""
        try:
            logger.info("Updating dashboard data...")

            # Load metrics
            metrics = self.load_performance_metrics()

            if not metrics:
                logger.warning("No metrics available")
                return

            # Aggregate metrics
            aggregated = self.aggregate_metrics(metrics, window_minutes=60)

            # Prepare chart data
            chart_data = self.prepare_chart_data(metrics, limit=100)

            # Determine health
            health = self.get_system_health(aggregated)

            # Prepare dashboard data
            dashboard_data = {
                "last_update": datetime.now().isoformat(),
                "system_health": health,
                "aggregated_metrics": aggregated,
                "chart_data": chart_data,
                "latest_metrics": metrics[-1] if metrics else {}
            }

            # Save
            with open(self.dashboard_data_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)

            logger.info(f"Dashboard data updated. Health: {health}")

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")

    def run_updater(self, interval_seconds: int = 300):
        """Run dashboard updater continuously"""
        logger.info(f"Dashboard updater started (interval: {interval_seconds}s)")

        while True:
            try:
                self.update_dashboard_data()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Dashboard updater stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in updater loop: {e}")
                time.sleep(interval_seconds)

def main():
    """Run dashboard updater"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    updater = DashboardUpdater()

    # Run once
    updater.update_dashboard_data()

    # Or run continuously (uncomment to enable)
    # updater.run_updater(interval_seconds=300)

if __name__ == "__main__":
    main()
