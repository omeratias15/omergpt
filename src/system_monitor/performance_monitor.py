"""
src/system_monitor/performance_monitor.py
System performance monitoring for omerGPT.
Tracks GPU/CPU usage, memory consumption, disk I/O, and network stats.
Logs metrics to DuckDB every 30 seconds and exposes FastAPI endpoints.
GPU monitoring via nvidia-ml-py3 (pynvml), CPU/memory via psutil.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio
import logging
import os
import platform
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from storage.db_manager import DatabaseManager

logger = logging.getLogger("omerGPT.system_monitor.performance")


class GPUMonitor:
    """
    Monitor NVIDIA GPU metrics using NVML (nvidia-ml-py3).
    Tracks utilization, memory, temperature, power, and clock speeds.
    """
    
    def __init__(self):
        """Initialize GPU monitor."""
        self.enabled = False
        self.device_count = 0
        self.handles = []
        
        if not PYNVML_AVAILABLE:
            logger.warning("pynvml not available, GPU monitoring disabled")
            return
        
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(self.device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.handles.append(handle)
            
            self.enabled = True
            logger.info(f"GPU monitoring initialized: {self.device_count} device(s)")
        
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
    
    def get_metrics(self, device_index: int = 0) -> Dict:
        """
        Get GPU metrics for specified device.
        
        Args:
            device_index: GPU device index
        
        Returns:
            Dictionary with GPU metrics
        """
        if not self.enabled or device_index >= self.device_count:
            return {}
        
        try:
            handle = self.handles[device_index]
            
            # Device info
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle,
                    pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temp = 0
            
            # Power
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            except:
                power_usage = 0.0
                power_limit = 0.0
            
            # Clock speeds
            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                    handle,
                    pynvml.NVML_CLOCK_GRAPHICS
                )
                clock_memory = pynvml.nvmlDeviceGetClockInfo(
                    handle,
                    pynvml.NVML_CLOCK_MEM
                )
            except:
                clock_graphics = 0
                clock_memory = 0
            
            # Fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except:
                fan_speed = 0
            
            return {
                "device_index": device_index,
                "device_name": name,
                "gpu_util_pct": util.gpu,
                "mem_util_pct": util.memory,
                "mem_used_mb": mem_info.used / (1024 ** 2),
                "mem_total_mb": mem_info.total / (1024 ** 2),
                "mem_free_mb": mem_info.free / (1024 ** 2),
                "temp_c": temp,
                "power_usage_w": power_usage,
                "power_limit_w": power_limit,
                "power_pct": (power_usage / power_limit * 100) if power_limit > 0 else 0.0,
                "clock_graphics_mhz": clock_graphics,
                "clock_memory_mhz": clock_memory,
                "fan_speed_pct": fan_speed,
            }
        
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {e}")
            return {}
    
    def get_all_devices(self) -> List[Dict]:
        """
        Get metrics for all GPU devices.
        
        Returns:
            List of GPU metrics dictionaries
        """
        if not self.enabled:
            return []
        
        metrics = []
        for i in range(self.device_count):
            device_metrics = self.get_metrics(i)
            if device_metrics:
                metrics.append(device_metrics)
        
        return metrics
    
    def shutdown(self):
        """Shutdown GPU monitoring."""
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
                logger.info("GPU monitoring shutdown")
            except:
                pass


class CPUMonitor:
    """
    Monitor CPU and system metrics using psutil.
    Tracks CPU usage, memory, disk I/O, and network traffic.
    """
    
    def __init__(self):
        """Initialize CPU monitor."""
        self.cpu_count = psutil.cpu_count()
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        
        # Initial values for delta calculations
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_time = time.time()
        
        logger.info(
            f"CPU monitoring initialized: {self.cpu_count} cores "
            f"({self.cpu_count_logical} logical)"
        )
    
    def get_cpu_metrics(self) -> Dict:
        """
        Get CPU utilization metrics.
        
        Returns:
            Dictionary with CPU metrics
        """
        try:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Per-core usage
            cpu_percent_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            
            # CPU stats
            cpu_stats = psutil.cpu_stats()
            
            # Load average (Unix-like systems)
            try:
                load_avg = os.getloadavg()
            except AttributeError:
                load_avg = (0.0, 0.0, 0.0)
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_percent_per_core": cpu_percent_per_core,
                "cpu_count": self.cpu_count,
                "cpu_count_logical": self.cpu_count_logical,
                "cpu_freq_current_mhz": cpu_freq.current if cpu_freq else 0.0,
                "cpu_freq_min_mhz": cpu_freq.min if cpu_freq else 0.0,
                "cpu_freq_max_mhz": cpu_freq.max if cpu_freq else 0.0,
                "cpu_ctx_switches": cpu_stats.ctx_switches,
                "cpu_interrupts": cpu_stats.interrupts,
                "load_avg_1m": load_avg[0],
                "load_avg_5m": load_avg[1],
                "load_avg_15m": load_avg[2],
            }
        
        except Exception as e:
            logger.error(f"Error getting CPU metrics: {e}")
            return {}
    
    def get_memory_metrics(self) -> Dict:
        """
        Get memory utilization metrics.
        
        Returns:
            Dictionary with memory metrics
        """
        try:
            # Virtual memory
            mem = psutil.virtual_memory()
            
            # Swap memory
            swap = psutil.swap_memory()
            
            return {
                "mem_total_mb": mem.total / (1024 ** 2),
                "mem_available_mb": mem.available / (1024 ** 2),
                "mem_used_mb": mem.used / (1024 ** 2),
                "mem_free_mb": mem.free / (1024 ** 2),
                "mem_percent": mem.percent,
                "swap_total_mb": swap.total / (1024 ** 2),
                "swap_used_mb": swap.used / (1024 ** 2),
                "swap_free_mb": swap.free / (1024 ** 2),
                "swap_percent": swap.percent,
            }
        
        except Exception as e:
            logger.error(f"Error getting memory metrics: {e}")
            return {}
    
    def get_disk_metrics(self) -> Dict:
        """
        Get disk I/O metrics.
        
        Returns:
            Dictionary with disk metrics
        """
        try:
            # Disk usage
            disk_usage = psutil.disk_usage('/')
            
            # Disk I/O
            current_disk_io = psutil.disk_io_counters()
            current_time = time.time()
            
            time_delta = current_time - self.last_time
            
            if time_delta > 0 and self.last_disk_io:
                read_rate = (
                    (current_disk_io.read_bytes - self.last_disk_io.read_bytes)
                    / time_delta / (1024 ** 2)  # MB/s
                )
                write_rate = (
                    (current_disk_io.write_bytes - self.last_disk_io.write_bytes)
                    / time_delta / (1024 ** 2)  # MB/s
                )
            else:
                read_rate = 0.0
                write_rate = 0.0
            
            self.last_disk_io = current_disk_io
            
            return {
                "disk_total_gb": disk_usage.total / (1024 ** 3),
                "disk_used_gb": disk_usage.used / (1024 ** 3),
                "disk_free_gb": disk_usage.free / (1024 ** 3),
                "disk_percent": disk_usage.percent,
                "disk_read_mb": current_disk_io.read_bytes / (1024 ** 2),
                "disk_write_mb": current_disk_io.write_bytes / (1024 ** 2),
                "disk_read_rate_mbps": read_rate,
                "disk_write_rate_mbps": write_rate,
                "disk_read_count": current_disk_io.read_count,
                "disk_write_count": current_disk_io.write_count,
            }
        
        except Exception as e:
            logger.error(f"Error getting disk metrics: {e}")
            return {}
    
    def get_network_metrics(self) -> Dict:
        """
        Get network I/O metrics.
        
        Returns:
            Dictionary with network metrics
        """
        try:
            current_net_io = psutil.net_io_counters()
            current_time = time.time()
            
            time_delta = current_time - self.last_time
            
            if time_delta > 0 and self.last_net_io:
                recv_rate = (
                    (current_net_io.bytes_recv - self.last_net_io.bytes_recv)
                    / time_delta / (1024 ** 2)  # MB/s
                )
                sent_rate = (
                    (current_net_io.bytes_sent - self.last_net_io.bytes_sent)
                    / time_delta / (1024 ** 2)  # MB/s
                )
            else:
                recv_rate = 0.0
                sent_rate = 0.0
            
            self.last_net_io = current_net_io
            self.last_time = current_time
            
            return {
                "net_bytes_sent_mb": current_net_io.bytes_sent / (1024 ** 2),
                "net_bytes_recv_mb": current_net_io.bytes_recv / (1024 ** 2),
                "net_sent_rate_mbps": sent_rate,
                "net_recv_rate_mbps": recv_rate,
                "net_packets_sent": current_net_io.packets_sent,
                "net_packets_recv": current_net_io.packets_recv,
                "net_errin": current_net_io.errin,
                "net_errout": current_net_io.errout,
                "net_dropin": current_net_io.dropin,
                "net_dropout": current_net_io.dropout,
            }
        
        except Exception as e:
            logger.error(f"Error getting network metrics: {e}")
            return {}
    
    def get_process_metrics(self) -> Dict:
        """
        Get current process metrics.
        
        Returns:
            Dictionary with process metrics
        """
        try:
            process = psutil.Process()
            
            # Memory info
            mem_info = process.memory_info()
            
            # CPU times
            cpu_times = process.cpu_times()
            
            # Threads
            num_threads = process.num_threads()
            
            # File descriptors (Unix)
            try:
                num_fds = process.num_fds()
            except AttributeError:
                num_fds = 0
            
            # Connections
            try:
                connections = len(process.connections())
            except (psutil.AccessDenied, AttributeError):
                connections = 0
            
            return {
                "process_memory_rss_mb": mem_info.rss / (1024 ** 2),
                "process_memory_vms_mb": mem_info.vms / (1024 ** 2),
                "process_cpu_percent": process.cpu_percent(interval=0.1),
                "process_num_threads": num_threads,
                "process_num_fds": num_fds,
                "process_connections": connections,
                "process_cpu_user_time": cpu_times.user,
                "process_cpu_system_time": cpu_times.system,
            }
        
        except Exception as e:
            logger.error(f"Error getting process metrics: {e}")
            return {}


class PerformanceMonitor:
    """
    Main performance monitoring orchestrator.
    Collects metrics from GPU and CPU monitors, stores in DuckDB,
    and exposes metrics via properties and FastAPI endpoints.
    """
    
    def __init__(
        self,
        db_path: str = "data/omergpt.db",
        poll_interval: int = 30,
        history_size: int = 1000,
    ):
        """
        Initialize performance monitor.
        
        Args:
            db_path: Path to DuckDB database
            poll_interval: Polling interval in seconds
            history_size: Number of metrics to keep in memory
        """
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.history_size = history_size
        
        # Initialize monitors
        self.gpu_monitor = GPUMonitor()
        self.cpu_monitor = CPUMonitor()
        
        # Database manager
        self.db_manager = None
        
        # In-memory metrics history
        self.metrics_history = deque(maxlen=history_size)
        
        # Monitoring state
        self.is_running = False
        self.start_time = None
        
        # Statistics
        self.stats = {
            "metrics_collected": 0,
            "errors": 0,
            "last_error": None,
        }
        
        logger.info(
            f"PerformanceMonitor initialized: poll_interval={poll_interval}s"
        )
    
    async def initialize(self):
        """Initialize database and create tables."""
        self.db_manager = DatabaseManager(self.db_path)
        
    
        
        self.start_time = datetime.now()
        
        logger.info("PerformanceMonitor database initialized")
    
    async def collect_metrics(self) -> Dict:
        """
        Collect all system metrics.
        
        Returns:
            Dictionary with all metrics
        """
        timestamp = int(time.time() * 1000)
        
        metrics = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp / 1000).isoformat(),
            "gpu": [],
            "cpu": {},
            "memory": {},
            "disk": {},
            "network": {},
            "process": {},
        }
        
        # GPU metrics
        gpu_devices = self.gpu_monitor.get_all_devices()
        metrics["gpu"] = gpu_devices
        
        # CPU metrics
        metrics["cpu"] = self.cpu_monitor.get_cpu_metrics()
        
        # Memory metrics
        metrics["memory"] = self.cpu_monitor.get_memory_metrics()
        
        # Disk metrics
        metrics["disk"] = self.cpu_monitor.get_disk_metrics()
        
        # Network metrics
        metrics["network"] = self.cpu_monitor.get_network_metrics()
        
        # Process metrics
        metrics["process"] = self.cpu_monitor.get_process_metrics()
        
        # Add to history
        self.metrics_history.append(metrics)
        
        self.stats["metrics_collected"] += 1
        
        return metrics
    
    async def store_metrics(self, metrics: Dict):
        """
        Store metrics in DuckDB.
        
        Args:
            metrics: Metrics dictionary
        """
        timestamp = metrics["timestamp"]
        
        records = []
        
        # GPU metrics
        for gpu in metrics.get("gpu", []):
            device_index = gpu.get("device_index", 0)
            
            for key, value in gpu.items():
                if key in ["device_index", "device_name"]:
                    continue
                
                if isinstance(value, (int, float)):
                    records.append((
                        timestamp,
                        "gpu",
                        key,
                        float(value),
                        device_index,
                    ))
        
        # CPU metrics
        for key, value in metrics.get("cpu", {}).items():
            if key == "cpu_percent_per_core":
                continue
            
            if isinstance(value, (int, float)):
                records.append((
                    timestamp,
                    "cpu",
                    key,
                    float(value),
                    0,
                ))
        
        # Memory metrics
        for key, value in metrics.get("memory", {}).items():
            if isinstance(value, (int, float)):
                records.append((
                    timestamp,
                    "memory",
                    key,
                    float(value),
                    0,
                ))
        
        # Disk metrics
        for key, value in metrics.get("disk", {}).items():
            if isinstance(value, (int, float)):
                records.append((
                    timestamp,
                    "disk",
                    key,
                    float(value),
                    0,
                ))
        
        # Network metrics
        for key, value in metrics.get("network", {}).items():
            if isinstance(value, (int, float)):
                records.append((
                    timestamp,
                    "network",
                    key,
                    float(value),
                    0,
                ))
        
        # Process metrics
        for key, value in metrics.get("process", {}).items():
            if isinstance(value, (int, float)):
                records.append((
                    timestamp,
                    "process",
                    key,
                    float(value),
                    0,
                ))
        
        # Batch insert
        
    
    async def monitor_loop(self):
        """
        Main monitoring loop.
        Collects and stores metrics every poll_interval seconds.
        """
        self.is_running = True
        
        logger.info(f"Starting monitoring loop (interval: {self.poll_interval}s)")
        
        while self.is_running:
            try:
                # Collect metrics
                metrics = await self.collect_metrics()
                
                # Store in database
                await self.store_metrics(metrics)
                
                # Log summary
                gpu_util = (
                    metrics["gpu"][0]["gpu_util_pct"]
                    if metrics.get("gpu") else 0.0
                )
                cpu_util = metrics.get("cpu", {}).get("cpu_percent", 0.0)
                mem_util = metrics.get("memory", {}).get("mem_percent", 0.0)
                
                logger.debug(
                    f"Metrics collected | "
                    f"GPU: {gpu_util:.1f}% | "
                    f"CPU: {cpu_util:.1f}% | "
                    f"RAM: {mem_util:.1f}%"
                )
                
                # Wait for next interval
                await asyncio.sleep(self.poll_interval)
            
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stats["errors"] += 1
                self.stats["last_error"] = str(e)
                
                # Wait before retrying
                await asyncio.sleep(5)
        
        self.is_running = False
        logger.info("Monitoring loop stopped")
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.is_running = False
    
    def get_latest_metrics(self) -> Optional[Dict]:
        """
        Get most recent metrics.
        
        Returns:
            Latest metrics dictionary or None
        """
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_summary(self) -> Dict:
        """
        Get summary statistics from recent metrics.
        
        Returns:
            Summary statistics dictionary
        """
        if not self.metrics_history:
            return {}
        
        # Extract key metrics over history
        gpu_utils = []
        cpu_utils = []
        mem_utils = []
        
        for metrics in self.metrics_history:
            if metrics.get("gpu"):
                gpu_utils.append(metrics["gpu"][0].get("gpu_util_pct", 0.0))
            
            cpu_utils.append(metrics.get("cpu", {}).get("cpu_percent", 0.0))
            mem_utils.append(metrics.get("memory", {}).get("mem_percent", 0.0))
        
        import numpy as np
        
        summary = {
            "window_size": len(self.metrics_history),
        }
        
        if gpu_utils:
            summary["gpu_util"] = {
                "current": gpu_utils[-1],
                "mean": float(np.mean(gpu_utils)),
                "max": float(np.max(gpu_utils)),
                "min": float(np.min(gpu_utils)),
            }
        
        if cpu_utils:
            summary["cpu_util"] = {
                "current": cpu_utils[-1],
                "mean": float(np.mean(cpu_utils)),
                "max": float(np.max(cpu_utils)),
                "min": float(np.min(cpu_utils)),
            }
        
        if mem_utils:
            summary["mem_util"] = {
                "current": mem_utils[-1],
                "mean": float(np.mean(mem_utils)),
                "max": float(np.max(mem_utils)),
                "min": float(np.min(mem_utils)),
            }
        
        return summary
    
    def get_uptime(self) -> Dict:
        """
        Get system uptime information.
        
        Returns:
            Uptime dictionary
        """
        if not self.start_time:
            return {"uptime_seconds": 0}
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "start_time": self.start_time.isoformat(),
            "current_time": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "uptime_days": uptime / 86400,
        }
    
    def get_status(self) -> Dict:
        """
        Get overall monitoring status.
        
        Returns:
            Status dictionary
        """
        latest = self.get_latest_metrics()
        summary = self.get_metrics_summary()
        uptime = self.get_uptime()
        
        return {
            "is_running": self.is_running,
            "uptime": uptime,
            "stats": self.stats,
            "latest_metrics": latest,
            "summary": summary,
            "system_info": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "cpu_count": self.cpu_monitor.cpu_count,
                "gpu_count": self.gpu_monitor.device_count,
            },
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.stop_monitoring()
        
        if self.gpu_monitor:
            self.gpu_monitor.shutdown()
        
        if self.db_manager:
            await self.db_manager.close()
        
        logger.info("PerformanceMonitor cleanup complete")


# FastAPI application
if FASTAPI_AVAILABLE:
    app = FastAPI(title="omerGPT Performance Monitor API")
    
    # Global monitor instance
    monitor_instance: Optional[PerformanceMonitor] = None
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize monitor on startup."""
        global monitor_instance
        monitor_instance = PerformanceMonitor()
        await monitor_instance.initialize()
        
        # Start monitoring loop in background
        asyncio.create_task(monitor_instance.monitor_loop())
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        global monitor_instance
        if monitor_instance:
            await monitor_instance.cleanup()
    
    @app.get("/status")
    async def get_status():
        """Get overall system status."""
        if not monitor_instance:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        status = monitor_instance.get_status()
        return JSONResponse(content=status)
    
    @app.get("/metrics/latest")
    async def get_latest_metrics():
        """Get latest metrics."""
        if not monitor_instance:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        metrics = monitor_instance.get_latest_metrics()
        
        if not metrics:
            raise HTTPException(status_code=404, detail="No metrics available")
        
        return JSONResponse(content=metrics)
    
    @app.get("/metrics/summary")
    async def get_metrics_summary():
        """Get metrics summary."""
        if not monitor_instance:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        summary = monitor_instance.get_metrics_summary()
        return JSONResponse(content=summary)
    
    @app.get("/uptime")
    async def get_uptime():
        """Get uptime information."""
        if not monitor_instance:
            raise HTTPException(status_code=503, detail="Monitor not initialized")
        
        uptime = monitor_instance.get_uptime()
        return JSONResponse(content=uptime)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if not monitor_instance or not monitor_instance.is_running:
            raise HTTPException(status_code=503, detail="Monitor not running")
        
        return {"status": "healthy"}


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_performance_monitor():
        """Test performance monitor."""
        print("Testing PerformanceMonitor...")
        print(f"pynvml available: {PYNVML_AVAILABLE}")
        print(f"FastAPI available: {FASTAPI_AVAILABLE}\n")
        
        # Initialize monitor
        monitor = PerformanceMonitor(
            db_path="data/omergpt.db",
            poll_interval=5,  # 5 seconds for testing
            history_size=100,
        )
        
        await monitor.initialize()
        
        print("1. Testing GPU monitoring...")
        if monitor.gpu_monitor.enabled:
            gpu_metrics = monitor.gpu_monitor.get_all_devices()
            print(f"   Found {len(gpu_metrics)} GPU(s)")
            
            for gpu in gpu_metrics:
                print(f"   GPU {gpu['device_index']}: {gpu['device_name']}")
                print(f"     Utilization: {gpu['gpu_util_pct']:.1f}%")
                print(f"     Memory: {gpu['mem_used_mb']:.0f} / {gpu['mem_total_mb']:.0f} MB")
                print(f"     Temperature: {gpu['temp_c']}°C")
                print(f"     Power: {gpu['power_usage_w']:.1f} W")
        else:
            print("   GPU monitoring not available")
        print()
        
        print("2. Testing CPU monitoring...")
        cpu_metrics = monitor.cpu_monitor.get_cpu_metrics()
        print(f"   CPU Usage: {cpu_metrics['cpu_percent']:.1f}%")
        print(f"   CPU Cores: {cpu_metrics['cpu_count']} physical, {cpu_metrics['cpu_count_logical']} logical")
        print(f"   CPU Frequency: {cpu_metrics['cpu_freq_current_mhz']:.0f} MHz")
        print()
        
        print("3. Testing memory monitoring...")
        mem_metrics = monitor.cpu_monitor.get_memory_metrics()
        print(f"   RAM Usage: {mem_metrics['mem_percent']:.1f}%")
        print(f"   RAM Used: {mem_metrics['mem_used_mb']:.0f} / {mem_metrics['mem_total_mb']:.0f} MB")
        print(f"   Swap Usage: {mem_metrics['swap_percent']:.1f}%")
        print()
        
        print("4. Testing disk monitoring...")
        disk_metrics = monitor.cpu_monitor.get_disk_metrics()
        print(f"   Disk Usage: {disk_metrics['disk_percent']:.1f}%")
        print(f"   Disk Used: {disk_metrics['disk_used_gb']:.1f} / {disk_metrics['disk_total_gb']:.1f} GB")
        print(f"   Disk Read Rate: {disk_metrics['disk_read_rate_mbps']:.2f} MB/s")
        print(f"   Disk Write Rate: {disk_metrics['disk_write_rate_mbps']:.2f} MB/s")
        print()
        
        print("5. Testing network monitoring...")
        net_metrics = monitor.cpu_monitor.get_network_metrics()
        print(f"   Network Sent: {net_metrics['net_bytes_sent_mb']:.2f} MB")
        print(f"   Network Received: {net_metrics['net_bytes_recv_mb']:.2f} MB")
        print(f"   Send Rate: {net_metrics['net_sent_rate_mbps']:.2f} MB/s")
        print(f"   Receive Rate: {net_metrics['net_recv_rate_mbps']:.2f} MB/s")
        print()
        
        print("6. Testing process monitoring...")
        proc_metrics = monitor.cpu_monitor.get_process_metrics()
        print(f"   Process Memory: {proc_metrics['process_memory_rss_mb']:.1f} MB")
        print(f"   Process CPU: {proc_metrics['process_cpu_percent']:.1f}%")
        print(f"   Process Threads: {proc_metrics['process_num_threads']}")
        print()
        
        print("7. Testing metrics collection and storage...")
        for i in range(3):
            print(f"   Collecting metrics (iteration {i+1})...")
            metrics = await monitor.collect_metrics()
            await monitor.store_metrics(metrics)
            await asyncio.sleep(2)
        print()
        
        print("8. Testing metrics history...")
        latest = monitor.get_latest_metrics()
        print(f"   Latest timestamp: {latest['datetime']}")
        print(f"   History size: {len(monitor.metrics_history)}")
        print()
        
        print("9. Testing metrics summary...")
        summary = monitor.get_metrics_summary()
        print(f"   Summary window: {summary['window_size']} samples")
        
        if "gpu_util" in summary:
            gpu_sum = summary["gpu_util"]
            print(f"   GPU util: current={gpu_sum['current']:.1f}%, avg={gpu_sum['mean']:.1f}%")
        
        cpu_sum = summary["cpu_util"]
        print(f"   CPU util: current={cpu_sum['current']:.1f}%, avg={cpu_sum['mean']:.1f}%")
        
        mem_sum = summary["mem_util"]
        print(f"   RAM util: current={mem_sum['current']:.1f}%, avg={mem_sum['mean']:.1f}%")
        print()
        
        print("10. Testing uptime...")
        uptime = monitor.get_uptime()
        print(f"   Start time: {uptime['start_time']}")
        print(f"   Uptime: {uptime['uptime_seconds']:.1f} seconds")
        print()
        
        print("11. Testing status endpoint...")
        status = monitor.get_status()
        print(f"   Running: {status['is_running']}")
        print(f"   Metrics collected: {status['stats']['metrics_collected']}")
        print(f"   Platform: {status['system_info']['platform']}")
        print()
        
        print("12. Running monitoring loop for 15 seconds...")
        monitor_task = asyncio.create_task(monitor.monitor_loop())
        
        await asyncio.sleep(15)
        
        monitor.stop_monitoring()
        
        try:
            await asyncio.wait_for(monitor_task, timeout=2)
        except asyncio.TimeoutError:
            monitor_task.cancel()
        
        print("   Monitoring loop stopped")
        print()
        
        print("13. Final statistics...")
        final_stats = monitor.get_status()
        print(f"   Total metrics collected: {final_stats['stats']['metrics_collected']}")
        print(f"   Errors: {final_stats['stats']['errors']}")
        print()
        
        # Cleanup
        await monitor.cleanup()
        
        print("Test completed successfully!")
        
        if FASTAPI_AVAILABLE:
            print("\nTo run the FastAPI server:")
            print("  python src/system_monitor/performance_monitor.py --server")
            print("  Then visit: http://localhost:8000/status")
    
    # Check for server mode
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        if FASTAPI_AVAILABLE:
            print("Starting FastAPI server on http://localhost:8000")
            print("Endpoints:")
            print("  GET /status - Overall system status")
            print("  GET /metrics/latest - Latest metrics")
            print("  GET /metrics/summary - Metrics summary")
            print("  GET /uptime - Uptime information")
            print("  GET /health - Health check")
            print("\nPress Ctrl+C to stop")
            
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        else:
            print("FastAPI not available. Install: pip install fastapi uvicorn")
    else:
        asyncio.run(test_performance_monitor())
