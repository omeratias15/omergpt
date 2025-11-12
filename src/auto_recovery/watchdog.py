"""
src/auto_recovery/watchdog.py
Automatic monitoring and recovery system for omerGPT core services.
Detects stalled processes, exceptions, and unresponsive components.
Implements graceful restart with exponential backoff and circuit breaker pattern.
Exposes FastAPI health check endpoints and logs recovery events.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import asyncio
import logging
import os
import sys
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

logger = logging.getLogger("omerGPT.auto_recovery.watchdog")


class ServiceState(Enum):
    """Service health states."""
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    STOPPED = "stopped"


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures exceeded threshold, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for fault tolerance.
    Prevents cascading failures by temporarily disabling failed services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            success_threshold: Successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state_changed_time = datetime.now()
        
        logger.info(
            f"CircuitBreaker initialized: threshold={failure_threshold}, "
            f"timeout={recovery_timeout}s"
        )
    
    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                self._transition_to_closed()
        
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._transition_to_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            self._transition_to_open()
    
    def can_execute(self) -> bool:
        """
        Check if operation can be executed.
        
        Returns:
            True if operation allowed
        """
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                
                if elapsed >= self.recovery_timeout:
                    self._transition_to_half_open()
                    return True
            
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info("Circuit breaker: CLOSED (service recovered)")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_changed_time = datetime.now()
    
    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(
            f"Circuit breaker: OPEN (failures: {self.failure_count})"
        )
        self.state = CircuitState.OPEN
        self.success_count = 0
        self.state_changed_time = datetime.now()
    
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        logger.info("Circuit breaker: HALF_OPEN (testing recovery)")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        self.state_changed_time = datetime.now()
    
    def get_state(self) -> Dict:
        """
        Get circuit breaker state.
        
        Returns:
            State dictionary
        """
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "state_changed": self.state_changed_time.isoformat(),
        }


class ServiceMonitor:
    """
    Monitor for individual service health and heartbeat.
    Tracks execution time, errors, and unresponsive behavior.
    """
    
    def __init__(
        self,
        service_name: str,
        heartbeat_interval: int = 60,
        timeout: int = 120,
    ):
        """
        Initialize service monitor.
        
        Args:
            service_name: Name of the service
            heartbeat_interval: Expected heartbeat interval (seconds)
            timeout: Timeout before considering service stalled (seconds)
        """
        self.service_name = service_name
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        
        self.state = ServiceState.STOPPED
        self.last_heartbeat = None
        self.last_error = None
        self.error_count = 0
        self.restart_count = 0
        self.start_time = None
        
        # Metrics
        self.execution_times = deque(maxlen=100)
        self.error_history = deque(maxlen=50)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=2,
        )
        
        logger.info(
            f"ServiceMonitor initialized: {service_name} "
            f"(timeout={timeout}s)"
        )
    
    def heartbeat(self):
        """Record heartbeat from service."""
        self.last_heartbeat = datetime.now()
        
        if self.state == ServiceState.STARTING:
            self.state = ServiceState.HEALTHY
            logger.info(f"Service {self.service_name}: HEALTHY")
        
        elif self.state == ServiceState.DEGRADED:
            self.state = ServiceState.HEALTHY
            logger.info(f"Service {self.service_name}: recovered to HEALTHY")
        
        self.circuit_breaker.record_success()
    
    def record_error(self, error: Exception):
        """
        Record service error.
        
        Args:
            error: Exception that occurred
        """
        self.error_count += 1
        self.last_error = {
            "timestamp": datetime.now().isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        
        self.error_history.append(self.last_error)
        self.circuit_breaker.record_failure()
        
        if self.state == ServiceState.HEALTHY:
            self.state = ServiceState.DEGRADED
        
        logger.error(
            f"Service {self.service_name}: error recorded "
            f"({type(error).__name__}: {error})"
        )
    
    def record_execution_time(self, duration: float):
        """
        Record service execution time.
        
        Args:
            duration: Execution duration in seconds
        """
        self.execution_times.append(duration)
    
    def is_stalled(self) -> bool:
        """
        Check if service is stalled (no heartbeat within timeout).
        
        Returns:
            True if service is stalled
        """
        if not self.last_heartbeat:
            return False
        
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        
        if elapsed > self.timeout:
            if self.state not in [ServiceState.FAILED, ServiceState.RECOVERING]:
                self.state = ServiceState.FAILED
                logger.warning(
                    f"Service {self.service_name}: STALLED "
                    f"(no heartbeat for {elapsed:.0f}s)"
                )
            return True
        
        return False
    
    def can_restart(self) -> bool:
        """
        Check if service can be restarted.
        
        Returns:
            True if restart allowed by circuit breaker
        """
        return self.circuit_breaker.can_execute()
    
    def mark_restarting(self):
        """Mark service as restarting."""
        self.state = ServiceState.RECOVERING
        self.restart_count += 1
        self.start_time = datetime.now()
        logger.info(f"Service {self.service_name}: RECOVERING (restart #{self.restart_count})")
    
    def mark_stopped(self):
        """Mark service as stopped."""
        self.state = ServiceState.STOPPED
        self.last_heartbeat = None
        logger.info(f"Service {self.service_name}: STOPPED")
    
    def get_status(self) -> Dict:
        """
        Get service status.
        
        Returns:
            Status dictionary
        """
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        avg_exec_time = None
        if self.execution_times:
            avg_exec_time = sum(self.execution_times) / len(self.execution_times)
        
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "uptime_seconds": uptime,
            "error_count": self.error_count,
            "restart_count": self.restart_count,
            "last_error": self.last_error,
            "avg_execution_time": avg_exec_time,
            "circuit_breaker": self.circuit_breaker.get_state(),
        }


class Watchdog:
    """
    Main watchdog orchestrator for system-wide health monitoring and recovery.
    Manages multiple service monitors and coordinates restart operations.
    """
    
    def __init__(
        self,
        check_interval: int = 10,
        log_file: str = "auto_recovery.log",
    ):
        """
        Initialize watchdog.
        
        Args:
            check_interval: Health check interval (seconds)
            log_file: Recovery event log file
        """
        self.check_interval = check_interval
        self.log_file = log_file
        
        # Service monitors
        self.monitors: Dict[str, ServiceMonitor] = {}
        
        # Service tasks
        self.service_tasks: Dict[str, asyncio.Task] = {}
        self.service_coroutines: Dict[str, Callable] = {}
        
        # Watchdog state
        self.is_running = False
        self.start_time = None
        
        # Recovery statistics
        self.stats = {
            "checks_performed": 0,
            "restarts_triggered": 0,
            "recovery_successes": 0,
            "recovery_failures": 0,
        }
        
        # Event log
        self.event_log = deque(maxlen=1000)
        
        # Setup file logging
        self._setup_file_logging()
        
        logger.info(f"Watchdog initialized: check_interval={check_interval}s")
    
    def _setup_file_logging(self):
        """Setup file logging for recovery events."""
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    def register_service(
        self,
        service_name: str,
        coroutine: Callable,
        heartbeat_interval: int = 60,
        timeout: int = 120,
    ):
        """
        Register a service for monitoring.
        
        Args:
            service_name: Unique service identifier
            coroutine: Async coroutine to run
            heartbeat_interval: Expected heartbeat interval
            timeout: Stall detection timeout
        """
        monitor = ServiceMonitor(
            service_name=service_name,
            heartbeat_interval=heartbeat_interval,
            timeout=timeout,
        )
        
        self.monitors[service_name] = monitor
        self.service_coroutines[service_name] = coroutine
        
        logger.info(f"Registered service: {service_name}")
    
    async def start_service(self, service_name: str) -> bool:
        """
        Start a monitored service.
        
        Args:
            service_name: Service to start
        
        Returns:
            True if started successfully
        """
        if service_name not in self.monitors:
            logger.error(f"Service not registered: {service_name}")
            return False
        
        monitor = self.monitors[service_name]
        
        # Check circuit breaker
        if not monitor.can_restart():
            logger.warning(
                f"Service {service_name}: restart blocked by circuit breaker "
                f"(state: {monitor.circuit_breaker.state.value})"
            )
            return False
        
        # Cancel existing task if running
        if service_name in self.service_tasks:
            task = self.service_tasks[service_name]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Start new task
        monitor.mark_restarting()
        
        coroutine = self.service_coroutines[service_name]
        task = asyncio.create_task(self._run_service(service_name, coroutine))
        
        self.service_tasks[service_name] = task
        
        self._log_event({
            "type": "service_started",
            "service": service_name,
            "restart_count": monitor.restart_count,
        })
        
        logger.info(f"Started service: {service_name}")
        
        return True
    
    async def _run_service(self, service_name: str, coroutine: Callable):
        """
        Run service with error handling and monitoring.
        
        Args:
            service_name: Service name
            coroutine: Service coroutine
        """
        monitor = self.monitors[service_name]
        
        try:
            start_time = time.time()
            
            # Run service coroutine with watchdog context
            await coroutine(self, service_name)
            
            duration = time.time() - start_time
            monitor.record_execution_time(duration)
            
            logger.info(f"Service {service_name} completed (duration: {duration:.2f}s)")
        
        except asyncio.CancelledError:
            logger.info(f"Service {service_name} cancelled")
            monitor.mark_stopped()
        
        except Exception as e:
            logger.error(f"Service {service_name} failed: {e}")
            logger.error(traceback.format_exc())
            
            monitor.record_error(e)
            
            self._log_event({
                "type": "service_error",
                "service": service_name,
                "error": str(e),
            })
    
    async def stop_service(self, service_name: str):
        """
        Stop a running service.
        
        Args:
            service_name: Service to stop
        """
        if service_name not in self.service_tasks:
            return
        
        task = self.service_tasks[service_name]
        
        if not task.done():
            task.cancel()
            
            try:
                await asyncio.wait_for(task, timeout=5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        monitor = self.monitors.get(service_name)
        if monitor:
            monitor.mark_stopped()
        
        self._log_event({
            "type": "service_stopped",
            "service": service_name,
        })
        
        logger.info(f"Stopped service: {service_name}")
    
    async def restart_service(self, service_name: str) -> bool:
        """
        Restart a service (stop + start).
        
        Args:
            service_name: Service to restart
        
        Returns:
            True if restarted successfully
        """
        logger.info(f"Restarting service: {service_name}")
        
        await self.stop_service(service_name)
        await asyncio.sleep(2)  # Grace period
        
        success = await self.start_service(service_name)
        
        if success:
            self.stats["restarts_triggered"] += 1
            self.stats["recovery_successes"] += 1
        else:
            self.stats["recovery_failures"] += 1
        
        self._log_event({
            "type": "service_restart",
            "service": service_name,
            "success": success,
        })
        
        return success
    
    def heartbeat(self, service_name: str):
        """
        Receive heartbeat from service.
        
        Args:
            service_name: Service sending heartbeat
        """
        monitor = self.monitors.get(service_name)
        
        if monitor:
            monitor.heartbeat()
    
    async def check_services(self):
        """Check health of all monitored services."""
        self.stats["checks_performed"] += 1
        
        for service_name, monitor in self.monitors.items():
            # Check if stalled
            if monitor.is_stalled():
                logger.warning(f"Service {service_name} stalled, attempting restart")
                await self.restart_service(service_name)
            
            # Check if task crashed
            task = self.service_tasks.get(service_name)
            
            if task and task.done() and monitor.state != ServiceState.STOPPED:
                # Task finished unexpectedly
                try:
                    exception = task.exception()
                    if exception:
                        logger.error(
                            f"Service {service_name} crashed: {exception}"
                        )
                        monitor.record_error(exception)
                except asyncio.CancelledError:
                    pass
                
                # Attempt restart
                if monitor.can_restart():
                    logger.info(f"Restarting crashed service: {service_name}")
                    await self.restart_service(service_name)
    
    async def watchdog_loop(self):
        """Main watchdog monitoring loop."""
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info(f"Starting watchdog loop (interval: {self.check_interval}s)")
        
        while self.is_running:
            try:
                await self.check_services()
                await asyncio.sleep(self.check_interval)
            
            except asyncio.CancelledError:
                logger.info("Watchdog loop cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in watchdog loop: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(5)
        
        self.is_running = False
        logger.info("Watchdog loop stopped")
    
    def stop_watchdog(self):
        """Stop the watchdog loop."""
        self.is_running = False
    
    async def start_all_services(self):
        """Start all registered services."""
        for service_name in self.monitors.keys():
            await self.start_service(service_name)
    
    async def stop_all_services(self):
        """Stop all running services."""
        for service_name in list(self.service_tasks.keys()):
            await self.stop_service(service_name)
    
    def _log_event(self, event: Dict):
        """
        Log recovery event.
        
        Args:
            event: Event dictionary
        """
        event["timestamp"] = datetime.now().isoformat()
        self.event_log.append(event)
    
    def get_status(self) -> Dict:
        """
        Get overall watchdog status.
        
        Returns:
            Status dictionary
        """
        service_statuses = {
            name: monitor.get_status()
            for name, monitor in self.monitors.items()
        }
        
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "is_running": self.is_running,
            "uptime_seconds": uptime,
            "stats": self.stats,
            "services": service_statuses,
            "recent_events": list(self.event_log)[-20:],  # Last 20 events
        }
    
    async def cleanup(self):
        """Cleanup all services and resources."""
        self.stop_watchdog()
        await self.stop_all_services()
        logger.info("Watchdog cleanup complete")


# FastAPI application
if FASTAPI_AVAILABLE:
    app = FastAPI(title="omerGPT Watchdog Health API")
    
    # Global watchdog instance
    watchdog_instance: Optional[Watchdog] = None
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize watchdog on startup."""
        global watchdog_instance
        watchdog_instance = Watchdog(check_interval=10)
        
        # Register example services (would be replaced with real services)
        # watchdog_instance.register_service("ingestion", example_service)
        
        # Start watchdog loop
        asyncio.create_task(watchdog_instance.watchdog_loop())
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        global watchdog_instance
        if watchdog_instance:
            await watchdog_instance.cleanup()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        if not watchdog_instance or not watchdog_instance.is_running:
            raise HTTPException(status_code=503, detail="Watchdog not running")
        
        status = watchdog_instance.get_status()
        
        # Check if any services are failed
        failed_services = [
            name for name, svc in status["services"].items()
            if svc["state"] == "failed"
        ]
        
        if failed_services:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "failed_services": failed_services,
                }
            )
        
        return {"status": "healthy"}
    
    @app.get("/status")
    async def get_status():
        """Get overall watchdog status."""
        if not watchdog_instance:
            raise HTTPException(status_code=503, detail="Watchdog not initialized")
        
        status = watchdog_instance.get_status()
        return JSONResponse(content=status)
    
    @app.get("/services")
    async def list_services():
        """List all monitored services."""
        if not watchdog_instance:
            raise HTTPException(status_code=503, detail="Watchdog not initialized")
        
        services = {
            name: monitor.get_status()
            for name, monitor in watchdog_instance.monitors.items()
        }
        
        return JSONResponse(content=services)
    
    @app.post("/services/{service_name}/restart")
    async def restart_service(service_name: str):
        """Restart a specific service."""
        if not watchdog_instance:
            raise HTTPException(status_code=503, detail="Watchdog not initialized")
        
        if service_name not in watchdog_instance.monitors:
            raise HTTPException(status_code=404, detail=f"Service not found: {service_name}")
        
        success = await watchdog_instance.restart_service(service_name)
        
        return {
            "service": service_name,
            "restart_triggered": success,
        }
    
    @app.post("/services/{service_name}/heartbeat")
    async def service_heartbeat(service_name: str):
        """Record heartbeat from service."""
        if not watchdog_instance:
            raise HTTPException(status_code=503, detail="Watchdog not initialized")
        
        if service_name not in watchdog_instance.monitors:
            raise HTTPException(status_code=404, detail=f"Service not found: {service_name}")
        
        watchdog_instance.heartbeat(service_name)
        
        return {"status": "heartbeat_recorded"}


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_watchdog():
        """Test watchdog with simulated services."""
        print("Testing Watchdog...")
        print(f"FastAPI available: {FASTAPI_AVAILABLE}\n")
        
        # Example service coroutines
        async def healthy_service(watchdog: Watchdog, service_name: str):
            """Simulated healthy service."""
            for i in range(10):
                watchdog.heartbeat(service_name)
                await asyncio.sleep(1)
        
        async def flaky_service(watchdog: Watchdog, service_name: str):
            """Simulated flaky service that fails occasionally."""
            for i in range(5):
                watchdog.heartbeat(service_name)
                await asyncio.sleep(1)
                
                if i == 3:
                    raise RuntimeError("Simulated service failure")
        
        async def stalling_service(watchdog: Watchdog, service_name: str):
            """Simulated service that stalls (no heartbeat)."""
            watchdog.heartbeat(service_name)
            await asyncio.sleep(100)  # Stall
        
        # Initialize watchdog
        watchdog = Watchdog(check_interval=5)
        
        print("1. Registering services...")
        watchdog.register_service(
            "healthy_service",
            healthy_service,
            heartbeat_interval=5,
            timeout=10,
        )
        watchdog.register_service(
            "flaky_service",
            flaky_service,
            heartbeat_interval=5,
            timeout=10,
        )
        watchdog.register_service(
            "stalling_service",
            stalling_service,
            heartbeat_interval=5,
            timeout=15,
        )
        print(f"   Registered {len(watchdog.monitors)} services\n")
        
        print("2. Testing circuit breaker...")
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5)
        print(f"   Initial state: {cb.state.value}")
        
        # Record failures
        for i in range(4):
            cb.record_failure()
            print(f"   After failure {i+1}: {cb.state.value}")
        
        print(f"   Can execute: {cb.can_execute()}")
        
        await asyncio.sleep(6)
        print(f"   After timeout: {cb.state.value}")
        print(f"   Can execute: {cb.can_execute()}")
        
        cb.record_success()
        cb.record_success()
        print(f"   After 2 successes: {cb.state.value}\n")
        
        print("3. Starting services...")
        await watchdog.start_service("healthy_service")
        await asyncio.sleep(1)
        await watchdog.start_service("flaky_service")
        await asyncio.sleep(1)
        await watchdog.start_service("stalling_service")
        print()
        
        print("4. Starting watchdog loop...")
        watchdog_task = asyncio.create_task(watchdog.watchdog_loop())
        
        print("   Monitoring for 20 seconds...\n")
        await asyncio.sleep(20)
        
        print("5. Checking service statuses...")
        status = watchdog.get_status()
        
        print(f"   Watchdog uptime: {status['uptime_seconds']:.1f}s")
        print(f"   Checks performed: {status['stats']['checks_performed']}")
        print(f"   Restarts triggered: {status['stats']['restarts_triggered']}")
        print()
        
        for service_name, svc_status in status["services"].items():
            print(f"   Service: {service_name}")
            print(f"     State: {svc_status['state']}")
            print(f"     Errors: {svc_status['error_count']}")
            print(f"     Restarts: {svc_status['restart_count']}")
            print(f"     Circuit breaker: {svc_status['circuit_breaker']['state']}")
        print()
        
        print("6. Recent events:")
        for event in status["recent_events"][-5:]:
            print(f"   {event['timestamp']}: {event['type']} - {event.get('service', 'N/A')}")
        print()
        
        print("7. Manually restarting a service...")
        success = await watchdog.restart_service("healthy_service")
        print(f"   Restart successful: {success}\n")
        
        print("8. Stopping watchdog...")
        watchdog.stop_watchdog()
        
        try:
            await asyncio.wait_for(watchdog_task, timeout=2)
        except asyncio.TimeoutError:
            watchdog_task.cancel()
        
        await watchdog.cleanup()
        print("   Watchdog stopped\n")
        
        print("Test completed successfully!")
        
        if FASTAPI_AVAILABLE:
            print("\nTo run the FastAPI server:")
            print("  python src/auto_recovery/watchdog.py --server")
            print("  Then visit: http://localhost:8001/health")
    
    # Check for server mode
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        if FASTAPI_AVAILABLE:
            print("Starting FastAPI server on http://localhost:8001")
            print("Endpoints:")
            print("  GET /health - Health check")
            print("  GET /status - Overall watchdog status")
            print("  GET /services - List all services")
            print("  POST /services/{name}/restart - Restart service")
            print("  POST /services/{name}/heartbeat - Record heartbeat")
            print("\nPress Ctrl+C to stop")
            
            uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
        else:
            print("FastAPI not available. Install: pip install fastapi uvicorn")
    else:
        asyncio.run(test_watchdog())
