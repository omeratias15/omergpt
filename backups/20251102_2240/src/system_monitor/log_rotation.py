"""
src/system_monitor/log_rotation.py
Automated log file rotation and archival system for omerGPT.
Rotates logs > 50 MB, compresses to logs/archive/, maintains 7-day retention.
Supports scheduled rotation, manual rotation, and cleanup of old archives.
Integrates with Python logging for seamless log management.
"""
import asyncio
import gzip
import logging
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("omerGPT.system_monitor.log_rotation")


class LogRotator:
    """
    Log file rotation manager with compression and archival.
    Handles size-based and time-based rotation policies.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        archive_dir: str = "logs/archive",
        max_size_mb: float = 50.0,
        retention_days: int = 7,
        compress: bool = True,
    ):
        """
        Initialize log rotator.
        
        Args:
            log_dir: Directory containing log files
            archive_dir: Directory for archived logs
            max_size_mb: Maximum log file size before rotation (MB)
            retention_days: Days to retain archived logs
            compress: Whether to compress archived logs
        """
        self.log_dir = log_dir
        self.archive_dir = archive_dir
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.retention_days = retention_days
        self.compress = compress
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(archive_dir, exist_ok=True)
        
        # Statistics
        self.stats = {
            "rotations_performed": 0,
            "files_archived": 0,
            "files_deleted": 0,
            "bytes_compressed": 0,
            "last_rotation": None,
            "last_cleanup": None,
        }
        
        logger.info(
            f"LogRotator initialized: max_size={max_size_mb}MB, "
            f"retention={retention_days}days"
        )
    
    def get_log_files(self) -> List[str]:
        """
        Get all log files in log directory.
        
        Returns:
            List of log file paths
        """
        log_files = []
        
        for filename in os.listdir(self.log_dir):
            filepath = os.path.join(self.log_dir, filename)
            
            # Skip directories and non-log files
            if os.path.isdir(filepath):
                continue
            
            if filename.endswith(('.log', '.txt')):
                log_files.append(filepath)
        
        return log_files
    
    def get_file_size(self, filepath: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            filepath: Path to file
        
        Returns:
            File size in bytes
        """
        try:
            return os.path.getsize(filepath)
        except OSError:
            return 0
    
    def should_rotate(self, filepath: str) -> bool:
        """
        Check if file should be rotated based on size.
        
        Args:
            filepath: Path to log file
        
        Returns:
            True if file should be rotated
        """
        size = self.get_file_size(filepath)
        return size >= self.max_size_bytes
    
    def generate_archive_name(self, log_filename: str) -> str:
        """
        Generate archive filename with timestamp.
        
        Args:
            log_filename: Original log filename
        
        Returns:
            Archive filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(log_filename)[0]
        extension = ".log.gz" if self.compress else ".log"
        
        return f"{base_name}_{timestamp}{extension}"
    
    def compress_file(self, input_path: str, output_path: str) -> int:
        """
        Compress file using gzip.
        
        Args:
            input_path: Input file path
            output_path: Output compressed file path
        
        Returns:
            Number of bytes saved by compression
        """
        input_size = os.path.getsize(input_path)
        
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=9) as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        output_size = os.path.getsize(output_path)
        bytes_saved = input_size - output_size
        
        logger.info(
            f"Compressed: {os.path.basename(input_path)} "
            f"({input_size / (1024**2):.2f} MB -> {output_size / (1024**2):.2f} MB, "
            f"saved {bytes_saved / (1024**2):.2f} MB)"
        )
        
        return bytes_saved
    
    def rotate_file(self, filepath: str) -> bool:
        """
        Rotate a single log file.
        
        Args:
            filepath: Path to log file to rotate
        
        Returns:
            True if rotation successful
        """
        try:
            filename = os.path.basename(filepath)
            archive_name = self.generate_archive_name(filename)
            archive_path = os.path.join(self.archive_dir, archive_name)
            
            logger.info(f"Rotating log file: {filename}")
            
            # Compress and move to archive
            if self.compress:
                bytes_saved = self.compress_file(filepath, archive_path)
                self.stats["bytes_compressed"] += bytes_saved
            else:
                shutil.copy2(filepath, archive_path)
            
            # Truncate original file
            with open(filepath, 'w') as f:
                f.write(f"# Log rotated at {datetime.now().isoformat()}\n")
            
            self.stats["rotations_performed"] += 1
            self.stats["files_archived"] += 1
            self.stats["last_rotation"] = datetime.now().isoformat()
            
            logger.info(f"Rotated: {filename} -> {archive_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to rotate {filepath}: {e}")
            return False
    
    def rotate_all(self) -> Dict[str, int]:
        """
        Rotate all log files that exceed size threshold.
        
        Returns:
            Dictionary with rotation statistics
        """
        log_files = self.get_log_files()
        
        rotated_count = 0
        skipped_count = 0
        
        for filepath in log_files:
            if self.should_rotate(filepath):
                if self.rotate_file(filepath):
                    rotated_count += 1
            else:
                skipped_count += 1
        
        logger.info(
            f"Rotation complete: {rotated_count} rotated, {skipped_count} skipped"
        )
        
        return {
            "rotated": rotated_count,
            "skipped": skipped_count,
            "total": len(log_files),
        }
    
    def get_archived_files(self) -> List[Tuple[str, datetime, int]]:
        """
        Get all archived log files with metadata.
        
        Returns:
            List of tuples (filepath, modified_time, size_bytes)
        """
        archived = []
        
        for filename in os.listdir(self.archive_dir):
            filepath = os.path.join(self.archive_dir, filename)
            
            if os.path.isfile(filepath):
                stat = os.stat(filepath)
                modified_time = datetime.fromtimestamp(stat.st_mtime)
                size = stat.st_size
                
                archived.append((filepath, modified_time, size))
        
        return archived
    
    def cleanup_old_archives(self) -> int:
        """
        Delete archived logs older than retention period.
        
        Returns:
            Number of files deleted
        """
        archived_files = self.get_archived_files()
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        deleted_count = 0
        
        for filepath, modified_time, size in archived_files:
            if modified_time < cutoff_date:
                try:
                    os.remove(filepath)
                    deleted_count += 1
                    
                    logger.info(
                        f"Deleted old archive: {os.path.basename(filepath)} "
                        f"(age: {(datetime.now() - modified_time).days} days)"
                    )
                
                except Exception as e:
                    logger.error(f"Failed to delete {filepath}: {e}")
        
        self.stats["files_deleted"] += deleted_count
        self.stats["last_cleanup"] = datetime.now().isoformat()
        
        logger.info(f"Cleanup complete: {deleted_count} old archives deleted")
        
        return deleted_count
    
    def get_disk_usage(self) -> Dict[str, float]:
        """
        Calculate disk usage for logs and archives.
        
        Returns:
            Dictionary with disk usage statistics (MB)
        """
        log_files = self.get_log_files()
        archived_files = self.get_archived_files()
        
        log_size = sum(self.get_file_size(f) for f in log_files)
        archive_size = sum(size for _, _, size in archived_files)
        
        return {
            "logs_mb": log_size / (1024 ** 2),
            "archives_mb": archive_size / (1024 ** 2),
            "total_mb": (log_size + archive_size) / (1024 ** 2),
            "log_count": len(log_files),
            "archive_count": len(archived_files),
        }
    
    def get_stats(self) -> Dict:
        """
        Get rotation statistics.
        
        Returns:
            Statistics dictionary
        """
        disk_usage = self.get_disk_usage()
        
        return {
            **self.stats,
            "disk_usage": disk_usage,
        }


class ScheduledLogRotator:
    """
    Scheduled log rotation service with automatic daily rotation and cleanup.
    Runs as async background task.
    """
    
    def __init__(
        self,
        rotator: LogRotator,
        rotation_hour: int = 2,  # 2 AM
        check_interval: int = 3600,  # 1 hour
    ):
        """
        Initialize scheduled log rotator.
        
        Args:
            rotator: LogRotator instance
            rotation_hour: Hour of day for daily rotation (0-23)
            check_interval: Check interval in seconds
        """
        self.rotator = rotator
        self.rotation_hour = rotation_hour
        self.check_interval = check_interval
        
        self.is_running = False
        self.last_rotation_date = None
        
        logger.info(
            f"ScheduledLogRotator initialized: rotation_hour={rotation_hour}, "
            f"check_interval={check_interval}s"
        )
    
    async def rotation_loop(self):
        """Main rotation loop."""
        self.is_running = True
        
        logger.info("Starting scheduled log rotation loop")
        
        while self.is_running:
            try:
                current_time = datetime.now()
                current_date = current_time.date()
                current_hour = current_time.hour
                
                # Check if daily rotation should run
                should_rotate_daily = (
                    current_hour == self.rotation_hour and
                    self.last_rotation_date != current_date
                )
                
                if should_rotate_daily:
                    logger.info("Performing scheduled daily rotation")
                    
                    # Rotate logs
                    rotation_stats = self.rotator.rotate_all()
                    
                    # Cleanup old archives
                    deleted = self.rotator.cleanup_old_archives()
                    
                    self.last_rotation_date = current_date
                    
                    logger.info(
                        f"Daily rotation complete: "
                        f"{rotation_stats['rotated']} rotated, {deleted} deleted"
                    )
                
                # Also check for size-based rotation
                for log_file in self.rotator.get_log_files():
                    if self.rotator.should_rotate(log_file):
                        logger.info(f"Size threshold exceeded: {log_file}")
                        self.rotator.rotate_file(log_file)
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
            
            except asyncio.CancelledError:
                logger.info("Rotation loop cancelled")
                break
            
            except Exception as e:
                logger.error(f"Error in rotation loop: {e}")
                await asyncio.sleep(60)
        
        self.is_running = False
        logger.info("Scheduled log rotation loop stopped")
    
    def stop(self):
        """Stop the rotation loop."""
        self.is_running = False


class RotatingFileHandler(logging.Handler):
    """
    Custom logging handler with automatic rotation.
    Integrates LogRotator with Python logging system.
    """
    
    def __init__(
        self,
        filename: str,
        rotator: LogRotator,
        mode: str = 'a',
        encoding: str = 'utf-8',
    ):
        """
        Initialize rotating file handler.
        
        Args:
            filename: Log file path
            rotator: LogRotator instance
            mode: File open mode
            encoding: File encoding
        """
        super().__init__()
        
        self.filename = filename
        self.rotator = rotator
        self.mode = mode
        self.encoding = encoding
        
        # Create file if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        self.stream = None
        self._open()
        
        logger.info(f"RotatingFileHandler initialized: {filename}")
    
    def _open(self):
        """Open the log file."""
        self.stream = open(self.filename, self.mode, encoding=self.encoding)
    
    def _close(self):
        """Close the log file."""
        if self.stream:
            self.stream.flush()
            self.stream.close()
            self.stream = None
    
    def emit(self, record: logging.LogRecord):
        """
        Emit a log record.
        
        Args:
            record: Log record
        """
        try:
            # Check if rotation needed
            if self.rotator.should_rotate(self.filename):
                self._close()
                self.rotator.rotate_file(self.filename)
                self._open()
            
            # Write log record
            msg = self.format(record)
            self.stream.write(msg + '\n')
            self.stream.flush()
        
        except Exception as e:
            self.handleError(record)
    
    def close(self):
        """Close the handler."""
        self._close()
        super().close()


def setup_rotating_logger(
    name: str,
    log_file: str = "logs/app.log",
    level: int = logging.INFO,
    max_size_mb: float = 50.0,
    retention_days: int = 7,
) -> Tuple[logging.Logger, LogRotator]:
    """
    Setup a logger with automatic log rotation.
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
        max_size_mb: Maximum log file size (MB)
        retention_days: Days to retain archives
    
    Returns:
        Tuple of (logger, rotator)
    """
    # Create rotator
    log_dir = os.path.dirname(log_file) or "logs"
    rotator = LogRotator(
        log_dir=log_dir,
        max_size_mb=max_size_mb,
        retention_days=retention_days,
    )
    
    # Create logger
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    
    # Create rotating handler
    handler = RotatingFileHandler(log_file, rotator)
    handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    
    # Add handler
    logger_instance.addHandler(handler)
    
    logger.info(f"Setup rotating logger: {name} -> {log_file}")
    
    return logger_instance, rotator


# Test block
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    async def test_log_rotation():
        """Test log rotation system."""
        print("Testing LogRotator...\n")
        
        # Test 1: Create rotator
        print("1. Initializing LogRotator...")
        test_log_dir = "logs/test"
        test_archive_dir = "logs/test/archive"
        
        rotator = LogRotator(
            log_dir=test_log_dir,
            archive_dir=test_archive_dir,
            max_size_mb=0.1,  # 100 KB for testing
            retention_days=7,
            compress=True,
        )
        
        print(f"   Log directory: {test_log_dir}")
        print(f"   Archive directory: {test_archive_dir}")
        print(f"   Max size: 100 KB")
        print()
        
        # Test 2: Create test log file
        print("2. Creating test log file...")
        test_log_path = os.path.join(test_log_dir, "test.log")
        
        with open(test_log_path, 'w') as f:
            # Write 200 KB of data
            for i in range(2000):
                f.write(f"Line {i}: " + "x" * 100 + "\n")
        
        file_size = os.path.getsize(test_log_path) / 1024
        print(f"   Created: {test_log_path}")
        print(f"   Size: {file_size:.2f} KB")
        print()
        
        # Test 3: Check if rotation needed
        print("3. Checking rotation threshold...")
        should_rotate = rotator.should_rotate(test_log_path)
        print(f"   Should rotate: {should_rotate}")
        print()
        
        # Test 4: Perform rotation
        print("4. Performing rotation...")
        success = rotator.rotate_file(test_log_path)
        print(f"   Rotation successful: {success}")
        
        # Check archive
        archived_files = rotator.get_archived_files()
        print(f"   Archived files: {len(archived_files)}")
        
        if archived_files:
            filepath, modified_time, size = archived_files[0]
            print(f"   Archive: {os.path.basename(filepath)}")
            print(f"   Size: {size / 1024:.2f} KB")
        print()
        
        # Test 5: Verify original file was truncated
        print("5. Verifying truncated log file...")
        new_size = os.path.getsize(test_log_path)
        print(f"   New size: {new_size} bytes")
        print(f"   File truncated: {new_size < 1000}")
        print()
        
        # Test 6: Create multiple log files
        print("6. Creating multiple log files...")
        for i in range(3):
            log_path = os.path.join(test_log_dir, f"test_{i}.log")
            with open(log_path, 'w') as f:
                for j in range(1500):
                    f.write(f"Log {i} Line {j}: " + "y" * 100 + "\n")
        
        log_files = rotator.get_log_files()
        print(f"   Created {len(log_files)} log files")
        print()
        
        # Test 7: Rotate all logs
        print("7. Rotating all logs...")
        rotation_stats = rotator.rotate_all()
        print(f"   Rotated: {rotation_stats['rotated']}")
        print(f"   Skipped: {rotation_stats['skipped']}")
        print(f"   Total: {rotation_stats['total']}")
        print()
        
        # Test 8: Check disk usage
        print("8. Checking disk usage...")
        disk_usage = rotator.get_disk_usage()
        print(f"   Active logs: {disk_usage['logs_mb']:.2f} MB ({disk_usage['log_count']} files)")
        print(f"   Archives: {disk_usage['archives_mb']:.2f} MB ({disk_usage['archive_count']} files)")
        print(f"   Total: {disk_usage['total_mb']:.2f} MB")
        print()
        
        # Test 9: Test cleanup with old files
        print("9. Testing archive cleanup...")
        
        # Artificially age some archives
        archived_files = rotator.get_archived_files()
        
        if archived_files:
            # Modify timestamp to simulate old file
            old_file_path = archived_files[0][0]
            old_time = time.time() - (8 * 86400)  # 8 days ago
            os.utime(old_file_path, (old_time, old_time))
            print(f"   Aged file: {os.path.basename(old_file_path)}")
        
        deleted_count = rotator.cleanup_old_archives()
        print(f"   Deleted old archives: {deleted_count}")
        print()
        
        # Test 10: Get statistics
        print("10. Getting rotation statistics...")
        stats = rotator.get_stats()
        print(f"   Rotations performed: {stats['rotations_performed']}")
        print(f"   Files archived: {stats['files_archived']}")
        print(f"   Files deleted: {stats['files_deleted']}")
        print(f"   Bytes compressed: {stats['bytes_compressed'] / 1024:.2f} KB")
        print(f"   Last rotation: {stats['last_rotation']}")
        print()
        
        # Test 11: Test scheduled rotator
        print("11. Testing ScheduledLogRotator...")
        scheduled = ScheduledLogRotator(
            rotator=rotator,
            rotation_hour=datetime.now().hour,
            check_interval=5,
        )
        
        # Run for 10 seconds
        print("   Running for 10 seconds...")
        rotation_task = asyncio.create_task(scheduled.rotation_loop())
        
        await asyncio.sleep(10)
        
        scheduled.stop()
        
        try:
            await asyncio.wait_for(rotation_task, timeout=2)
        except asyncio.TimeoutError:
            rotation_task.cancel()
        
        print("   Scheduled rotation test complete")
        print()
        
        # Test 12: Test RotatingFileHandler
        print("12. Testing RotatingFileHandler...")
        
        test_logger = logging.getLogger("test_rotating")
        test_logger.setLevel(logging.INFO)
        
        handler_log_path = os.path.join(test_log_dir, "handler_test.log")
        handler = RotatingFileHandler(handler_log_path, rotator)
        
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        
        test_logger.addHandler(handler)
        
        # Write logs
        for i in range(200):
            test_logger.info(f"Test message {i}: " + "z" * 500)
        
        handler.close()
        
        print(f"   Logged 200 messages")
        print(f"   Final log size: {os.path.getsize(handler_log_path) / 1024:.2f} KB")
        print()
        
        # Test 13: Cleanup test files
        print("13. Cleaning up test files...")
        if os.path.exists(test_log_dir):
            shutil.rmtree(test_log_dir)
        print("   Test files removed")
        print()
        
        print("All tests passed successfully!")
        print("\nUsage Example:")
        print("  from src.system_monitor.log_rotation import setup_rotating_logger")
        print()
        print("  logger, rotator = setup_rotating_logger(")
        print("      'myapp',")
        print("      'logs/myapp.log',")
        print("      max_size_mb=50,")
        print("      retention_days=7")
        print("  )")
        print()
        print("  # Use logger normally")
        print("  logger.info('This message will be automatically rotated')")
    
    asyncio.run(test_log_rotation())
