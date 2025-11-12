"""
Auto Recovery Module for omerGPT
"""
from .watchdog import ProcessWatchdog
from .error_notifier import ErrorNotifier, send_error_notification, send_critical_notification

__all__ = ['ProcessWatchdog', 'ErrorNotifier', 'send_error_notification', 'send_critical_notification']
