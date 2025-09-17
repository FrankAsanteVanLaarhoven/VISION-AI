"""
Monitoring Tools for QEP-VLA Application
Provides production monitoring, alerting, and performance tracking
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import json
import time
import threading
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdk_wrapper import QEPVLASDK

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of metrics to monitor"""
    PERFORMANCE = "performance"
    PRIVACY = "privacy"
    SECURITY = "security"
    SYSTEM = "system"
    BUSINESS = "business"

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: Union[int, float, str]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str]
    metadata: Dict[str, Any]

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    level: AlertLevel
    message: str
    metric_name: str
    threshold: Union[int, float]
    current_value: Union[int, float]
    timestamp: datetime
    acknowledged: bool
    resolved: bool

class QEPVLAMonitor:
    """Main monitoring class for QEP-VLA system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.metrics: List[Metric] = []
        self.alerts: List[Alert] = []
        self.thresholds: Dict[str, Dict[str, Union[int, float]]] = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # SDK instance
        self.sdk: Optional[QEPVLASDK] = None
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.start_time = datetime.now()
        
        # Initialize default thresholds
        self._initialize_default_thresholds()
        
    def _initialize_default_thresholds(self):
        """Initialize default monitoring thresholds"""
        self.thresholds = {
            'system_cpu_usage': {'warning': 70.0, 'error': 90.0, 'critical': 95.0},
            'system_memory_usage': {'warning': 80.0, 'error': 90.0, 'critical': 95.0},
            'system_disk_usage': {'warning': 80.0, 'error': 90.0, 'critical': 95.0},
            'privacy_violations': {'warning': 1, 'error': 5, 'critical': 10},
            'response_time_ms': {'warning': 1000, 'error': 5000, 'critical': 10000},
            'error_rate_percent': {'warning': 5.0, 'error': 10.0, 'critical': 20.0},
            'quantum_entanglement_quality': {'warning': 0.7, 'error': 0.5, 'critical': 0.3},
            'federated_learning_convergence': {'warning': 0.8, 'error': 0.6, 'critical': 0.4}
        }
    
    def set_sdk(self, sdk: QEPVLASDK):
        """Set SDK instance for monitoring"""
        self.sdk = sdk
        self.logger.info("SDK instance set for monitoring")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)
        self.logger.info("Alert handler added")
    
    def set_threshold(self, metric_name: str, level: str, value: Union[int, float]):
        """Set custom threshold for a metric"""
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        
        self.thresholds[metric_name][level] = value
        self.logger.info(f"Threshold set for {metric_name}: {level} = {value}")
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect SDK metrics if available
                if self.sdk:
                    self._collect_sdk_metrics()
                
                # Check thresholds and generate alerts
                self._check_thresholds()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Wait for next interval
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric(
                'system_cpu_usage',
                cpu_percent,
                MetricType.SYSTEM,
                {'component': 'system', 'unit': 'percent'}
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._record_metric(
                'system_memory_usage',
                memory_percent,
                MetricType.SYSTEM,
                {'component': 'system', 'unit': 'percent', 'total_gb': memory.total / (1024**3)}
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._record_metric(
                'system_disk_usage',
                disk_percent,
                MetricType.SYSTEM,
                {'component': 'system', 'unit': 'percent', 'total_gb': disk.total / (1024**3)}
            )
            
            # Network I/O
            network = psutil.net_io_counters()
            self._record_metric(
                'system_network_bytes_sent',
                network.bytes_sent,
                MetricType.SYSTEM,
                {'component': 'network', 'unit': 'bytes'}
            )
            self._record_metric(
                'system_network_bytes_recv',
                network.bytes_recv,
                MetricType.SYSTEM,
                {'component': 'network', 'unit': 'bytes'}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _collect_sdk_metrics(self):
        """Collect SDK-specific metrics"""
        try:
            if not self.sdk:
                return
            
            # Get system status
            status = self.sdk.get_system_status()
            
            # Component health metrics
            for component, health in status.get('component_status', {}).items():
                health_value = 1.0 if health == 'active' else 0.0
                self._record_metric(
                    f'component_{component}_health',
                    health_value,
                    MetricType.SYSTEM,
                    {'component': component, 'unit': 'boolean'}
                )
            
            # Privacy transform metrics
            if self.sdk.quantum_privacy_transform:
                transform_summary = self.sdk.quantum_privacy_transform.get_transform_summary()
                
                self._record_metric(
                    'privacy_transforms_total',
                    transform_summary.get('total_transforms', 0),
                    MetricType.PRIVACY,
                    {'component': 'quantum_privacy_transform', 'unit': 'count'}
                )
                
                self._record_metric(
                    'privacy_transform_avg_duration',
                    transform_summary.get('average_duration', 0.0),
                    MetricType.PERFORMANCE,
                    {'component': 'quantum_privacy_transform', 'unit': 'seconds'}
                )
            
            # Navigation metrics
            if self.sdk.navigation_engine:
                nav_status = self.sdk.navigation_engine.get_navigation_status()
                
                self._record_metric(
                    'navigation_waypoints_count',
                    nav_status.get('waypoints_count', 0),
                    MetricType.SYSTEM,
                    {'component': 'navigation_engine', 'unit': 'count'}
                )
                
                self._record_metric(
                    'navigation_privacy_zones_count',
                    nav_status.get('privacy_zones_count', 0),
                    MetricType.PRIVACY,
                    {'component': 'navigation_engine', 'unit': 'count'}
                )
            
            # Scenario generator metrics
            if self.sdk.scenario_generator:
                scenario_summary = self.sdk.scenario_generator.get_scenario_summary()
                
                self._record_metric(
                    'scenarios_generated_total',
                    scenario_summary.get('total_scenarios', 0),
                    MetricType.SYSTEM,
                    {'component': 'scenario_generator', 'unit': 'count'}
                )
            
            # Secure aggregation metrics
            if self.sdk.secure_aggregation:
                agg_summary = self.sdk.secure_aggregation.get_aggregation_summary()
                
                self._record_metric(
                    'secure_aggregations_total',
                    agg_summary.get('total_aggregations', 0),
                    MetricType.SECURITY,
                    {'component': 'secure_aggregation', 'unit': 'count'}
                )
                
                self._record_metric(
                    'secure_aggregation_avg_duration',
                    agg_summary.get('average_duration', 0.0),
                    MetricType.PERFORMANCE,
                    {'component': 'secure_aggregation', 'unit': 'seconds'}
                )
            
        except Exception as e:
            self.logger.error(f"Failed to collect SDK metrics: {e}")
    
    def _record_metric(self, name: str, value: Union[int, float, str], 
                      metric_type: MetricType, tags: Dict[str, str], 
                      metadata: Optional[Dict[str, Any]] = None):
        """Record a new metric"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            tags=tags,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Update performance history
        if name not in self.performance_history:
            self.performance_history[name] = []
        
        if isinstance(value, (int, float)):
            self.performance_history[name].append(float(value))
            
            # Keep only last 1000 values
            if len(self.performance_history[name]) > 1000:
                self.performance_history[name] = self.performance_history[name][-1000:]
    
    def _check_thresholds(self):
        """Check metrics against thresholds and generate alerts"""
        for metric_name, thresholds in self.thresholds.items():
            # Get latest metric value
            latest_metric = self._get_latest_metric(metric_name)
            if not latest_metric:
                continue
            
            current_value = latest_metric.value
            if not isinstance(current_value, (int, float)):
                continue
            
            # Check each threshold level
            for level, threshold in thresholds.items():
                if self._is_threshold_violated(current_value, threshold, level):
                    self._generate_alert(metric_name, level, threshold, current_value)
    
    def _get_latest_metric(self, metric_name: str) -> Optional[Metric]:
        """Get the latest metric for a given name"""
        for metric in reversed(self.metrics):
            if metric.name == metric_name:
                return metric
        return None
    
    def _is_threshold_violated(self, value: float, threshold: float, level: str) -> bool:
        """Check if a threshold is violated"""
        if level == 'critical':
            return value >= threshold
        elif level == 'error':
            return value >= threshold
        elif level == 'warning':
            return value >= threshold
        return False
    
    def _generate_alert(self, metric_name: str, level: str, threshold: float, current_value: float):
        """Generate an alert for threshold violation"""
        # Check if alert already exists
        alert_id = f"{metric_name}_{level}_{int(time.time())}"
        
        for alert in self.alerts:
            if (alert.metric_name == metric_name and 
                alert.level.value == level and 
                not alert.resolved):
                return  # Alert already exists
        
        # Create new alert
        alert = Alert(
            id=alert_id,
            level=AlertLevel(level),
            message=f"{metric_name} threshold {level} violated: {current_value} >= {threshold}",
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.now(),
            acknowledged=False,
            resolved=False
        )
        
        self.alerts.append(alert)
        self.logger.warning(f"Alert generated: {alert.message}")
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory issues"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
        
        # Remove old metrics
        self.metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        # Remove old alerts (keep for 7 days)
        alert_cutoff = datetime.now() - timedelta(days=7)
        self.alerts = [a for a in self.alerts if a.timestamp > alert_cutoff]
    
    def get_metrics_summary(self, metric_name: Optional[str] = None, 
                           metric_type: Optional[MetricType] = None,
                           time_range: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get summary of metrics"""
        if time_range is None:
            time_range = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_range
        filtered_metrics = [m for m in self.metrics if m.timestamp > cutoff_time]
        
        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.name == metric_name]
        
        if metric_type:
            filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]
        
        if not filtered_metrics:
            return {}
        
        # Calculate statistics
        numeric_values = [m.value for m in filtered_metrics if isinstance(m.value, (int, float))]
        
        summary = {
            'count': len(filtered_metrics),
            'time_range': str(time_range),
            'metric_name': metric_name,
            'metric_type': metric_type.value if metric_type else None
        }
        
        if numeric_values:
            summary.update({
                'min': min(numeric_values),
                'max': max(numeric_values),
                'mean': np.mean(numeric_values),
                'std': np.std(numeric_values),
                'median': np.median(numeric_values)
            })
        
        return summary
    
    def get_alerts_summary(self, level: Optional[AlertLevel] = None, 
                          resolved: Optional[bool] = None) -> Dict[str, Any]:
        """Get summary of alerts"""
        filtered_alerts = self.alerts
        
        if level:
            filtered_alerts = [a for a in filtered_alerts if a.level == level]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
        
        # Group by level
        alerts_by_level = {}
        for alert in filtered_alerts:
            level_str = alert.level.value
            if level_str not in alerts_by_level:
                alerts_by_level[level_str] = []
            alerts_by_level[level_str].append(asdict(alert))
        
        return {
            'total_alerts': len(filtered_alerts),
            'unresolved_alerts': len([a for a in filtered_alerts if not a.resolved]),
            'alerts_by_level': alerts_by_level,
            'timestamp': datetime.now().isoformat()
        }
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self.logger.info(f"Alert {alert_id} resolved")
                return True
        return False
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump([asdict(m) for m in self.metrics], f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def export_alerts(self, filepath: str, format: str = 'json'):
        """Export alerts to file"""
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump([asdict(a) for a in self.alerts], f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Alerts exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export alerts: {e}")
            return False

# Alert handlers
class ConsoleAlertHandler:
    """Console-based alert handler"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, alert: Alert):
        """Handle alert by printing to console"""
        color_map = {
            AlertLevel.INFO: '\033[94m',      # Blue
            AlertLevel.WARNING: '\033[93m',   # Yellow
            AlertLevel.ERROR: '\033[91m',     # Red
            AlertLevel.CRITICAL: '\033[95m'   # Magenta
        }
        
        color = color_map.get(alert.level, '\033[0m')
        reset = '\033[0m'
        
        print(f"{color}[{alert.level.value.upper()}] {alert.message}{reset}")
        print(f"  Metric: {alert.metric_name}")
        print(f"  Current: {alert.current_value}, Threshold: {alert.threshold}")
        print(f"  Time: {alert.timestamp}")
        print()

class FileAlertHandler:
    """File-based alert handler"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, alert: Alert):
        """Handle alert by writing to file"""
        try:
            with open(self.filepath, 'a') as f:
                f.write(f"[{alert.timestamp}] {alert.level.value.upper()}: {alert.message}\n")
                f.write(f"  Metric: {alert.metric_name}\n")
                f.write(f"  Current: {alert.current_value}, Threshold: {alert.threshold}\n\n")
        except Exception as e:
            self.logger.error(f"Failed to write alert to file: {e}")

# Convenience functions
def create_monitor(config: Optional[Dict[str, Any]] = None) -> QEPVLAMonitor:
    """Create and return a monitoring instance"""
    return QEPVLAMonitor(config)

def quick_monitoring(sdk: QEPVLASDK, duration: int = 300, interval: int = 30):
    """Quick monitoring session"""
    monitor = QEPVLAMonitor()
    monitor.set_sdk(sdk)
    
    # Add console handler
    monitor.add_alert_handler(ConsoleAlertHandler())
    
    # Start monitoring
    monitor.start_monitoring(interval)
    
    try:
        time.sleep(duration)
    finally:
        monitor.stop_monitoring()
    
    return monitor
