"""
Prometheus Metrics Exporter for Medical Device AI Security.

Provides metrics export capabilities for:
- Model drift detection metrics
- Model performance metrics (accuracy, latency)
- Security event metrics
- SBOM vulnerability metrics

Compatible with Prometheus and OpenMetrics format.

Example:
    >>> exporter = MetricsExporter(
    ...     model_name="diagnostic_classifier",
    ...     model_version="1.0.0",
    ... )
    >>> exporter.record_drift("kl_divergence", 0.15)
    >>> exporter.record_prediction(latency_ms=25.5, correct=True)
    >>> metrics = exporter.export()
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MetricType(Enum):
    """Prometheus metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class DriftSeverity(Enum):
    """Drift severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricSample:
    """Individual metric sample."""

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float | None = None


@dataclass
class DriftMetrics:
    """Drift detection metrics container."""

    model_name: str
    model_version: str
    drift_type: str  # e.g., "kl_divergence", "psi", "ks_statistic"
    drift_value: float
    threshold: float
    severity: DriftSeverity
    timestamp: str
    feature_drifts: dict[str, float] = field(default_factory=dict)

    def is_drifted(self) -> bool:
        """Check if drift exceeds threshold."""
        return self.drift_value > self.threshold

    def to_prometheus_metrics(self) -> list[MetricSample]:
        """Convert to Prometheus metrics format."""
        labels = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "drift_type": self.drift_type,
        }

        metrics = [
            MetricSample(
                name="ml_model_drift_score",
                value=self.drift_value,
                labels=labels,
            ),
            MetricSample(
                name="ml_model_drift_threshold",
                value=self.threshold,
                labels=labels,
            ),
            MetricSample(
                name="ml_model_drift_detected",
                value=1.0 if self.is_drifted() else 0.0,
                labels=labels,
            ),
            MetricSample(
                name="ml_model_drift_severity",
                value=float(list(DriftSeverity).index(self.severity)),
                labels={**labels, "severity": self.severity.value},
            ),
        ]

        # Add per-feature drift metrics
        for feature_name, drift_val in self.feature_drifts.items():
            metrics.append(
                MetricSample(
                    name="ml_model_feature_drift",
                    value=drift_val,
                    labels={**labels, "feature": feature_name},
                )
            )

        return metrics


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics container."""

    model_name: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float | None = None
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    predictions_total: int = 0
    errors_total: int = 0
    timestamp: str = ""

    def to_prometheus_metrics(self) -> list[MetricSample]:
        """Convert to Prometheus metrics format."""
        labels = {
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

        metrics = [
            MetricSample(
                name="ml_model_accuracy",
                value=self.accuracy,
                labels=labels,
            ),
            MetricSample(
                name="ml_model_precision",
                value=self.precision,
                labels=labels,
            ),
            MetricSample(
                name="ml_model_recall",
                value=self.recall,
                labels=labels,
            ),
            MetricSample(
                name="ml_model_f1_score",
                value=self.f1_score,
                labels=labels,
            ),
            MetricSample(
                name="ml_model_predictions_total",
                value=float(self.predictions_total),
                labels=labels,
            ),
            MetricSample(
                name="ml_model_errors_total",
                value=float(self.errors_total),
                labels=labels,
            ),
            MetricSample(
                name="ml_model_latency_milliseconds",
                value=self.latency_p50_ms,
                labels={**labels, "quantile": "0.5"},
            ),
            MetricSample(
                name="ml_model_latency_milliseconds",
                value=self.latency_p95_ms,
                labels={**labels, "quantile": "0.95"},
            ),
            MetricSample(
                name="ml_model_latency_milliseconds",
                value=self.latency_p99_ms,
                labels={**labels, "quantile": "0.99"},
            ),
        ]

        if self.auc_roc is not None:
            metrics.append(
                MetricSample(
                    name="ml_model_auc_roc",
                    value=self.auc_roc,
                    labels=labels,
                )
            )

        return metrics


@dataclass
class SecurityEventMetrics:
    """Security event metrics container."""

    model_name: str
    event_type: str  # e.g., "adversarial_detected", "anomaly_detected"
    event_count: int
    severity: str
    timestamp: str

    def to_prometheus_metrics(self) -> list[MetricSample]:
        """Convert to Prometheus metrics format."""
        return [
            MetricSample(
                name="ml_security_events_total",
                value=float(self.event_count),
                labels={
                    "model_name": self.model_name,
                    "event_type": self.event_type,
                    "severity": self.severity,
                },
            ),
        ]


class MetricsExporter:
    """
    Prometheus metrics exporter for ML model monitoring.

    Collects and exports metrics in Prometheus/OpenMetrics format for:
    - Model drift detection
    - Model performance monitoring
    - Security event tracking
    - SBOM vulnerability tracking

    Example:
        >>> exporter = MetricsExporter(
        ...     model_name="classifier",
        ...     model_version="1.0.0",
        ...     namespace="medtech",
        ... )
        >>> exporter.record_drift("psi", 0.2, threshold=0.1)
        >>> exporter.record_prediction(latency_ms=15.0, correct=True)
        >>> print(exporter.export())  # Prometheus format
    """

    def __init__(
        self,
        model_name: str,
        model_version: str = "unknown",
        namespace: str = "medtech_ai",
        labels: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize metrics exporter.

        Args:
            model_name: Name of the model being monitored
            model_version: Version of the model
            namespace: Prometheus metric namespace prefix
            labels: Additional labels to add to all metrics
        """
        self.model_name = model_name
        self.model_version = model_version
        self.namespace = namespace
        self.base_labels = labels or {}

        # Internal state
        self._drift_metrics: list[DriftMetrics] = []
        self._performance_metrics: list[ModelPerformanceMetrics] = []
        self._security_events: list[SecurityEventMetrics] = []
        self._prediction_latencies: list[float] = []
        self._prediction_results: list[bool] = []
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}

    def record_drift(
        self,
        drift_type: str,
        drift_value: float,
        threshold: float = 0.1,
        feature_drifts: dict[str, float] | None = None,
    ) -> DriftMetrics:
        """
        Record drift detection result.

        Args:
            drift_type: Type of drift metric (e.g., "psi", "kl_divergence")
            drift_value: Calculated drift value
            threshold: Drift threshold for alerting
            feature_drifts: Per-feature drift values

        Returns:
            DriftMetrics object
        """
        severity = self._classify_drift_severity(drift_value, threshold)

        metrics = DriftMetrics(
            model_name=self.model_name,
            model_version=self.model_version,
            drift_type=drift_type,
            drift_value=drift_value,
            threshold=threshold,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            feature_drifts=feature_drifts or {},
        )

        self._drift_metrics.append(metrics)
        return metrics

    def record_prediction(
        self,
        latency_ms: float,
        correct: bool | None = None,
    ) -> None:
        """
        Record prediction for performance metrics.

        Args:
            latency_ms: Prediction latency in milliseconds
            correct: Whether prediction was correct (if known)
        """
        self._prediction_latencies.append(latency_ms)
        if correct is not None:
            self._prediction_results.append(correct)

    def record_security_event(
        self,
        event_type: str,
        severity: str = "medium",
    ) -> None:
        """
        Record security event.

        Args:
            event_type: Type of security event
            severity: Event severity level
        """
        # Increment counter
        key = f"{event_type}:{severity}"
        self._counters[key] = self._counters.get(key, 0) + 1

        event = SecurityEventMetrics(
            model_name=self.model_name,
            event_type=event_type,
            event_count=int(self._counters[key]),
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._security_events.append(event)

    def record_performance(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        auc_roc: float | None = None,
    ) -> ModelPerformanceMetrics:
        """
        Record model performance metrics.

        Args:
            accuracy: Model accuracy
            precision: Model precision
            recall: Model recall
            f1_score: Model F1 score
            auc_roc: Area under ROC curve

        Returns:
            ModelPerformanceMetrics object
        """
        # Calculate latency percentiles
        latencies = sorted(self._prediction_latencies) if self._prediction_latencies else [0.0]
        p50_idx = int(len(latencies) * 0.5)
        p95_idx = int(len(latencies) * 0.95)
        p99_idx = int(len(latencies) * 0.99)

        metrics = ModelPerformanceMetrics(
            model_name=self.model_name,
            model_version=self.model_version,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            auc_roc=auc_roc,
            latency_p50_ms=latencies[p50_idx] if latencies else 0.0,
            latency_p95_ms=latencies[min(p95_idx, len(latencies) - 1)] if latencies else 0.0,
            latency_p99_ms=latencies[min(p99_idx, len(latencies) - 1)] if latencies else 0.0,
            predictions_total=len(self._prediction_latencies),
            errors_total=sum(1 for r in self._prediction_results if not r),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self._performance_metrics.append(metrics)
        return metrics

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """
        Set a custom gauge metric.

        Args:
            name: Metric name
            value: Metric value
            labels: Additional labels
        """
        label_str = json.dumps(labels or {}, sort_keys=True)
        self._gauges[f"{name}:{label_str}"] = value

    def increment_counter(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """
        Increment a custom counter metric.

        Args:
            name: Metric name
            value: Increment value
            labels: Additional labels
        """
        label_str = json.dumps(labels or {}, sort_keys=True)
        key = f"{name}:{label_str}"
        self._counters[key] = self._counters.get(key, 0) + value

    def export(self, format: str = "prometheus") -> str:
        """
        Export all metrics in specified format.

        Args:
            format: Export format ("prometheus" or "json")

        Returns:
            Formatted metrics string
        """
        if format == "json":
            return self._export_json()
        return self._export_prometheus()

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        int(time.time() * 1000)

        # Helper to format metric line
        def format_metric(sample: MetricSample) -> str:
            name = f"{self.namespace}_{sample.name}"
            labels = {**self.base_labels, **sample.labels}
            if labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
                return f"{name}{{{label_str}}} {sample.value}"
            return f"{name} {sample.value}"

        # Add drift metrics
        if self._drift_metrics:
            lines.append("# HELP medtech_ai_ml_model_drift_score Current drift score")
            lines.append("# TYPE medtech_ai_ml_model_drift_score gauge")
            for dm in self._drift_metrics:
                for sample in dm.to_prometheus_metrics():
                    lines.append(format_metric(sample))

        # Add performance metrics
        if self._performance_metrics:
            lines.append("")
            lines.append("# HELP medtech_ai_ml_model_accuracy Model accuracy")
            lines.append("# TYPE medtech_ai_ml_model_accuracy gauge")
            for pm in self._performance_metrics:
                for sample in pm.to_prometheus_metrics():
                    lines.append(format_metric(sample))

        # Add security events
        if self._security_events:
            lines.append("")
            lines.append("# HELP medtech_ai_ml_security_events_total Total security events")
            lines.append("# TYPE medtech_ai_ml_security_events_total counter")
            for se in self._security_events:
                for sample in se.to_prometheus_metrics():
                    lines.append(format_metric(sample))

        # Add custom gauges
        if self._gauges:
            lines.append("")
            lines.append("# HELP medtech_ai_custom_gauge Custom gauge metrics")
            lines.append("# TYPE medtech_ai_custom_gauge gauge")
            for key, value in self._gauges.items():
                name, label_str = key.split(":", 1)
                labels = json.loads(label_str)
                sample = MetricSample(name=name, value=value, labels=labels)
                lines.append(format_metric(sample))

        # Add custom counters
        if self._counters:
            lines.append("")
            lines.append("# HELP medtech_ai_custom_counter Custom counter metrics")
            lines.append("# TYPE medtech_ai_custom_counter counter")
            for key, value in self._counters.items():
                if ":" in key:
                    name, label_str = key.split(":", 1)
                    try:
                        labels = json.loads(label_str)
                    except json.JSONDecodeError:
                        # Handle simple key:value format
                        labels = {"type": label_str}
                else:
                    name = key
                    labels = {}
                sample = MetricSample(name=name, value=value, labels=labels)
                lines.append(format_metric(sample))

        # Add model info metric
        lines.append("")
        lines.append("# HELP medtech_ai_model_info Model information")
        lines.append("# TYPE medtech_ai_model_info gauge")
        info_sample = MetricSample(
            name="model_info",
            value=1.0,
            labels={
                "model_name": self.model_name,
                "model_version": self.model_version,
            },
        )
        lines.append(format_metric(info_sample))

        return "\n".join(lines)

    def _export_json(self) -> str:
        """Export metrics in JSON format."""
        data: dict[str, Any] = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_metrics": [],
            "performance_metrics": [],
            "security_events": [],
            "custom_gauges": {},
            "custom_counters": {},
        }

        for dm in self._drift_metrics:
            data["drift_metrics"].append(
                {
                    "drift_type": dm.drift_type,
                    "drift_value": dm.drift_value,
                    "threshold": dm.threshold,
                    "severity": dm.severity.value,
                    "is_drifted": dm.is_drifted(),
                    "feature_drifts": dm.feature_drifts,
                    "timestamp": dm.timestamp,
                }
            )

        for pm in self._performance_metrics:
            data["performance_metrics"].append(
                {
                    "accuracy": pm.accuracy,
                    "precision": pm.precision,
                    "recall": pm.recall,
                    "f1_score": pm.f1_score,
                    "auc_roc": pm.auc_roc,
                    "latency_p50_ms": pm.latency_p50_ms,
                    "latency_p95_ms": pm.latency_p95_ms,
                    "latency_p99_ms": pm.latency_p99_ms,
                    "predictions_total": pm.predictions_total,
                    "errors_total": pm.errors_total,
                    "timestamp": pm.timestamp,
                }
            )

        for se in self._security_events:
            data["security_events"].append(
                {
                    "event_type": se.event_type,
                    "event_count": se.event_count,
                    "severity": se.severity,
                    "timestamp": se.timestamp,
                }
            )

        data["custom_gauges"] = dict(self._gauges)
        data["custom_counters"] = dict(self._counters)

        return json.dumps(data, indent=2)

    def generate_grafana_dashboard(self) -> dict[str, Any]:
        """
        Generate Grafana dashboard JSON.

        Returns:
            Grafana dashboard configuration
        """
        return {
            "annotations": {"list": []},
            "editable": True,
            "fiscalYearStartMonth": 0,
            "graphTooltip": 0,
            "id": None,
            "links": [],
            "liveNow": False,
            "panels": [
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "fieldConfig": {
                        "defaults": {
                            "color": {"mode": "palette-classic"},
                            "thresholds": {
                                "mode": "absolute",
                                "steps": [
                                    {"color": "green", "value": None},
                                    {"color": "yellow", "value": 0.1},
                                    {"color": "red", "value": 0.2},
                                ],
                            },
                        },
                    },
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                    "id": 1,
                    "options": {},
                    "title": "Model Drift Score",
                    "type": "gauge",
                    "targets": [
                        {
                            "expr": f'{self.namespace}_ml_model_drift_score{{model_name="{self.model_name}"}}',
                            "refId": "A",
                        }
                    ],
                },
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
                    "id": 2,
                    "title": "Model Performance",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": f'{self.namespace}_ml_model_accuracy{{model_name="{self.model_name}"}}',
                            "legendFormat": "Accuracy",
                            "refId": "A",
                        },
                        {
                            "expr": f'{self.namespace}_ml_model_precision{{model_name="{self.model_name}"}}',
                            "legendFormat": "Precision",
                            "refId": "B",
                        },
                        {
                            "expr": f'{self.namespace}_ml_model_recall{{model_name="{self.model_name}"}}',
                            "legendFormat": "Recall",
                            "refId": "C",
                        },
                    ],
                },
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "id": 3,
                    "title": "Prediction Latency (ms)",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": f'{self.namespace}_ml_model_latency_milliseconds{{model_name="{self.model_name}",quantile="0.5"}}',
                            "legendFormat": "p50",
                            "refId": "A",
                        },
                        {
                            "expr": f'{self.namespace}_ml_model_latency_milliseconds{{model_name="{self.model_name}",quantile="0.95"}}',
                            "legendFormat": "p95",
                            "refId": "B",
                        },
                        {
                            "expr": f'{self.namespace}_ml_model_latency_milliseconds{{model_name="{self.model_name}",quantile="0.99"}}',
                            "legendFormat": "p99",
                            "refId": "C",
                        },
                    ],
                },
                {
                    "datasource": {"type": "prometheus", "uid": "${datasource}"},
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "id": 4,
                    "title": "Security Events",
                    "type": "timeseries",
                    "targets": [
                        {
                            "expr": f'rate({self.namespace}_ml_security_events_total{{model_name="{self.model_name}"}}[5m])',
                            "legendFormat": "{{event_type}}",
                            "refId": "A",
                        },
                    ],
                },
            ],
            "refresh": "30s",
            "schemaVersion": 38,
            "style": "dark",
            "tags": ["ml", "monitoring", "medtech"],
            "templating": {
                "list": [
                    {
                        "current": {"selected": False, "text": "default", "value": "default"},
                        "hide": 0,
                        "includeAll": False,
                        "label": "Data Source",
                        "multi": False,
                        "name": "datasource",
                        "options": [],
                        "query": "prometheus",
                        "refresh": 1,
                        "regex": "",
                        "skipUrlSync": False,
                        "type": "datasource",
                    }
                ]
            },
            "time": {"from": "now-6h", "to": "now"},
            "timepicker": {},
            "timezone": "",
            "title": f"ML Model Monitoring - {self.model_name}",
            "uid": f"ml-monitoring-{self.model_name.replace(' ', '-').lower()}",
            "version": 1,
            "weekStart": "",
        }

    def _classify_drift_severity(self, drift_value: float, threshold: float) -> DriftSeverity:
        """Classify drift severity based on value relative to threshold."""
        if drift_value <= threshold * 0.5:
            return DriftSeverity.NONE
        elif drift_value <= threshold:
            return DriftSeverity.LOW
        elif drift_value <= threshold * 1.5:
            return DriftSeverity.MEDIUM
        elif drift_value <= threshold * 2:
            return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL

    def reset(self) -> None:
        """Reset all metrics."""
        self._drift_metrics.clear()
        self._performance_metrics.clear()
        self._security_events.clear()
        self._prediction_latencies.clear()
        self._prediction_results.clear()
        self._counters.clear()
        self._gauges.clear()
