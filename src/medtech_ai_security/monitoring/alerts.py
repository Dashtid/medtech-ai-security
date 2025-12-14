"""
Alerting Rules Generator for Medical Device AI Security.

Generates Prometheus Alertmanager rules for:
- Model drift detection alerts
- Performance degradation alerts
- Security event alerts
- SBOM vulnerability alerts

Supports FDA compliance by ensuring timely notification of critical issues.

Example:
    >>> generator = AlertRulesGenerator(
    ...     model_name="diagnostic_classifier",
    ...     namespace="medtech_ai",
    ... )
    >>> generator.add_drift_alerts(critical_threshold=0.3)
    >>> generator.add_performance_alerts(min_accuracy=0.95)
    >>> rules_yaml = generator.export_prometheus_rules()
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import yaml


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"  # Requires immediate human attention


class AlertCategory(Enum):
    """Alert categories for grouping."""

    DRIFT = "drift"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AVAILABILITY = "availability"
    COMPLIANCE = "compliance"


@dataclass
class AlertRule:
    """Individual alerting rule."""

    name: str
    expr: str
    duration: str  # e.g., "5m", "15m"
    severity: AlertSeverity
    category: AlertCategory
    summary: str
    description: str
    runbook_url: str = ""
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    def to_prometheus_rule(self) -> dict[str, Any]:
        """Convert to Prometheus alerting rule format."""
        rule: dict[str, Any] = {
            "alert": self.name,
            "expr": self.expr,
            "for": self.duration,
            "labels": {
                "severity": self.severity.value,
                "category": self.category.value,
                **self.labels,
            },
            "annotations": {
                "summary": self.summary,
                "description": self.description,
                **self.annotations,
            },
        }

        if self.runbook_url:
            rule["annotations"]["runbook_url"] = self.runbook_url

        return rule


@dataclass
class AlertRuleGroup:
    """Group of related alerting rules."""

    name: str
    rules: list[AlertRule] = field(default_factory=list)
    interval: str = "30s"

    def to_prometheus_group(self) -> dict[str, Any]:
        """Convert to Prometheus rule group format."""
        return {
            "name": self.name,
            "interval": self.interval,
            "rules": [rule.to_prometheus_rule() for rule in self.rules],
        }


class AlertRulesGenerator:
    """
    Generator for Prometheus Alertmanager rules.

    Creates alerting rules for ML model monitoring including:
    - Drift detection alerts
    - Performance degradation alerts
    - Security event alerts
    - Availability alerts

    Example:
        >>> generator = AlertRulesGenerator(
        ...     model_name="classifier",
        ...     namespace="medtech_ai",
        ... )
        >>> generator.add_drift_alerts()
        >>> generator.add_performance_alerts()
        >>> generator.add_security_alerts()
        >>> print(generator.export_prometheus_rules())
    """

    def __init__(
        self,
        model_name: str,
        namespace: str = "medtech_ai",
        team: str = "ml-platform",
        environment: str = "production",
    ) -> None:
        """
        Initialize alert rules generator.

        Args:
            model_name: Name of the model being monitored
            namespace: Prometheus metric namespace
            team: Team responsible for alerts
            environment: Deployment environment
        """
        self.model_name = model_name
        self.namespace = namespace
        self.team = team
        self.environment = environment
        self._rule_groups: list[AlertRuleGroup] = []
        self._created = datetime.now(timezone.utc).isoformat()

    def add_drift_alerts(
        self,
        warning_threshold: float = 0.1,
        critical_threshold: float = 0.2,
        evaluation_duration: str = "5m",
    ) -> None:
        """
        Add drift detection alerting rules.

        Args:
            warning_threshold: Threshold for warning alerts
            critical_threshold: Threshold for critical alerts
            evaluation_duration: Time to wait before firing alert
        """
        group = AlertRuleGroup(
            name=f"{self.model_name}_drift_alerts",
            rules=[
                AlertRule(
                    name=f"{self.model_name}_drift_warning",
                    expr=f'{self.namespace}_ml_model_drift_score{{model_name="{self.model_name}"}} > {warning_threshold}',
                    duration=evaluation_duration,
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.DRIFT,
                    summary=f"Model drift detected for {self.model_name}",
                    description=f"Drift score for model {self.model_name} has exceeded warning threshold of {warning_threshold}. "
                    "Current value: {{ $value }}. Investigate data distribution changes.",
                    labels={"model": self.model_name, "team": self.team},
                ),
                AlertRule(
                    name=f"{self.model_name}_drift_critical",
                    expr=f'{self.namespace}_ml_model_drift_score{{model_name="{self.model_name}"}} > {critical_threshold}',
                    duration=evaluation_duration,
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.DRIFT,
                    summary=f"Critical model drift detected for {self.model_name}",
                    description=f"Drift score for model {self.model_name} has exceeded critical threshold of {critical_threshold}. "
                    "Current value: {{ $value }}. Immediate investigation required. Consider model rollback.",
                    labels={"model": self.model_name, "team": self.team},
                    annotations={
                        "action": "Consider triggering model rollback if drift persists",
                        "fda_impact": "Model performance may affect clinical decisions",
                    },
                ),
                AlertRule(
                    name=f"{self.model_name}_feature_drift",
                    expr=f'{self.namespace}_ml_model_feature_drift{{model_name="{self.model_name}"}} > {warning_threshold}',
                    duration="10m",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.DRIFT,
                    summary=f"Feature drift detected for {self.model_name}",
                    description=f"Feature drift detected for feature {{{{ $labels.feature }}}} in model {self.model_name}. "
                    "Value: {{ $value }}. Review input data quality.",
                    labels={"model": self.model_name, "team": self.team},
                ),
            ],
        )
        self._rule_groups.append(group)

    def add_performance_alerts(
        self,
        min_accuracy: float = 0.90,
        min_precision: float = 0.85,
        min_recall: float = 0.85,
        max_latency_p99_ms: float = 500.0,
        max_error_rate: float = 0.05,
    ) -> None:
        """
        Add performance monitoring alerting rules.

        Args:
            min_accuracy: Minimum acceptable accuracy
            min_precision: Minimum acceptable precision
            min_recall: Minimum acceptable recall
            max_latency_p99_ms: Maximum acceptable p99 latency
            max_error_rate: Maximum acceptable error rate
        """
        group = AlertRuleGroup(
            name=f"{self.model_name}_performance_alerts",
            rules=[
                AlertRule(
                    name=f"{self.model_name}_accuracy_degraded",
                    expr=f'{self.namespace}_ml_model_accuracy{{model_name="{self.model_name}"}} < {min_accuracy}',
                    duration="10m",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.PERFORMANCE,
                    summary=f"Model accuracy degraded for {self.model_name}",
                    description=f"Model accuracy for {self.model_name} has dropped below {min_accuracy}. "
                    "Current value: {{ $value }}. This may impact clinical decision support.",
                    labels={"model": self.model_name, "team": self.team},
                    annotations={
                        "fda_impact": "Accuracy degradation may affect diagnostic reliability",
                        "action": "Review model performance and consider retraining",
                    },
                ),
                AlertRule(
                    name=f"{self.model_name}_precision_degraded",
                    expr=f'{self.namespace}_ml_model_precision{{model_name="{self.model_name}"}} < {min_precision}',
                    duration="10m",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.PERFORMANCE,
                    summary=f"Model precision degraded for {self.model_name}",
                    description=f"Model precision for {self.model_name} has dropped below {min_precision}. "
                    "Current value: {{ $value }}. Increased false positives may occur.",
                    labels={"model": self.model_name, "team": self.team},
                ),
                AlertRule(
                    name=f"{self.model_name}_recall_degraded",
                    expr=f'{self.namespace}_ml_model_recall{{model_name="{self.model_name}"}} < {min_recall}',
                    duration="10m",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.PERFORMANCE,
                    summary=f"Model recall degraded for {self.model_name}",
                    description=f"Model recall for {self.model_name} has dropped below {min_recall}. "
                    "Current value: {{ $value }}. Increased false negatives may occur - critical for medical applications.",
                    labels={"model": self.model_name, "team": self.team},
                    annotations={
                        "fda_impact": "Low recall may result in missed diagnoses",
                        "action": "Immediate review required for patient safety",
                    },
                ),
                AlertRule(
                    name=f"{self.model_name}_latency_high",
                    expr=f'{self.namespace}_ml_model_latency_milliseconds{{model_name="{self.model_name}",quantile="0.99"}} > {max_latency_p99_ms}',
                    duration="5m",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.PERFORMANCE,
                    summary=f"High prediction latency for {self.model_name}",
                    description=f"P99 latency for model {self.model_name} exceeds {max_latency_p99_ms}ms. "
                    "Current value: {{ $value }}ms. May impact user experience.",
                    labels={"model": self.model_name, "team": self.team},
                ),
                AlertRule(
                    name=f"{self.model_name}_error_rate_high",
                    expr=f'rate({self.namespace}_ml_model_errors_total{{model_name="{self.model_name}"}}[5m]) / '
                    f'rate({self.namespace}_ml_model_predictions_total{{model_name="{self.model_name}"}}[5m]) > {max_error_rate}',
                    duration="5m",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.PERFORMANCE,
                    summary=f"High error rate for {self.model_name}",
                    description=f"Error rate for model {self.model_name} exceeds {max_error_rate*100}%. "
                    "Current rate: {{ $value }}. Investigate model health and input data.",
                    labels={"model": self.model_name, "team": self.team},
                ),
            ],
        )
        self._rule_groups.append(group)

    def add_security_alerts(
        self,
        adversarial_threshold: int = 5,
        anomaly_threshold: int = 10,
    ) -> None:
        """
        Add security event alerting rules.

        Args:
            adversarial_threshold: Number of adversarial detections to alert
            anomaly_threshold: Number of anomaly detections to alert
        """
        group = AlertRuleGroup(
            name=f"{self.model_name}_security_alerts",
            rules=[
                AlertRule(
                    name=f"{self.model_name}_adversarial_detected",
                    expr=f'increase({self.namespace}_ml_security_events_total{{model_name="{self.model_name}",event_type="adversarial_detected"}}[15m]) > {adversarial_threshold}',
                    duration="0m",  # Alert immediately
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.SECURITY,
                    summary=f"Adversarial inputs detected for {self.model_name}",
                    description=f"More than {adversarial_threshold} adversarial inputs detected for model {self.model_name} in the last 15 minutes. "
                    "This may indicate an attack. Investigate input sources.",
                    labels={"model": self.model_name, "team": self.team, "security": "true"},
                    annotations={
                        "action": "Review input sources and consider rate limiting",
                        "security_event": "Potential adversarial attack",
                    },
                ),
                AlertRule(
                    name=f"{self.model_name}_anomaly_spike",
                    expr=f'increase({self.namespace}_ml_security_events_total{{model_name="{self.model_name}",event_type="anomaly_detected"}}[15m]) > {anomaly_threshold}',
                    duration="5m",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.SECURITY,
                    summary=f"Anomaly spike detected for {self.model_name}",
                    description=f"More than {anomaly_threshold} anomalous inputs detected for model {self.model_name} in the last 15 minutes. "
                    "Review input data quality and distribution.",
                    labels={"model": self.model_name, "team": self.team},
                ),
                AlertRule(
                    name=f"{self.model_name}_data_poisoning_suspected",
                    expr=f'{self.namespace}_ml_security_events_total{{model_name="{self.model_name}",event_type="poisoning_suspected"}} > 0',
                    duration="0m",
                    severity=AlertSeverity.PAGE,
                    category=AlertCategory.SECURITY,
                    summary=f"Data poisoning suspected for {self.model_name}",
                    description=f"Potential data poisoning attack detected for model {self.model_name}. "
                    "Immediate investigation required. Consider suspending model updates.",
                    labels={"model": self.model_name, "team": self.team, "security": "true", "priority": "P1"},
                    annotations={
                        "action": "Suspend training pipeline and investigate data sources",
                        "fda_impact": "Poisoned model may produce unsafe outputs",
                        "security_event": "Critical - potential supply chain attack",
                    },
                ),
            ],
        )
        self._rule_groups.append(group)

    def add_availability_alerts(
        self,
        min_predictions_per_minute: int = 1,
        stale_threshold_minutes: int = 15,
    ) -> None:
        """
        Add availability and health alerting rules.

        Args:
            min_predictions_per_minute: Minimum expected prediction rate
            stale_threshold_minutes: Minutes without metrics before alert
        """
        group = AlertRuleGroup(
            name=f"{self.model_name}_availability_alerts",
            rules=[
                AlertRule(
                    name=f"{self.model_name}_not_serving",
                    expr=f'absent({self.namespace}_ml_model_predictions_total{{model_name="{self.model_name}"}})',
                    duration="5m",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.AVAILABILITY,
                    summary=f"Model {self.model_name} is not serving",
                    description=f"No predictions metric found for model {self.model_name}. "
                    "The model service may be down or not reporting metrics.",
                    labels={"model": self.model_name, "team": self.team},
                ),
                AlertRule(
                    name=f"{self.model_name}_low_throughput",
                    expr=f'rate({self.namespace}_ml_model_predictions_total{{model_name="{self.model_name}"}}[5m]) * 60 < {min_predictions_per_minute}',
                    duration="10m",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.AVAILABILITY,
                    summary=f"Low throughput for {self.model_name}",
                    description=f"Model {self.model_name} is receiving fewer than {min_predictions_per_minute} predictions per minute. "
                    "Expected traffic may have stopped or been redirected.",
                    labels={"model": self.model_name, "team": self.team},
                ),
                AlertRule(
                    name=f"{self.model_name}_metrics_stale",
                    expr=f'time() - {self.namespace}_ml_model_info{{model_name="{self.model_name}"}} > {stale_threshold_minutes * 60}',
                    duration="5m",
                    severity=AlertSeverity.WARNING,
                    category=AlertCategory.AVAILABILITY,
                    summary=f"Stale metrics for {self.model_name}",
                    description=f"Metrics for model {self.model_name} have not been updated in {stale_threshold_minutes} minutes. "
                    "Check the metrics exporter health.",
                    labels={"model": self.model_name, "team": self.team},
                ),
            ],
        )
        self._rule_groups.append(group)

    def add_compliance_alerts(self) -> None:
        """Add FDA compliance monitoring alerts."""
        group = AlertRuleGroup(
            name=f"{self.model_name}_compliance_alerts",
            rules=[
                AlertRule(
                    name=f"{self.model_name}_pccp_threshold_exceeded",
                    expr=f'{self.namespace}_ml_model_drift_score{{model_name="{self.model_name}"}} > 0.15 and '
                    f'{self.namespace}_ml_model_accuracy{{model_name="{self.model_name}"}} < 0.92',
                    duration="30m",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.COMPLIANCE,
                    summary=f"FDA PCCP threshold exceeded for {self.model_name}",
                    description=f"Model {self.model_name} has exceeded Predetermined Change Control Plan (PCCP) thresholds. "
                    "Drift combined with accuracy degradation may require regulatory notification.",
                    labels={"model": self.model_name, "team": self.team, "compliance": "fda"},
                    annotations={
                        "fda_impact": "May require PCCP notification or 510(k) amendment",
                        "action": "Evaluate against PCCP criteria and notify regulatory affairs",
                        "documentation": "Record incident in DHF for potential regulatory submission",
                    },
                ),
                AlertRule(
                    name=f"{self.model_name}_bias_detected",
                    expr=f'{self.namespace}_ml_model_bias_score{{model_name="{self.model_name}"}} > 0.1',
                    duration="15m",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.COMPLIANCE,
                    summary=f"Model bias detected for {self.model_name}",
                    description=f"Significant bias detected in model {self.model_name} predictions. "
                    "This may indicate discriminatory behavior requiring immediate review.",
                    labels={"model": self.model_name, "team": self.team, "compliance": "fda"},
                    annotations={
                        "fda_impact": "Bias may affect patient safety across demographic groups",
                        "action": "Conduct fairness review and document findings",
                    },
                ),
            ],
        )
        self._rule_groups.append(group)

    def add_custom_alert(
        self,
        name: str,
        expr: str,
        duration: str,
        severity: AlertSeverity,
        category: AlertCategory,
        summary: str,
        description: str,
        group_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Add a custom alerting rule.

        Args:
            name: Alert name
            expr: PromQL expression
            duration: Duration before firing
            severity: Alert severity
            category: Alert category
            summary: Short summary
            description: Detailed description
            group_name: Optional group name (creates new group if not exists)
            **kwargs: Additional labels and annotations
        """
        rule = AlertRule(
            name=name,
            expr=expr,
            duration=duration,
            severity=severity,
            category=category,
            summary=summary,
            description=description,
            labels=kwargs.get("labels", {}),
            annotations=kwargs.get("annotations", {}),
            runbook_url=kwargs.get("runbook_url", ""),
        )

        group_name = group_name or f"{self.model_name}_custom_alerts"

        # Find or create group
        for group in self._rule_groups:
            if group.name == group_name:
                group.rules.append(rule)
                return

        # Create new group
        self._rule_groups.append(AlertRuleGroup(name=group_name, rules=[rule]))

    def export_prometheus_rules(self) -> str:
        """
        Export rules in Prometheus alerting rules format (YAML).

        Returns:
            YAML formatted alerting rules
        """
        rules_config = {
            "groups": [group.to_prometheus_group() for group in self._rule_groups],
        }
        return yaml.dump(rules_config, default_flow_style=False, sort_keys=False)

    def export_alertmanager_config(
        self,
        slack_webhook: str | None = None,
        pagerduty_key: str | None = None,
        email_to: str | None = None,
    ) -> str:
        """
        Export Alertmanager routing configuration.

        Args:
            slack_webhook: Slack webhook URL for notifications
            pagerduty_key: PagerDuty integration key
            email_to: Email address for notifications

        Returns:
            YAML formatted Alertmanager config
        """
        receivers: list[dict[str, Any]] = []
        routes: list[dict[str, Any]] = []

        # Default receiver
        default_receiver: dict[str, Any] = {"name": "default"}
        if email_to:
            default_receiver["email_configs"] = [{"to": email_to}]
        receivers.append(default_receiver)

        # Critical alerts - PagerDuty
        if pagerduty_key:
            receivers.append({
                "name": "pagerduty-critical",
                "pagerduty_configs": [
                    {
                        "service_key": pagerduty_key,
                        "severity": "critical",
                    }
                ],
            })
            routes.append({
                "match": {"severity": "critical"},
                "receiver": "pagerduty-critical",
            })
            routes.append({
                "match": {"severity": "page"},
                "receiver": "pagerduty-critical",
            })

        # Slack for warnings
        if slack_webhook:
            receivers.append({
                "name": "slack-warnings",
                "slack_configs": [
                    {
                        "api_url": slack_webhook,
                        "channel": "#ml-alerts",
                        "title": "{{ .GroupLabels.alertname }}",
                        "text": "{{ .Annotations.description }}",
                    }
                ],
            })
            routes.append({
                "match": {"severity": "warning"},
                "receiver": "slack-warnings",
            })

        config = {
            "global": {
                "resolve_timeout": "5m",
            },
            "route": {
                "group_by": ["alertname", "model_name"],
                "group_wait": "30s",
                "group_interval": "5m",
                "repeat_interval": "4h",
                "receiver": "default",
                "routes": routes,
            },
            "receivers": receivers,
            "inhibit_rules": [
                {
                    "source_match": {"severity": "critical"},
                    "target_match": {"severity": "warning"},
                    "equal": ["alertname", "model_name"],
                }
            ],
        }

        return yaml.dump(config, default_flow_style=False, sort_keys=False)

    def export_json(self) -> str:
        """
        Export rules in JSON format.

        Returns:
            JSON formatted alerting rules
        """
        data = {
            "model_name": self.model_name,
            "namespace": self.namespace,
            "team": self.team,
            "environment": self.environment,
            "created": self._created,
            "rule_groups": [
                {
                    "name": group.name,
                    "interval": group.interval,
                    "rules": [
                        {
                            "name": rule.name,
                            "expr": rule.expr,
                            "duration": rule.duration,
                            "severity": rule.severity.value,
                            "category": rule.category.value,
                            "summary": rule.summary,
                            "description": rule.description,
                            "labels": rule.labels,
                            "annotations": rule.annotations,
                        }
                        for rule in group.rules
                    ],
                }
                for group in self._rule_groups
            ],
        }
        return json.dumps(data, indent=2)

    def get_rule_count(self) -> int:
        """Get total number of alerting rules."""
        return sum(len(group.rules) for group in self._rule_groups)

    def get_rules_by_severity(self, severity: AlertSeverity) -> list[AlertRule]:
        """Get all rules of a given severity."""
        rules = []
        for group in self._rule_groups:
            rules.extend(rule for rule in group.rules if rule.severity == severity)
        return rules

    def get_rules_by_category(self, category: AlertCategory) -> list[AlertRule]:
        """Get all rules of a given category."""
        rules = []
        for group in self._rule_groups:
            rules.extend(rule for rule in group.rules if rule.category == category)
        return rules
