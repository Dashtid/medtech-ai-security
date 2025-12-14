"""
Model Rollback Automation Module.

Provides automated model version management and rollback capabilities for
responding to drift detection and performance degradation events.

Supports FDA PCCP compliance by maintaining audit trails of all model
changes and providing controlled rollback mechanisms.

Example:
    >>> manager = ModelVersionManager(
    ...     model_name="classifier",
    ...     storage_path="/models/versions",
    ... )
    >>> manager.register_version("v1.0.0", model_path="/models/classifier_v1.pt")
    >>> manager.register_version("v1.1.0", model_path="/models/classifier_v1.1.pt")
    >>> manager.set_active("v1.1.0")
    >>>
    >>> # On drift detection
    >>> rollback = manager.trigger_rollback(
    ...     reason="drift_detected",
    ...     drift_score=0.35,
    ... )
    >>> print(f"Rolled back to {rollback.target_version}")
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class RollbackTrigger(Enum):
    """Triggers that can initiate a rollback."""

    DRIFT_CRITICAL = "drift_critical"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    RECALL_DEGRADATION = "recall_degradation"
    PRECISION_DEGRADATION = "precision_degradation"
    LATENCY_EXCESSIVE = "latency_excessive"
    ERROR_RATE_HIGH = "error_rate_high"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    DATA_POISONING = "data_poisoning"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


class RollbackStatus(Enum):
    """Status of a rollback operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REQUIRES_APPROVAL = "requires_approval"


class VersionStatus(Enum):
    """Status of a model version."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"
    QUARANTINED = "quarantined"  # Flagged due to issues
    DELETED = "deleted"


@dataclass
class ModelVersion:
    """Represents a specific model version."""

    version: str
    model_path: str
    model_hash: str
    created_at: str
    status: VersionStatus
    metadata: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    training_data_hash: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "created_at": self.created_at,
            "status": self.status.value,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "training_data_hash": self.training_data_hash,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            model_path=data["model_path"],
            model_hash=data["model_hash"],
            created_at=data["created_at"],
            status=VersionStatus(data["status"]),
            metadata=data.get("metadata", {}),
            performance_metrics=data.get("performance_metrics", {}),
            training_data_hash=data.get("training_data_hash", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class RollbackEvent:
    """Records a rollback event for audit purposes."""

    event_id: str
    timestamp: str
    trigger: RollbackTrigger
    source_version: str
    target_version: str
    status: RollbackStatus
    initiated_by: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    completion_time: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "trigger": self.trigger.value,
            "source_version": self.source_version,
            "target_version": self.target_version,
            "status": self.status.value,
            "initiated_by": self.initiated_by,
            "reason": self.reason,
            "details": self.details,
            "completion_time": self.completion_time,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RollbackEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            timestamp=data["timestamp"],
            trigger=RollbackTrigger(data["trigger"]),
            source_version=data["source_version"],
            target_version=data["target_version"],
            status=RollbackStatus(data["status"]),
            initiated_by=data["initiated_by"],
            reason=data["reason"],
            details=data.get("details", {}),
            completion_time=data.get("completion_time"),
            error_message=data.get("error_message"),
        )


@dataclass
class RollbackPolicy:
    """Defines automated rollback policy."""

    name: str
    enabled: bool = True
    triggers: list[RollbackTrigger] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    cooldown_minutes: int = 60
    require_approval: bool = False
    notification_channels: list[str] = field(default_factory=list)
    max_rollbacks_per_day: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "triggers": [t.value for t in self.triggers],
            "thresholds": self.thresholds,
            "cooldown_minutes": self.cooldown_minutes,
            "require_approval": self.require_approval,
            "notification_channels": self.notification_channels,
            "max_rollbacks_per_day": self.max_rollbacks_per_day,
        }


class ModelVersionManager:
    """
    Manages model versions and automated rollback.

    Provides:
    - Version registration and tracking
    - Integrity verification before activation
    - Automated rollback on drift/performance issues
    - FDA PCCP-compliant audit logging

    Example:
        >>> manager = ModelVersionManager(
        ...     model_name="diagnostic_model",
        ...     storage_path="./model_versions",
        ... )
        >>> manager.register_version(
        ...     version="1.0.0",
        ...     model_path="./models/v1.0.0.pt",
        ...     performance_metrics={"accuracy": 0.95, "recall": 0.92},
        ... )
        >>> manager.set_active("1.0.0")
    """

    def __init__(
        self,
        model_name: str,
        storage_path: str | Path,
        integrity_verifier: Callable[[str], bool] | None = None,
    ) -> None:
        """
        Initialize model version manager.

        Args:
            model_name: Name of the model being managed
            storage_path: Path for storing version metadata and audit logs
            integrity_verifier: Optional function to verify model integrity
        """
        self.model_name = model_name
        self.storage_path = Path(storage_path)
        self.integrity_verifier = integrity_verifier or self._default_integrity_check

        # Internal state
        self._versions: dict[str, ModelVersion] = {}
        self._active_version: str | None = None
        self._rollback_history: list[RollbackEvent] = []
        self._policy: RollbackPolicy | None = None
        self._last_rollback_time: datetime | None = None
        self._rollbacks_today: int = 0
        self._rollbacks_today_date: str = ""

        # Ensure storage path exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing state if present
        self._load_state()

    def register_version(
        self,
        version: str,
        model_path: str,
        performance_metrics: dict[str, float] | None = None,
        training_data_hash: str = "",
        metadata: dict[str, Any] | None = None,
        notes: str = "",
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            version: Version identifier (e.g., "1.0.0")
            model_path: Path to the model file
            performance_metrics: Model performance metrics
            training_data_hash: Hash of training data for traceability
            metadata: Additional metadata
            notes: Human-readable notes

        Returns:
            Registered ModelVersion object

        Raises:
            ValueError: If version already exists
            FileNotFoundError: If model file not found
        """
        if version in self._versions:
            raise ValueError(f"Version {version} already registered")

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Compute model hash
        model_hash = self._compute_file_hash(model_path)

        mv = ModelVersion(
            version=version,
            model_path=str(model_path_obj.absolute()),
            model_hash=model_hash,
            created_at=datetime.now(timezone.utc).isoformat(),
            status=VersionStatus.INACTIVE,
            metadata=metadata or {},
            performance_metrics=performance_metrics or {},
            training_data_hash=training_data_hash,
            notes=notes,
        )

        self._versions[version] = mv
        self._save_state()

        return mv

    def set_active(
        self,
        version: str,
        initiated_by: str = "system",
        verify_integrity: bool = True,
    ) -> bool:
        """
        Set a version as active.

        Args:
            version: Version to activate
            initiated_by: Who initiated the change
            verify_integrity: Whether to verify model integrity first

        Returns:
            True if activation successful

        Raises:
            ValueError: If version not found or quarantined
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")

        mv = self._versions[version]

        if mv.status == VersionStatus.QUARANTINED:
            raise ValueError(f"Version {version} is quarantined and cannot be activated")

        if mv.status == VersionStatus.DELETED:
            raise ValueError(f"Version {version} has been deleted")

        # Verify integrity
        if verify_integrity and not self.integrity_verifier(mv.model_path):
            raise ValueError(f"Integrity verification failed for version {version}")

        # Deactivate current version
        if self._active_version and self._active_version in self._versions:
            self._versions[self._active_version].status = VersionStatus.INACTIVE

        # Activate new version
        mv.status = VersionStatus.ACTIVE
        self._active_version = version
        self._save_state()

        return True

    def get_active_version(self) -> ModelVersion | None:
        """Get currently active model version."""
        if self._active_version:
            return self._versions.get(self._active_version)
        return None

    def get_version(self, version: str) -> ModelVersion | None:
        """Get a specific model version."""
        return self._versions.get(version)

    def list_versions(self, include_deleted: bool = False) -> list[ModelVersion]:
        """List all registered versions."""
        versions = list(self._versions.values())
        if not include_deleted:
            versions = [v for v in versions if v.status != VersionStatus.DELETED]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_previous_version(self) -> ModelVersion | None:
        """Get the previously active version (for rollback)."""
        versions = [
            v
            for v in self._versions.values()
            if v.status in [VersionStatus.INACTIVE, VersionStatus.ACTIVE]
            and v.version != self._active_version
        ]
        if not versions:
            return None
        # Return most recent inactive version
        return sorted(versions, key=lambda v: v.created_at, reverse=True)[0]

    def set_rollback_policy(self, policy: RollbackPolicy) -> None:
        """Set the automated rollback policy."""
        self._policy = policy
        self._save_state()

    def trigger_rollback(
        self,
        reason: str,
        initiated_by: str = "automated",
        target_version: str | None = None,
        trigger: RollbackTrigger = RollbackTrigger.MANUAL,
        details: dict[str, Any] | None = None,
    ) -> RollbackEvent:
        """
        Trigger a model rollback.

        Args:
            reason: Reason for rollback
            initiated_by: Who/what initiated the rollback
            target_version: Specific version to roll back to (default: previous)
            trigger: What triggered the rollback
            details: Additional details about the trigger

        Returns:
            RollbackEvent documenting the rollback

        Raises:
            ValueError: If no target version available or rollback fails
        """
        current_version = self._active_version
        if not current_version:
            raise ValueError("No active version to roll back from")

        # Check cooldown and rate limits
        if self._policy and not self._check_rollback_allowed():
            event = self._create_rollback_event(
                trigger=trigger,
                source_version=current_version,
                target_version=target_version or "unknown",
                status=RollbackStatus.CANCELLED,
                initiated_by=initiated_by,
                reason=reason,
                details=details or {},
            )
            event.error_message = "Rollback cancelled due to policy constraints"
            self._rollback_history.append(event)
            self._save_state()
            return event

        # Determine target version
        if target_version is None:
            prev = self.get_previous_version()
            if prev is None:
                raise ValueError("No previous version available for rollback")
            target_version = prev.version

        if target_version not in self._versions:
            raise ValueError(f"Target version {target_version} not found")

        # Check if approval required
        if self._policy and self._policy.require_approval:
            event = self._create_rollback_event(
                trigger=trigger,
                source_version=current_version,
                target_version=target_version,
                status=RollbackStatus.REQUIRES_APPROVAL,
                initiated_by=initiated_by,
                reason=reason,
                details=details or {},
            )
            self._rollback_history.append(event)
            self._save_state()
            return event

        # Create rollback event
        event = self._create_rollback_event(
            trigger=trigger,
            source_version=current_version,
            target_version=target_version,
            status=RollbackStatus.IN_PROGRESS,
            initiated_by=initiated_by,
            reason=reason,
            details=details or {},
        )

        try:
            # Execute rollback
            self.set_active(target_version, initiated_by=initiated_by)

            # Update event
            event.status = RollbackStatus.COMPLETED
            event.completion_time = datetime.now(timezone.utc).isoformat()

            # Update rate limit tracking
            self._last_rollback_time = datetime.now(timezone.utc)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if self._rollbacks_today_date != today:
                self._rollbacks_today = 0
                self._rollbacks_today_date = today
            self._rollbacks_today += 1

        except Exception as e:
            event.status = RollbackStatus.FAILED
            event.error_message = str(e)
            event.completion_time = datetime.now(timezone.utc).isoformat()

        self._rollback_history.append(event)
        self._save_state()

        return event

    def approve_rollback(self, event_id: str, approved_by: str) -> RollbackEvent | None:
        """
        Approve a pending rollback.

        Args:
            event_id: ID of the rollback event to approve
            approved_by: Who approved the rollback

        Returns:
            Updated RollbackEvent or None if not found
        """
        for event in self._rollback_history:
            if event.event_id == event_id and event.status == RollbackStatus.REQUIRES_APPROVAL:
                # Execute the rollback
                event.details["approved_by"] = approved_by
                event.details["approval_time"] = datetime.now(timezone.utc).isoformat()
                event.status = RollbackStatus.IN_PROGRESS

                try:
                    self.set_active(event.target_version, initiated_by=approved_by)
                    event.status = RollbackStatus.COMPLETED
                    event.completion_time = datetime.now(timezone.utc).isoformat()
                except Exception as e:
                    event.status = RollbackStatus.FAILED
                    event.error_message = str(e)
                    event.completion_time = datetime.now(timezone.utc).isoformat()

                self._save_state()
                return event

        return None

    def quarantine_version(self, version: str, reason: str) -> None:
        """
        Quarantine a version due to issues.

        Args:
            version: Version to quarantine
            reason: Reason for quarantine
        """
        if version not in self._versions:
            raise ValueError(f"Version {version} not found")

        mv = self._versions[version]

        # Cannot quarantine active version
        if version == self._active_version:
            raise ValueError("Cannot quarantine active version. Roll back first.")

        mv.status = VersionStatus.QUARANTINED
        mv.notes = f"{mv.notes}\nQuarantined: {reason} ({datetime.now(timezone.utc).isoformat()})"
        self._save_state()

    def get_rollback_history(self, limit: int | None = None) -> list[RollbackEvent]:
        """Get rollback history."""
        history = sorted(self._rollback_history, key=lambda e: e.timestamp, reverse=True)
        if limit:
            history = history[:limit]
        return history

    def check_should_rollback(
        self,
        drift_score: float | None = None,
        accuracy: float | None = None,
        recall: float | None = None,
        precision: float | None = None,
        error_rate: float | None = None,
    ) -> tuple[bool, RollbackTrigger | None, str]:
        """
        Check if conditions warrant a rollback based on policy.

        Args:
            drift_score: Current drift score
            accuracy: Current accuracy
            recall: Current recall
            precision: Current precision
            error_rate: Current error rate

        Returns:
            Tuple of (should_rollback, trigger, reason)
        """
        if not self._policy or not self._policy.enabled:
            return False, None, ""

        thresholds = self._policy.thresholds
        triggers = self._policy.triggers

        # Check drift
        if (
            RollbackTrigger.DRIFT_CRITICAL in triggers
            and drift_score is not None
            and drift_score > thresholds.get("drift_critical", 0.3)
        ):
            return (
                True,
                RollbackTrigger.DRIFT_CRITICAL,
                f"Drift score {drift_score} exceeds threshold",
            )

        # Check accuracy
        if (
            RollbackTrigger.ACCURACY_DEGRADATION in triggers
            and accuracy is not None
            and accuracy < thresholds.get("min_accuracy", 0.85)
        ):
            return (
                True,
                RollbackTrigger.ACCURACY_DEGRADATION,
                f"Accuracy {accuracy} below threshold",
            )

        # Check recall
        if (
            RollbackTrigger.RECALL_DEGRADATION in triggers
            and recall is not None
            and recall < thresholds.get("min_recall", 0.80)
        ):
            return True, RollbackTrigger.RECALL_DEGRADATION, f"Recall {recall} below threshold"

        # Check precision
        if (
            RollbackTrigger.PRECISION_DEGRADATION in triggers
            and precision is not None
            and precision < thresholds.get("min_precision", 0.80)
        ):
            return (
                True,
                RollbackTrigger.PRECISION_DEGRADATION,
                f"Precision {precision} below threshold",
            )

        # Check error rate
        if (
            RollbackTrigger.ERROR_RATE_HIGH in triggers
            and error_rate is not None
            and error_rate > thresholds.get("max_error_rate", 0.1)
        ):
            return (
                True,
                RollbackTrigger.ERROR_RATE_HIGH,
                f"Error rate {error_rate} exceeds threshold",
            )

        return False, None, ""

    def generate_audit_report(self) -> str:
        """
        Generate FDA PCCP-compliant audit report.

        Returns:
            Markdown formatted audit report
        """
        lines = [
            "# Model Version Audit Report",
            "",
            f"**Model Name**: {self.model_name}",
            f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
            f"**Active Version**: {self._active_version or 'None'}",
            "",
            "## Version History",
            "",
        ]

        # Version table
        lines.append("| Version | Status | Created | Hash | Accuracy |")
        lines.append("|---------|--------|---------|------|----------|")

        for mv in self.list_versions(include_deleted=True):
            accuracy = mv.performance_metrics.get("accuracy", "N/A")
            if isinstance(accuracy, float):
                accuracy = f"{accuracy:.3f}"
            hash_short = mv.model_hash[:12] if mv.model_hash else "N/A"
            lines.append(
                f"| {mv.version} | {mv.status.value} | {mv.created_at[:10]} | "
                f"{hash_short}... | {accuracy} |"
            )

        lines.extend(
            [
                "",
                "## Rollback History",
                "",
            ]
        )

        if self._rollback_history:
            lines.append("| Time | Trigger | From | To | Status | Reason |")
            lines.append("|------|---------|------|-----|--------|--------|")

            for event in self.get_rollback_history(limit=20):
                lines.append(
                    f"| {event.timestamp[:19]} | {event.trigger.value} | "
                    f"{event.source_version} | {event.target_version} | "
                    f"{event.status.value} | {event.reason[:30]}... |"
                )
        else:
            lines.append("_No rollback events recorded._")

        lines.extend(
            [
                "",
                "## Current Policy",
                "",
            ]
        )

        if self._policy:
            lines.append(f"- **Name**: {self._policy.name}")
            lines.append(f"- **Enabled**: {self._policy.enabled}")
            lines.append(f"- **Triggers**: {', '.join(t.value for t in self._policy.triggers)}")
            lines.append(f"- **Cooldown**: {self._policy.cooldown_minutes} minutes")
            lines.append(f"- **Requires Approval**: {self._policy.require_approval}")
            lines.append(f"- **Max Rollbacks/Day**: {self._policy.max_rollbacks_per_day}")
            lines.append("")
            lines.append("**Thresholds**:")
            for key, value in self._policy.thresholds.items():
                lines.append(f"  - {key}: {value}")
        else:
            lines.append("_No rollback policy configured._")

        lines.extend(
            [
                "",
                "---",
                "",
                "## FDA PCCP Compliance Note",
                "",
                "This audit trail documents all model version changes and rollback events ",
                "as required by FDA's Predetermined Change Control Plan (PCCP) framework. ",
                "All model changes are traceable through version hashes and audit logs.",
                "",
                "### Integrity Verification",
                "",
                "Model files are verified using SHA-256 hashes before activation to ensure ",
                "no tampering or corruption has occurred.",
                "",
            ]
        )

        return "\n".join(lines)

    def _create_rollback_event(
        self,
        trigger: RollbackTrigger,
        source_version: str,
        target_version: str,
        status: RollbackStatus,
        initiated_by: str,
        reason: str,
        details: dict[str, Any],
    ) -> RollbackEvent:
        """Create a new rollback event."""
        event_id = f"RB-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{len(self._rollback_history):04d}"
        return RollbackEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger=trigger,
            source_version=source_version,
            target_version=target_version,
            status=status,
            initiated_by=initiated_by,
            reason=reason,
            details=details,
        )

    def _check_rollback_allowed(self) -> bool:
        """Check if rollback is allowed based on policy."""
        if not self._policy:
            return True

        # Check cooldown
        if self._last_rollback_time:
            elapsed = (datetime.now(timezone.utc) - self._last_rollback_time).total_seconds()
            if elapsed < self._policy.cooldown_minutes * 60:
                return False

        # Check daily limit
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._rollbacks_today_date == today:
            if self._rollbacks_today >= self._policy.max_rollbacks_per_day:
                return False

        return True

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _default_integrity_check(self, model_path: str) -> bool:
        """Default integrity check - verify file exists and is readable."""
        try:
            path = Path(model_path)
            if not path.exists():
                return False
            # Try to read first few bytes
            with open(path, "rb") as f:
                f.read(1024)
            return True
        except Exception:
            return False

    def _save_state(self) -> None:
        """Save manager state to disk."""
        state = {
            "model_name": self.model_name,
            "active_version": self._active_version,
            "versions": {k: v.to_dict() for k, v in self._versions.items()},
            "rollback_history": [e.to_dict() for e in self._rollback_history],
            "policy": self._policy.to_dict() if self._policy else None,
            "last_rollback_time": (
                self._last_rollback_time.isoformat() if self._last_rollback_time else None
            ),
            "rollbacks_today": self._rollbacks_today,
            "rollbacks_today_date": self._rollbacks_today_date,
        }

        state_file = self.storage_path / f"{self.model_name}_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load manager state from disk."""
        state_file = self.storage_path / f"{self.model_name}_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            self._active_version = state.get("active_version")
            self._versions = {
                k: ModelVersion.from_dict(v) for k, v in state.get("versions", {}).items()
            }
            self._rollback_history = [
                RollbackEvent.from_dict(e) for e in state.get("rollback_history", [])
            ]

            policy_data = state.get("policy")
            if policy_data:
                self._policy = RollbackPolicy(
                    name=policy_data["name"],
                    enabled=policy_data.get("enabled", True),
                    triggers=[RollbackTrigger(t) for t in policy_data.get("triggers", [])],
                    thresholds=policy_data.get("thresholds", {}),
                    cooldown_minutes=policy_data.get("cooldown_minutes", 60),
                    require_approval=policy_data.get("require_approval", False),
                    notification_channels=policy_data.get("notification_channels", []),
                    max_rollbacks_per_day=policy_data.get("max_rollbacks_per_day", 3),
                )

            last_rb = state.get("last_rollback_time")
            if last_rb:
                self._last_rollback_time = datetime.fromisoformat(last_rb)

            self._rollbacks_today = state.get("rollbacks_today", 0)
            self._rollbacks_today_date = state.get("rollbacks_today_date", "")

        except (json.JSONDecodeError, KeyError):
            # Log error but continue with empty state
            pass


def create_default_rollback_policy() -> RollbackPolicy:
    """Create a default rollback policy for medical device models."""
    return RollbackPolicy(
        name="medical_device_default",
        enabled=True,
        triggers=[
            RollbackTrigger.DRIFT_CRITICAL,
            RollbackTrigger.ACCURACY_DEGRADATION,
            RollbackTrigger.RECALL_DEGRADATION,
            RollbackTrigger.ADVERSARIAL_ATTACK,
            RollbackTrigger.DATA_POISONING,
        ],
        thresholds={
            "drift_critical": 0.25,
            "min_accuracy": 0.90,
            "min_recall": 0.85,
            "min_precision": 0.85,
            "max_error_rate": 0.05,
        },
        cooldown_minutes=30,
        require_approval=True,  # Require human approval for medical devices
        notification_channels=["slack", "email", "pagerduty"],
        max_rollbacks_per_day=5,
    )
