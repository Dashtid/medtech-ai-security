"""
Tests for Model Rollback Automation Module.

Tests model version management, rollback triggers, and FDA PCCP compliance.
"""

import json
import tempfile
from pathlib import Path

import pytest

from medtech_ai_security.ml.rollback import (
    ModelVersion,
    ModelVersionManager,
    RollbackEvent,
    RollbackPolicy,
    RollbackStatus,
    RollbackTrigger,
    VersionStatus,
    create_default_rollback_policy,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    return tmp_path / "model_versions"


@pytest.fixture
def sample_model_file(tmp_path):
    """Create a sample model file."""
    model_path = tmp_path / "model_v1.pt"
    model_path.write_bytes(b"fake model content for testing")
    return model_path


@pytest.fixture
def sample_model_files(tmp_path):
    """Create multiple sample model files."""
    files = {}
    for version in ["1.0.0", "1.1.0", "1.2.0"]:
        model_path = tmp_path / f"model_v{version}.pt"
        model_path.write_bytes(f"fake model content v{version}".encode())
        files[version] = model_path
    return files


@pytest.fixture
def manager(temp_storage, sample_model_file):
    """Create a ModelVersionManager with one registered version."""
    mgr = ModelVersionManager(
        model_name="test_model",
        storage_path=temp_storage,
    )
    mgr.register_version(
        version="1.0.0",
        model_path=str(sample_model_file),
        performance_metrics={"accuracy": 0.95, "recall": 0.92},
    )
    mgr.set_active("1.0.0")
    return mgr


@pytest.fixture
def manager_with_versions(temp_storage, sample_model_files):
    """Create a ModelVersionManager with multiple versions."""
    mgr = ModelVersionManager(
        model_name="test_model",
        storage_path=temp_storage,
    )
    for version, path in sample_model_files.items():
        mgr.register_version(
            version=version,
            model_path=str(path),
            performance_metrics={"accuracy": 0.95 - float(version.split(".")[1]) * 0.01},
        )
    mgr.set_active("1.2.0")
    return mgr


# =============================================================================
# ModelVersion Tests
# =============================================================================


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_creation(self):
        """Test creating a ModelVersion."""
        mv = ModelVersion(
            version="1.0.0",
            model_path="/path/to/model.pt",
            model_hash="abc123",
            created_at="2024-01-01T00:00:00Z",
            status=VersionStatus.ACTIVE,
        )
        assert mv.version == "1.0.0"
        assert mv.status == VersionStatus.ACTIVE

    def test_to_dict(self):
        """Test converting ModelVersion to dictionary."""
        mv = ModelVersion(
            version="1.0.0",
            model_path="/path/to/model.pt",
            model_hash="abc123",
            created_at="2024-01-01T00:00:00Z",
            status=VersionStatus.ACTIVE,
            performance_metrics={"accuracy": 0.95},
        )
        data = mv.to_dict()
        assert data["version"] == "1.0.0"
        assert data["status"] == "active"
        assert data["performance_metrics"]["accuracy"] == 0.95

    def test_from_dict(self):
        """Test creating ModelVersion from dictionary."""
        data = {
            "version": "1.0.0",
            "model_path": "/path/to/model.pt",
            "model_hash": "abc123",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "active",
            "metadata": {},
            "performance_metrics": {"accuracy": 0.95},
        }
        mv = ModelVersion.from_dict(data)
        assert mv.version == "1.0.0"
        assert mv.status == VersionStatus.ACTIVE


# =============================================================================
# RollbackEvent Tests
# =============================================================================


class TestRollbackEvent:
    """Tests for RollbackEvent dataclass."""

    def test_creation(self):
        """Test creating a RollbackEvent."""
        event = RollbackEvent(
            event_id="RB-001",
            timestamp="2024-01-01T00:00:00Z",
            trigger=RollbackTrigger.DRIFT_CRITICAL,
            source_version="1.1.0",
            target_version="1.0.0",
            status=RollbackStatus.COMPLETED,
            initiated_by="system",
            reason="Drift detected",
        )
        assert event.event_id == "RB-001"
        assert event.trigger == RollbackTrigger.DRIFT_CRITICAL
        assert event.status == RollbackStatus.COMPLETED

    def test_to_dict(self):
        """Test converting RollbackEvent to dictionary."""
        event = RollbackEvent(
            event_id="RB-001",
            timestamp="2024-01-01T00:00:00Z",
            trigger=RollbackTrigger.MANUAL,
            source_version="1.1.0",
            target_version="1.0.0",
            status=RollbackStatus.COMPLETED,
            initiated_by="user",
            reason="Manual rollback",
        )
        data = event.to_dict()
        assert data["event_id"] == "RB-001"
        assert data["trigger"] == "manual"
        assert data["status"] == "completed"

    def test_from_dict(self):
        """Test creating RollbackEvent from dictionary."""
        data = {
            "event_id": "RB-001",
            "timestamp": "2024-01-01T00:00:00Z",
            "trigger": "drift_critical",
            "source_version": "1.1.0",
            "target_version": "1.0.0",
            "status": "completed",
            "initiated_by": "system",
            "reason": "Test",
        }
        event = RollbackEvent.from_dict(data)
        assert event.trigger == RollbackTrigger.DRIFT_CRITICAL
        assert event.status == RollbackStatus.COMPLETED


# =============================================================================
# RollbackPolicy Tests
# =============================================================================


class TestRollbackPolicy:
    """Tests for RollbackPolicy dataclass."""

    def test_creation(self):
        """Test creating a RollbackPolicy."""
        policy = RollbackPolicy(
            name="test_policy",
            triggers=[RollbackTrigger.DRIFT_CRITICAL, RollbackTrigger.ACCURACY_DEGRADATION],
            thresholds={"drift_critical": 0.3, "min_accuracy": 0.90},
        )
        assert policy.name == "test_policy"
        assert len(policy.triggers) == 2
        assert policy.thresholds["drift_critical"] == 0.3

    def test_default_policy(self):
        """Test creating default rollback policy."""
        policy = create_default_rollback_policy()
        assert policy.name == "medical_device_default"
        assert policy.require_approval is True
        assert RollbackTrigger.DRIFT_CRITICAL in policy.triggers
        assert "drift_critical" in policy.thresholds

    def test_to_dict(self):
        """Test converting RollbackPolicy to dictionary."""
        policy = RollbackPolicy(
            name="test_policy",
            triggers=[RollbackTrigger.MANUAL],
            thresholds={"test": 1.0},
        )
        data = policy.to_dict()
        assert data["name"] == "test_policy"
        assert "manual" in data["triggers"]


# =============================================================================
# ModelVersionManager Tests
# =============================================================================


class TestModelVersionManager:
    """Tests for ModelVersionManager class."""

    def test_initialization(self, temp_storage):
        """Test manager initialization."""
        mgr = ModelVersionManager(
            model_name="test_model",
            storage_path=temp_storage,
        )
        assert mgr.model_name == "test_model"
        assert mgr.storage_path.exists()

    def test_register_version(self, temp_storage, sample_model_file):
        """Test registering a model version."""
        mgr = ModelVersionManager(
            model_name="test_model",
            storage_path=temp_storage,
        )
        mv = mgr.register_version(
            version="1.0.0",
            model_path=str(sample_model_file),
            performance_metrics={"accuracy": 0.95},
        )
        assert mv.version == "1.0.0"
        assert mv.status == VersionStatus.INACTIVE
        assert mv.model_hash is not None

    def test_register_duplicate_version(self, manager, sample_model_file):
        """Test registering duplicate version raises error."""
        with pytest.raises(ValueError, match="already registered"):
            manager.register_version(
                version="1.0.0",
                model_path=str(sample_model_file),
            )

    def test_register_nonexistent_file(self, temp_storage):
        """Test registering nonexistent file raises error."""
        mgr = ModelVersionManager(
            model_name="test_model",
            storage_path=temp_storage,
        )
        with pytest.raises(FileNotFoundError):
            mgr.register_version(
                version="1.0.0",
                model_path="/nonexistent/model.pt",
            )

    def test_set_active(self, manager):
        """Test setting active version."""
        active = manager.get_active_version()
        assert active is not None
        assert active.version == "1.0.0"
        assert active.status == VersionStatus.ACTIVE

    def test_set_active_nonexistent(self, manager):
        """Test setting nonexistent version as active."""
        with pytest.raises(ValueError, match="not found"):
            manager.set_active("9.9.9")

    def test_get_version(self, manager):
        """Test getting a specific version."""
        mv = manager.get_version("1.0.0")
        assert mv is not None
        assert mv.version == "1.0.0"

    def test_list_versions(self, manager_with_versions):
        """Test listing all versions."""
        versions = manager_with_versions.list_versions()
        assert len(versions) == 3
        # Should be sorted by creation time (newest first)
        assert versions[0].version == "1.2.0"

    def test_get_previous_version(self, manager_with_versions):
        """Test getting previous version for rollback."""
        prev = manager_with_versions.get_previous_version()
        assert prev is not None
        assert prev.version in ["1.0.0", "1.1.0"]


class TestModelVersionManagerRollback:
    """Tests for rollback functionality."""

    def test_trigger_rollback(self, manager_with_versions):
        """Test triggering a rollback."""
        active_before = manager_with_versions.get_active_version()
        assert active_before.version == "1.2.0"

        event = manager_with_versions.trigger_rollback(
            reason="Test rollback",
            initiated_by="test",
            trigger=RollbackTrigger.MANUAL,
        )

        assert event.status == RollbackStatus.COMPLETED
        assert event.source_version == "1.2.0"

        active_after = manager_with_versions.get_active_version()
        assert active_after.version != "1.2.0"

    def test_trigger_rollback_to_specific_version(self, manager_with_versions):
        """Test rolling back to a specific version."""
        event = manager_with_versions.trigger_rollback(
            reason="Rollback to specific version",
            target_version="1.0.0",
            trigger=RollbackTrigger.MANUAL,
        )

        assert event.status == RollbackStatus.COMPLETED
        assert event.target_version == "1.0.0"

        active = manager_with_versions.get_active_version()
        assert active.version == "1.0.0"

    def test_rollback_no_previous_version(self, manager):
        """Test rollback when no previous version available."""
        # Manager has only one version (1.0.0)
        with pytest.raises(ValueError, match="No previous version"):
            manager.trigger_rollback(
                reason="Test",
                trigger=RollbackTrigger.MANUAL,
            )

    def test_rollback_history(self, manager_with_versions):
        """Test rollback history tracking."""
        manager_with_versions.trigger_rollback(
            reason="First rollback",
            trigger=RollbackTrigger.DRIFT_CRITICAL,
        )

        history = manager_with_versions.get_rollback_history()
        assert len(history) == 1
        assert history[0].trigger == RollbackTrigger.DRIFT_CRITICAL

    def test_rollback_with_policy_approval(self, manager_with_versions):
        """Test rollback requiring approval."""
        policy = RollbackPolicy(
            name="approval_required",
            enabled=True,
            triggers=[RollbackTrigger.MANUAL],
            require_approval=True,
        )
        manager_with_versions.set_rollback_policy(policy)

        event = manager_with_versions.trigger_rollback(
            reason="Test with approval",
            trigger=RollbackTrigger.MANUAL,
        )

        assert event.status == RollbackStatus.REQUIRES_APPROVAL

        # Active version should not have changed
        active = manager_with_versions.get_active_version()
        assert active.version == "1.2.0"

    def test_approve_rollback(self, manager_with_versions):
        """Test approving a pending rollback."""
        policy = RollbackPolicy(
            name="approval_required",
            enabled=True,
            triggers=[RollbackTrigger.MANUAL],
            require_approval=True,
        )
        manager_with_versions.set_rollback_policy(policy)

        event = manager_with_versions.trigger_rollback(
            reason="Test with approval",
            trigger=RollbackTrigger.MANUAL,
        )

        # Approve the rollback
        approved_event = manager_with_versions.approve_rollback(
            event_id=event.event_id,
            approved_by="admin",
        )

        assert approved_event.status == RollbackStatus.COMPLETED
        assert "approved_by" in approved_event.details


class TestRollbackPolicyEnforcement:
    """Tests for rollback policy enforcement."""

    def test_check_should_rollback_drift(self, manager_with_versions):
        """Test drift-triggered rollback check."""
        policy = create_default_rollback_policy()
        manager_with_versions.set_rollback_policy(policy)

        should, trigger, reason = manager_with_versions.check_should_rollback(
            drift_score=0.35,  # Above threshold of 0.25
        )

        assert should is True
        assert trigger == RollbackTrigger.DRIFT_CRITICAL
        assert "drift" in reason.lower()

    def test_check_should_rollback_accuracy(self, manager_with_versions):
        """Test accuracy-triggered rollback check."""
        policy = create_default_rollback_policy()
        manager_with_versions.set_rollback_policy(policy)

        should, trigger, reason = manager_with_versions.check_should_rollback(
            accuracy=0.85,  # Below threshold of 0.90
        )

        assert should is True
        assert trigger == RollbackTrigger.ACCURACY_DEGRADATION

    def test_check_should_rollback_recall(self, manager_with_versions):
        """Test recall-triggered rollback check."""
        policy = create_default_rollback_policy()
        manager_with_versions.set_rollback_policy(policy)

        should, trigger, reason = manager_with_versions.check_should_rollback(
            recall=0.80,  # Below threshold of 0.85
        )

        assert should is True
        assert trigger == RollbackTrigger.RECALL_DEGRADATION

    def test_check_should_not_rollback(self, manager_with_versions):
        """Test no rollback when metrics are healthy."""
        policy = create_default_rollback_policy()
        manager_with_versions.set_rollback_policy(policy)

        should, trigger, reason = manager_with_versions.check_should_rollback(
            drift_score=0.05,
            accuracy=0.95,
            recall=0.92,
        )

        assert should is False
        assert trigger is None


class TestVersionQuarantine:
    """Tests for version quarantine functionality."""

    def test_quarantine_version(self, manager_with_versions):
        """Test quarantining a version."""
        manager_with_versions.quarantine_version(
            version="1.0.0",
            reason="Security vulnerability detected",
        )

        mv = manager_with_versions.get_version("1.0.0")
        assert mv.status == VersionStatus.QUARANTINED
        assert "Quarantined" in mv.notes

    def test_cannot_quarantine_active(self, manager_with_versions):
        """Test cannot quarantine active version."""
        with pytest.raises(ValueError, match="Cannot quarantine active"):
            manager_with_versions.quarantine_version(
                version="1.2.0",
                reason="Test",
            )

    def test_cannot_activate_quarantined(self, manager_with_versions):
        """Test cannot activate quarantined version."""
        manager_with_versions.quarantine_version(
            version="1.0.0",
            reason="Test",
        )

        with pytest.raises(ValueError, match="quarantined"):
            manager_with_versions.set_active("1.0.0")


class TestStateePersistence:
    """Tests for state persistence."""

    def test_state_persisted(self, temp_storage, sample_model_files):
        """Test that state is persisted and loaded correctly."""
        # Create manager and register versions
        mgr1 = ModelVersionManager(
            model_name="persist_test",
            storage_path=temp_storage,
        )
        mgr1.register_version("1.0.0", str(sample_model_files["1.0.0"]))
        mgr1.set_active("1.0.0")

        # Create new manager instance (should load state)
        mgr2 = ModelVersionManager(
            model_name="persist_test",
            storage_path=temp_storage,
        )

        assert mgr2.get_active_version().version == "1.0.0"
        versions = mgr2.list_versions()
        assert len(versions) == 1

    def test_rollback_history_persisted(self, temp_storage, sample_model_files):
        """Test that rollback history is persisted."""
        mgr1 = ModelVersionManager(
            model_name="history_test",
            storage_path=temp_storage,
        )
        mgr1.register_version("1.0.0", str(sample_model_files["1.0.0"]))
        mgr1.register_version("1.1.0", str(sample_model_files["1.1.0"]))
        mgr1.set_active("1.1.0")
        mgr1.trigger_rollback(reason="Test", trigger=RollbackTrigger.MANUAL)

        # Create new manager instance
        mgr2 = ModelVersionManager(
            model_name="history_test",
            storage_path=temp_storage,
        )

        history = mgr2.get_rollback_history()
        assert len(history) == 1
        assert history[0].trigger == RollbackTrigger.MANUAL


class TestAuditReport:
    """Tests for audit report generation."""

    def test_generate_audit_report(self, manager_with_versions):
        """Test generating audit report."""
        # Trigger a rollback for history
        manager_with_versions.trigger_rollback(
            reason="Test rollback",
            trigger=RollbackTrigger.DRIFT_CRITICAL,
        )

        report = manager_with_versions.generate_audit_report()

        assert "# Model Version Audit Report" in report
        assert "test_model" in report
        assert "Version History" in report
        assert "Rollback History" in report
        assert "FDA PCCP" in report

    def test_audit_report_with_policy(self, manager_with_versions):
        """Test audit report includes policy information."""
        policy = create_default_rollback_policy()
        manager_with_versions.set_rollback_policy(policy)

        report = manager_with_versions.generate_audit_report()

        assert "Current Policy" in report
        assert "medical_device_default" in report
        assert "Requires Approval" in report


class TestIntegration:
    """Integration tests for the rollback module."""

    def test_full_rollback_workflow(self, temp_storage, sample_model_files):
        """Test complete rollback workflow."""
        # Initialize manager
        mgr = ModelVersionManager(
            model_name="integration_test",
            storage_path=temp_storage,
        )

        # Register versions
        for version, path in sample_model_files.items():
            mgr.register_version(
                version=version,
                model_path=str(path),
                performance_metrics={"accuracy": 0.95},
            )

        # Set policy
        policy = RollbackPolicy(
            name="test_policy",
            enabled=True,
            triggers=[RollbackTrigger.DRIFT_CRITICAL],
            thresholds={"drift_critical": 0.2},
            require_approval=False,
        )
        mgr.set_rollback_policy(policy)

        # Activate latest version
        mgr.set_active("1.2.0")
        assert mgr.get_active_version().version == "1.2.0"

        # Check if rollback needed
        should_rollback, trigger, _ = mgr.check_should_rollback(drift_score=0.35)
        assert should_rollback is True

        # Execute rollback
        event = mgr.trigger_rollback(
            reason="Drift exceeded threshold",
            trigger=trigger,
        )
        assert event.status == RollbackStatus.COMPLETED
        assert mgr.get_active_version().version != "1.2.0"

        # Verify history
        history = mgr.get_rollback_history()
        assert len(history) == 1

        # Generate audit report
        report = mgr.generate_audit_report()
        assert "integration_test" in report
        assert "DRIFT_CRITICAL" in report or "drift_critical" in report

    def test_integrity_verification_on_activate(self, temp_storage, sample_model_file):
        """Test integrity verification during activation."""
        def custom_verifier(path: str) -> bool:
            # Custom verifier that checks file exists and has content
            p = Path(path)
            return p.exists() and p.stat().st_size > 0

        mgr = ModelVersionManager(
            model_name="integrity_test",
            storage_path=temp_storage,
            integrity_verifier=custom_verifier,
        )

        mgr.register_version("1.0.0", str(sample_model_file))

        # Should succeed with valid file
        mgr.set_active("1.0.0", verify_integrity=True)
        assert mgr.get_active_version().version == "1.0.0"
