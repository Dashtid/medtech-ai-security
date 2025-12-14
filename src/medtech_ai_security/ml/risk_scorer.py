"""
ML-Powered Vulnerability Risk Scorer for Medical Devices

This module provides machine learning-based risk scoring for medical device
vulnerabilities. It uses a lightweight implementation with numpy/scipy
for maximum compatibility.

Features used for prediction:
- CVSS v3 score and vector components
- CWE categories (mapped to security domains)
- Device type (imaging, monitoring, infusion, etc.)
- Clinical impact assessment
- Exploitability rating
- Vulnerability age and status

Usage:
    from medtech_ai_security.ml import VulnerabilityRiskScorer

    scorer = VulnerabilityRiskScorer()
    scorer.load_training_data("data/threat_intel/cves/medical_devices_enriched.json")
    scorer.train()

    # Score a new vulnerability
    prediction = scorer.predict({
        "cvss_v3_score": 9.8,
        "cvss_v3_vector": "CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "cwe_ids": ["CWE-798"],
        "device_type": "imaging",
        "clinical_impact": "HIGH",
        "exploitability": "EASY"
    })
    print(f"Risk Score: {prediction.risk_score:.2f}")
    print(f"Priority: {prediction.priority}")
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CWE to security domain mapping
CWE_DOMAINS = {
    # Memory Safety
    "CWE-119": "memory_safety",
    "CWE-120": "memory_safety",
    "CWE-121": "memory_safety",
    "CWE-122": "memory_safety",
    "CWE-125": "memory_safety",
    "CWE-787": "memory_safety",
    "CWE-788": "memory_safety",
    "CWE-189": "memory_safety",
    "CWE-190": "memory_safety",
    "CWE-191": "memory_safety",
    # Authentication
    "CWE-287": "authentication",
    "CWE-798": "authentication",
    "CWE-259": "authentication",
    "CWE-288": "authentication",
    "CWE-306": "authentication",
    "CWE-307": "authentication",
    # Injection
    "CWE-78": "injection",
    "CWE-79": "injection",
    "CWE-89": "injection",
    "CWE-94": "injection",
    "CWE-77": "injection",
    # Cryptography
    "CWE-295": "cryptography",
    "CWE-310": "cryptography",
    "CWE-311": "cryptography",
    "CWE-312": "cryptography",
    "CWE-327": "cryptography",
    "CWE-328": "cryptography",
    # Access Control
    "CWE-22": "access_control",
    "CWE-200": "access_control",
    "CWE-264": "access_control",
    "CWE-269": "access_control",
    "CWE-284": "access_control",
    "CWE-285": "access_control",
    # Input Validation
    "CWE-20": "input_validation",
    "CWE-129": "input_validation",
    "CWE-131": "input_validation",
    # Resource Management
    "CWE-400": "resource_management",
    "CWE-401": "resource_management",
    "CWE-404": "resource_management",
    "CWE-416": "resource_management",
    "CWE-772": "resource_management",
}

# CVSS v3 vector component mappings
CVSS_AV_VALUES = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}  # Attack Vector
CVSS_AC_VALUES = {"L": 0.77, "H": 0.44}  # Attack Complexity
CVSS_PR_VALUES = {"N": 0.85, "L": 0.62, "H": 0.27}  # Privileges Required
CVSS_UI_VALUES = {"N": 0.85, "R": 0.62}  # User Interaction
CVSS_S_VALUES = {"U": 0, "C": 1}  # Scope
CVSS_CIA_VALUES = {"N": 0, "L": 0.22, "H": 0.56}  # C/I/A Impact

# Priority class weights
PRIORITY_WEIGHTS = {"CRITICAL": 100, "HIGH": 75, "MEDIUM": 50, "LOW": 25}
PRIORITY_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class RiskPrediction:
    """Container for risk prediction results."""

    cve_id: str
    risk_score: float  # 0-100 normalized score
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    confidence: float  # Model confidence 0-1
    contributing_factors: dict = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "cve_id": self.cve_id,
            "risk_score": round(self.risk_score, 2),
            "priority": self.priority,
            "confidence": round(self.confidence, 3),
            "contributing_factors": self.contributing_factors,
            "recommendation": self.recommendation,
        }


class NaiveBayesClassifier:
    """
    Simple Gaussian Naive Bayes classifier implemented with numpy.

    This provides basic ML classification without sklearn dependency.
    """

    def __init__(self) -> None:
        self.class_priors: dict[Any, float] = {}
        self.class_means: dict[Any, np.ndarray] = {}
        self.class_vars: dict[Any, np.ndarray] = {}
        self.classes: np.ndarray = np.array([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayesClassifier":
        """
        Fit the classifier.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels

        Returns:
            Self for chaining.
        """
        self.classes = np.unique(y)
        n_samples = len(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples
            self.class_means[c] = np.mean(X_c, axis=0)
            # Add small epsilon for numerical stability
            self.class_vars[c] = np.var(X_c, axis=0) + 1e-9

        return self

    def _gaussian_likelihood(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Calculate Gaussian probability density."""
        eps = 1e-9
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        result: np.ndarray = coeff * exponent
        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes):
            prior = np.log(self.class_priors[c])
            # Sum of log likelihoods
            likelihood = np.sum(
                np.log(
                    self._gaussian_likelihood(X, self.class_means[c], self.class_vars[c]) + 1e-300
                ),
                axis=1,
            )
            probs[:, idx] = prior + likelihood

        # Convert log probabilities to probabilities using softmax
        probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

        return np.asarray(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return np.asarray(self.classes[np.argmax(probs, axis=1)])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))


class KNNClassifier:
    """
    K-Nearest Neighbors classifier implemented with numpy/scipy.

    Uses weighted voting based on distance for better predictions.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        Fit the classifier by storing training data.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Self for chaining.
        """
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using distance-weighted voting.

        Args:
            X: Feature matrix

        Returns:
            Probability matrix (n_samples, n_classes)
        """
        assert (
            self.X_train is not None and self.y_train is not None and self.classes is not None
        ), "Model must be fitted before prediction"

        # Calculate distances to all training points
        distances = cdist(X, self.X_train, metric="euclidean")

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            # Get k nearest neighbors
            k_indices = np.argsort(distances[i])[: self.k]
            k_distances = distances[i, k_indices]
            k_labels = self.y_train[k_indices]

            # Weight by inverse distance (add epsilon to avoid division by zero)
            weights = 1.0 / (k_distances + 1e-5)
            weights /= weights.sum()

            # Weighted vote for each class
            for idx, c in enumerate(self.classes):
                mask = k_labels == c
                probs[i, idx] = weights[mask].sum()

        return np.asarray(probs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        assert self.classes is not None, "Model must be fitted before prediction"
        probs = self.predict_proba(X)
        return np.asarray(self.classes[np.argmax(probs, axis=1)])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return float(np.mean(predictions == y))


class VulnerabilityRiskScorer:
    """
    ML-based vulnerability risk scorer for medical devices.

    Uses ensemble of Naive Bayes and KNN classifiers to predict
    vulnerability priority and compute risk scores.
    """

    def __init__(self, model_path: Path | None = None):
        """
        Initialize the risk scorer.

        Args:
            model_path: Optional path to load a pre-trained model.
        """
        self.model_nb = NaiveBayesClassifier()
        self.model_knn = KNNClassifier(k=5)
        self.feature_names: list[str] = []
        self.feature_means: np.ndarray | None = None
        self.feature_stds: np.ndarray | None = None
        self.is_trained = False
        self.training_data: dict[str, Any] = {}
        self.feature_importance: dict[str, float] = {}

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def _parse_cvss_vector(self, vector: str | None) -> dict:
        """
        Parse CVSS v3 vector string into numeric components.

        Args:
            vector: CVSS v3.x vector string (e.g., "CVSS:3.0/AV:N/AC:L/...")

        Returns:
            Dictionary of numeric component values.
        """
        defaults = {
            "av": 0.5,
            "ac": 0.5,
            "pr": 0.5,
            "ui": 0.5,
            "scope": 0,
            "c": 0.3,
            "i": 0.3,
            "a": 0.3,
        }

        if not vector:
            return defaults

        try:
            parts = vector.upper().split("/")
            result = defaults.copy()

            for part in parts:
                if part.startswith("AV:"):
                    result["av"] = CVSS_AV_VALUES.get(part[3:], 0.5)
                elif part.startswith("AC:"):
                    result["ac"] = CVSS_AC_VALUES.get(part[3:], 0.5)
                elif part.startswith("PR:"):
                    result["pr"] = CVSS_PR_VALUES.get(part[3:], 0.5)
                elif part.startswith("UI:"):
                    result["ui"] = CVSS_UI_VALUES.get(part[3:], 0.5)
                elif part.startswith("S:"):
                    result["scope"] = CVSS_S_VALUES.get(part[2:], 0)
                elif part.startswith("C:"):
                    result["c"] = CVSS_CIA_VALUES.get(part[2:], 0.3)
                elif part.startswith("I:"):
                    result["i"] = CVSS_CIA_VALUES.get(part[2:], 0.3)
                elif part.startswith("A:") and len(part) == 3:
                    result["a"] = CVSS_CIA_VALUES.get(part[2:], 0.3)

            return result
        except Exception as e:
            logger.warning(f"Failed to parse CVSS vector '{vector}': {e}")
            return defaults

    def _map_cwe_to_domain(self, cwe_ids: list[str]) -> str:
        """Map CWE IDs to security domain."""
        if not cwe_ids:
            return "other"

        for cwe_id in cwe_ids:
            cwe_normalized = cwe_id.upper().replace("CWE-", "CWE-")
            if cwe_normalized in CWE_DOMAINS:
                return CWE_DOMAINS[cwe_normalized]

        return "other"

    def _encode_device_type(self, device_type: str | None) -> dict:
        """One-hot encode device type."""
        device_types = [
            "imaging",
            "monitoring",
            "infusion",
            "implantable",
            "diagnostic",
            "therapeutic",
            "network",
            "other",
        ]
        result = {f"device_{dt}": 0 for dt in device_types}

        if device_type:
            dt_lower = device_type.lower()
            if dt_lower in device_types:
                result[f"device_{dt_lower}"] = 1
            else:
                result["device_other"] = 1
        else:
            result["device_other"] = 1

        return result

    def _encode_clinical_impact(self, impact: str | None) -> float:
        """Encode clinical impact to numeric value."""
        impact_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3, None: 0.5}
        return impact_map.get(impact.upper() if impact else None, 0.5)

    def _encode_exploitability(self, exploitability: str | None) -> float:
        """Encode exploitability to numeric value."""
        exploit_map = {"EASY": 1.0, "MODERATE": 0.6, "HARD": 0.3, None: 0.5}
        return exploit_map.get(exploitability.upper() if exploitability else None, 0.5)

    def _encode_cwe_domain(self, domain: str) -> dict:
        """One-hot encode CWE domain."""
        domains = [
            "memory_safety",
            "authentication",
            "injection",
            "cryptography",
            "access_control",
            "input_validation",
            "resource_management",
            "other",
        ]
        result = {f"cwe_{d}": 0 for d in domains}
        result[f"cwe_{domain}"] = 1
        return result

    def _calculate_vulnerability_age(self, published_date: str | None) -> float:
        """Calculate vulnerability age in years."""
        if not published_date:
            return 5.0

        try:
            pub_date = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            age_days = (datetime.now(pub_date.tzinfo) - pub_date).days
            return min(age_days / 365.0, 10.0)
        except Exception:
            return 5.0

    def _extract_features(self, cve: dict) -> dict:
        """Extract all features from a CVE entry."""
        features = {}

        # Base CVSS score
        features["cvss_score"] = cve.get("cvss_v3_score") or 5.0

        # CVSS vector components
        cvss_components = self._parse_cvss_vector(cve.get("cvss_v3_vector"))
        features.update(cvss_components)

        # CWE domain
        cwe_ids = cve.get("cwe_ids", [])
        cwe_domain = self._map_cwe_to_domain(cwe_ids)
        features.update(self._encode_cwe_domain(cwe_domain))

        # Device type
        features.update(self._encode_device_type(cve.get("device_type")))

        # Clinical impact and exploitability
        features["clinical_impact"] = self._encode_clinical_impact(cve.get("clinical_impact"))
        features["exploitability"] = self._encode_exploitability(cve.get("exploitability"))

        # Vulnerability age
        features["vuln_age"] = self._calculate_vulnerability_age(cve.get("published_date"))

        # Reference count
        refs = cve.get("references", [])
        features["reference_count"] = min(len(refs), 20) / 20.0

        # Has exploit reference
        exploit_keywords = ["exploit", "poc", "metasploit", "exploit-db"]
        features["has_exploit"] = int(any(kw in str(refs).lower() for kw in exploit_keywords))

        return features

    def _compute_target(self, cve: dict) -> str:
        """Compute target priority label from CVE data."""
        cvss_score = cve.get("cvss_v3_score") or 0
        clinical = (cve.get("clinical_impact") or "").upper()
        exploit = (cve.get("exploitability") or "").upper()

        # Composite score calculation
        score = cvss_score * 10

        if clinical == "HIGH":
            score += 15
        elif clinical == "MEDIUM":
            score += 8

        if exploit == "EASY":
            score += 10
        elif exploit == "MODERATE":
            score += 5

        device = (cve.get("device_type") or "").lower()
        if device in ["monitoring", "infusion", "implantable"]:
            score += 10
        elif device == "imaging":
            score += 5

        if score >= 90:
            return "CRITICAL"
        elif score >= 70:
            return "HIGH"
        elif score >= 50:
            return "MEDIUM"
        else:
            return "LOW"

    def load_training_data(self, data_path: Path | str) -> int:
        """
        Load training data from enriched CVE JSON file.

        Args:
            data_path: Path to enriched CVE JSON file.

        Returns:
            Number of samples loaded.
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        cves = data.get("cves", [])
        if not cves:
            raise ValueError("No CVE data found in file")

        features_list = []
        targets = []
        cve_ids = []

        for cve in cves:
            features = self._extract_features(cve)
            features_list.append(features)
            targets.append(self._compute_target(cve))
            cve_ids.append(cve.get("cve_id", ""))

        # Convert to numpy arrays
        self.feature_names = sorted(features_list[0].keys())
        X = np.array([[f[name] for name in self.feature_names] for f in features_list])

        # Encode targets to indices
        label_map = {label: idx for idx, label in enumerate(PRIORITY_ORDER)}
        y = np.array([label_map[t] for t in targets])

        self.training_data = {
            "X": X,
            "y": y,
            "cve_ids": cve_ids,
            "targets": targets,
        }

        logger.info(f"Loaded {len(cves)} CVE samples for training")
        return len(cves)

    def train(self, test_size: float = 0.2) -> dict:
        """
        Train the risk scoring models.

        Args:
            test_size: Fraction of data to use for testing.

        Returns:
            Dictionary of training metrics.
        """
        if not self.training_data:
            raise ValueError("No training data loaded. Call load_training_data first.")

        X = self.training_data["X"]
        y = self.training_data["y"]

        # Normalize features
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-9
        X_scaled = (X - self.feature_means) / self.feature_stds

        # Train/test split
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        indices = np.random.RandomState(42).permutation(n_samples)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Train models
        logger.info("Training Naive Bayes classifier...")
        self.model_nb.fit(X_train, y_train)
        nb_score = self.model_nb.score(X_test, y_test)

        logger.info("Training KNN classifier...")
        self.model_knn.fit(X_train, y_train)
        knn_score = self.model_knn.score(X_test, y_test)

        # Calculate feature importance via permutation
        self._calculate_feature_importance(X_test, y_test)

        self.is_trained = True

        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = {PRIORITY_ORDER[int(u)]: int(c) for u, c in zip(unique, counts, strict=False)}

        metrics = {
            "samples_total": len(X),
            "samples_train": len(X_train),
            "samples_test": len(X_test),
            "nb_test_accuracy": nb_score,
            "knn_test_accuracy": knn_score,
            "feature_importance": self.feature_importance,
            "class_distribution": class_dist,
        }

        logger.info(
            f"Training complete. NB accuracy: {nb_score:.3f}, KNN accuracy: {knn_score:.3f}"
        )

        importance_sorted = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        logger.info("Top 10 important features:")
        for feat, imp in importance_sorted:
            logger.info(f"  {feat}: {imp:.4f}")

        return metrics

    def _calculate_feature_importance(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Calculate feature importance via permutation importance."""
        baseline_score = self.model_knn.score(X_test, y_test)

        for idx, feat_name in enumerate(self.feature_names):
            X_permuted = X_test.copy()
            np.random.shuffle(X_permuted[:, idx])
            permuted_score = self.model_knn.score(X_permuted, y_test)
            self.feature_importance[feat_name] = max(0, baseline_score - permuted_score)

        # Normalize
        total = sum(self.feature_importance.values()) + 1e-9
        self.feature_importance = {k: v / total for k, v in self.feature_importance.items()}

    def predict(self, cve: dict) -> RiskPrediction:
        """
        Predict risk score for a single vulnerability.

        Args:
            cve: CVE dictionary with vulnerability data.

        Returns:
            RiskPrediction with score and priority.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        features = self._extract_features(cve)
        cve_id = cve.get("cve_id", "UNKNOWN")

        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X_scaled = (X - self.feature_means) / self.feature_stds

        # Ensemble prediction
        proba_nb = self.model_nb.predict_proba(X_scaled)[0]
        proba_knn = self.model_knn.predict_proba(X_scaled)[0]
        proba_ensemble = (proba_nb + proba_knn) / 2

        pred_idx = proba_ensemble.argmax()
        priority = PRIORITY_ORDER[pred_idx]
        confidence = proba_ensemble[pred_idx]

        # Calculate risk score
        risk_score = sum(
            proba_ensemble[i] * PRIORITY_WEIGHTS[PRIORITY_ORDER[i]]
            for i in range(len(proba_ensemble))
        )

        # Contributing factors
        feature_values = dict(zip(self.feature_names, X[0], strict=False))
        contributions = {
            f: self.feature_importance.get(f, 0) * abs(feature_values[f])
            for f in self.feature_names
        }
        top_factors = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5])

        recommendation = self._generate_recommendation(priority, features, top_factors)

        return RiskPrediction(
            cve_id=cve_id,
            risk_score=risk_score,
            priority=priority,
            confidence=confidence,
            contributing_factors=top_factors,
            recommendation=recommendation,
        )

    def _generate_recommendation(self, priority: str, features: dict, top_factors: dict) -> str:
        """Generate remediation recommendation based on prediction."""
        recommendations = []

        if priority == "CRITICAL":
            recommendations.append(
                "IMMEDIATE ACTION REQUIRED: This vulnerability poses critical risk."
            )
        elif priority == "HIGH":
            recommendations.append("HIGH PRIORITY: Address this vulnerability within 7 days.")
        elif priority == "MEDIUM":
            recommendations.append("MEDIUM PRIORITY: Schedule remediation within 30 days.")
        else:
            recommendations.append("LOW PRIORITY: Include in regular maintenance cycle.")

        if features.get("has_exploit"):
            recommendations.append("Active exploit exists - isolate affected systems immediately.")

        if features.get("clinical_impact", 0) >= 0.8:
            recommendations.append("High clinical impact - prioritize patient safety assessment.")

        for dt in ["monitoring", "infusion", "implantable"]:
            if features.get(f"device_{dt}", 0):
                recommendations.append(
                    f"Critical device type ({dt}) - follow FDA premarket notification guidelines."
                )
                break

        if features.get("cwe_authentication", 0):
            recommendations.append("Authentication weakness - enforce strong credentials and MFA.")
        elif features.get("cwe_memory_safety", 0):
            recommendations.append(
                "Memory safety issue - apply vendor patches or implement ASLR/DEP."
            )
        elif features.get("cwe_cryptography", 0):
            recommendations.append(
                "Cryptographic weakness - update to current encryption standards."
            )

        return " ".join(recommendations)

    def predict_batch(self, cves: list[dict]) -> list[RiskPrediction]:
        """Predict risk scores for multiple vulnerabilities."""
        return [self.predict(cve) for cve in cves]

    def save_model(self, path: Path | str) -> None:
        """Save trained model to disk."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Cannot save.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model_nb": self.model_nb,
            "model_knn": self.model_knn,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "feature_importance": self.feature_importance,
        }

        joblib.dump(model_data, path / "risk_model.joblib")

        logger.info(f"Model saved to {path}")

    def load_model(self, path: Path | str) -> None:
        """Load trained model from disk."""
        path = Path(path)

        model_data = joblib.load(path / "risk_model.joblib")

        self.model_nb = model_data["model_nb"]
        self.model_knn = model_data["model_knn"]
        self.feature_names = model_data["feature_names"]
        self.feature_means = model_data["feature_means"]
        self.feature_stds = model_data["feature_stds"]
        self.feature_importance = model_data["feature_importance"]
        self.is_trained = True

        logger.info(f"Model loaded from {path}")


def main() -> None:
    """CLI entry point for training and testing the risk scorer."""
    import argparse
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser(
        description="ML-powered vulnerability risk scorer for medical devices"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to enriched CVE JSON file",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default=None,
        help="Path to save trained model",
    )
    parser.add_argument(
        "--load-model",
        type=str,
        default=None,
        help="Path to load pre-trained model",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run predictions on all CVEs and output results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for predictions JSON",
    )

    args = parser.parse_args()

    scorer = VulnerabilityRiskScorer()

    if args.load_model:
        scorer.load_model(Path(args.load_model))
    else:
        scorer.load_training_data(Path(args.data))
        metrics = scorer.train()

        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Total samples: {metrics['samples_total']}")
        print(f"Train samples: {metrics['samples_train']}")
        print(f"Test samples: {metrics['samples_test']}")
        print(f"\nNaive Bayes Test Accuracy: {metrics['nb_test_accuracy']:.3f}")
        print(f"KNN Test Accuracy: {metrics['knn_test_accuracy']:.3f}")
        print(f"\nClass Distribution: {metrics['class_distribution']}")

        if args.save_model:
            scorer.save_model(Path(args.save_model))

    if args.predict:
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f)

        predictions = scorer.predict_batch(data.get("cves", []))
        predictions.sort(key=lambda x: x.risk_score, reverse=True)

        print("\n" + "=" * 60)
        print("TOP 10 HIGHEST RISK VULNERABILITIES")
        print("=" * 60)

        for pred in predictions[:10]:
            print(f"\n{pred.cve_id}")
            print(f"  Risk Score: {pred.risk_score:.1f}")
            print(f"  Priority: {pred.priority}")
            print(f"  Confidence: {pred.confidence:.2%}")
            print(f"  Recommendation: {pred.recommendation[:100]}...")

        if args.output:
            output_data = {
                "predictions": [p.to_dict() for p in predictions],
                "summary": {
                    "total": len(predictions),
                    "critical": sum(1 for p in predictions if p.priority == "CRITICAL"),
                    "high": sum(1 for p in predictions if p.priority == "HIGH"),
                    "medium": sum(1 for p in predictions if p.priority == "MEDIUM"),
                    "low": sum(1 for p in predictions if p.priority == "LOW"),
                },
            }

            # Handle OneDrive file write issue by using temp file + powershell copy
            temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            json.dump(output_data, temp_file, indent=2)
            temp_file.close()

            # Try direct write first, fall back to powershell
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2)
            except OSError:
                subprocess.run(
                    [
                        "powershell.exe",
                        "-Command",
                        f'Copy-Item -Path "{temp_file.name}" -Destination "{args.output}" -Force',
                    ],
                    check=True,
                )

            print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
