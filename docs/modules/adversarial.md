# Adversarial ML Module

Phase 4 of the MedTech AI Security platform. Tests robustness of medical AI models
against adversarial attacks.

## Overview

The Adversarial ML module evaluates and improves the robustness of machine learning
models used in medical devices, ensuring they resist adversarial manipulation.

## Attack Methods

| Method | Type | Description |
|--------|------|-------------|
| FGSM | White-box | Fast Gradient Sign Method - single-step gradient attack |
| PGD | White-box | Projected Gradient Descent - iterative attack |
| C&W | White-box | Carlini & Wagner - optimization-based minimal perturbation |
| DeepFool | White-box | Minimal perturbation to cross decision boundary |
| Square | Black-box | Query-efficient random search attack |

## Defense Methods

| Method | Type | Description |
|--------|------|-------------|
| JPEG Compression | Preprocessing | Remove high-frequency perturbations |
| Gaussian Blur | Preprocessing | Smooth adversarial noise |
| Feature Squeezing | Preprocessing | Reduce color depth |
| Bit Depth Reduction | Preprocessing | Quantize pixel values |
| Adversarial Training | Model-based | Train with adversarial examples |
| Randomized Smoothing | Certified | Provable robustness via noise |

## CLI Usage

### Evaluate Robustness

```bash
# Full robustness evaluation
medsec-adversarial evaluate --model models/classifier.keras

# Specific attack
medsec-adversarial evaluate --model model.keras --attack fgsm --epsilon 0.1
```

### Generate Adversarial Examples

```bash
# FGSM attack
medsec-adversarial attack --method fgsm --epsilon 0.03 --input images/

# PGD attack
medsec-adversarial attack --method pgd --epsilon 0.1 --steps 40
```

### Apply Defenses

```bash
# Test defense effectiveness
medsec-adversarial defend --method jpeg --quality 75 --input adversarial/
```

## Python API

```python
from medtech_ai_security.adversarial import (
    AdversarialAttacker,
    AdversarialDefender,
    RobustnessEvaluator
)

# Attack
attacker = AdversarialAttacker(model)
adv_images = attacker.fgsm_attack(images, labels, epsilon=0.03)

# Defend
defender = AdversarialDefender()
defended = defender.jpeg_compression(adv_images, quality=75)

# Evaluate
evaluator = RobustnessEvaluator(model)
report = evaluator.full_evaluation(test_images, test_labels)
print(f"Clean Accuracy: {report.clean_accuracy}")
print(f"Robust Accuracy: {report.robust_accuracy}")
```

## Clinical Impact Assessment

The module assesses clinical implications of adversarial attacks:

| Scenario | Impact Level | Example |
|----------|--------------|---------|
| False Negative | Critical | Missed tumor in radiology AI |
| False Positive | High | Unnecessary intervention |
| Degraded Confidence | Medium | Unreliable predictions |
| No Change | Low | Attack unsuccessful |

## Robustness Report

```python
report = evaluator.full_evaluation(images, labels)

# Access results
print(report.to_dict())
```

Output:

```json
{
  "clean_accuracy": 0.95,
  "attacks": {
    "fgsm_0.03": {"accuracy": 0.42, "success_rate": 0.58},
    "pgd_0.1": {"accuracy": 0.15, "success_rate": 0.85},
    "cw_l2": {"accuracy": 0.08, "success_rate": 0.92}
  },
  "defenses": {
    "jpeg_75": {"recovered_accuracy": 0.78},
    "gaussian_blur": {"recovered_accuracy": 0.65}
  },
  "clinical_impact": "high",
  "recommendations": [
    "Implement adversarial training",
    "Add input validation layer",
    "Consider ensemble methods"
  ]
}
```

## Best Practices for Medical AI

1. **Test Before Deployment**: Evaluate all models with this module
2. **Defense in Depth**: Combine multiple defense methods
3. **Monitor in Production**: Track prediction confidence over time
4. **Document Robustness**: Include in FDA 510(k) submissions
5. **Regular Re-evaluation**: Test after model updates
