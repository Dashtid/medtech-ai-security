"""
Adversarial ML Module for Medical AI Security

Provides adversarial attack and defense capabilities for testing
robustness of medical imaging AI models.

Attack Methods:
- FGSM (Fast Gradient Sign Method): Fast, single-step attack
- PGD (Projected Gradient Descent): Iterative, stronger attack
- C&W (Carlini & Wagner): Optimization-based, most powerful

Defense Methods:
- Adversarial Training: Train with adversarial examples
- Input Preprocessing: JPEG compression, bit-depth reduction
- Feature Squeezing: Gaussian blur, spatial smoothing

Usage:
    from medtech_ai_security.adversarial import AdversarialAttacker, RobustnessEvaluator

    attacker = AdversarialAttacker(model)
    adv_images = attacker.fgsm(images, labels, epsilon=0.01)

    evaluator = RobustnessEvaluator(model)
    report = evaluator.evaluate(test_images, test_labels)
"""

from medtech_ai_security.adversarial.attacks import (
    AdversarialAttacker,
    AttackResult,
    AttackType,
)
from medtech_ai_security.adversarial.defenses import (
    AdversarialDefender,
    DefenseResult,
    DefenseType,
)
from medtech_ai_security.adversarial.evaluator import (
    RobustnessEvaluator,
    RobustnessReport,
)

__all__ = [
    "AdversarialAttacker",
    "AttackResult",
    "AttackType",
    "AdversarialDefender",
    "DefenseResult",
    "DefenseType",
    "RobustnessEvaluator",
    "RobustnessReport",
]
