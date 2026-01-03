import numpy as np
from dataclasses import dataclass
from freshness_classifiers import ClassificationResult, FreshnessClass


@dataclass
class FusionResult:
    """Result from fusing eye and gill classifications."""
    final_class: FreshnessClass
    fused_probabilities: np.ndarray
    confidence: float
    eye_confidence: float
    gill_confidence: float
    eye_class: FreshnessClass
    gill_class: FreshnessClass


@dataclass
class DecisionOutcome:
    """Final decision outcome with business logic."""
    is_acceptable: bool
    freshness_grade: str  # "A", "B", "C", "REJECT"
    recommended_action: str
    confidence: float
    fusion_result: FusionResult


class FreshnessFusion:
    """
    Fuses eye and gill freshness classifications using weighted voting.
    
    Strategy:
    - Eye classifier: 3-class (NOT_FRESH, FRESH, HIGHLY_FRESH)
    - Gill classifier: 2-class (NOT_FRESH, FRESH) - already expanded to 3-class
    - Fusion weights based on individual classifier confidence
    """
    
    def __init__(self, eye_weight=0.55, gill_weight=0.45):
        """
        Args:
            eye_weight: Base weight for eye classifier (0-1)
            gill_weight: Base weight for gill classifier (0-1)
        """
        self.eye_weight = eye_weight
        self.gill_weight = gill_weight
        
        # Normalize weights
        total = eye_weight + gill_weight
        self.eye_weight /= total
        self.gill_weight /= total
    
    def fuse(
        self,
        eye_result: ClassificationResult,
        gill_result: ClassificationResult
    ) -> FusionResult:
        """
        Fuse eye and gill classification results.
        
        Args:
            eye_result: Eye freshness classification
            gill_result: Gill freshness classification (already 3-class)
        
        Returns:
            FusionResult with combined prediction
        """
        # ✅ FIXED: gill_result.probabilities is already 3-class
        # No need to expand - just use directly
        eye_probs = eye_result.probabilities
        gill_probs = gill_result.probabilities
        
        # Dynamic weighting based on confidence
        eye_conf = eye_result.confidence
        gill_conf = gill_result.confidence
        
        # Adjust weights by confidence (higher confidence = more influence)
        dynamic_eye_weight = self.eye_weight * eye_conf
        dynamic_gill_weight = self.gill_weight * gill_conf
        
        # Normalize dynamic weights
        total_weight = dynamic_eye_weight + dynamic_gill_weight
        dynamic_eye_weight /= total_weight
        dynamic_gill_weight /= total_weight
        
        # Weighted fusion
        fused_probs = (
            dynamic_eye_weight * eye_probs +
            dynamic_gill_weight * gill_probs
        )
        
        # Normalize to ensure valid probability distribution
        fused_probs = fused_probs / fused_probs.sum()
        
        # Determine final class
        final_idx = int(np.argmax(fused_probs))
        final_class = FreshnessClass(final_idx)
        
        # Calculate overall confidence
        entropy = -np.sum(fused_probs * np.log(fused_probs + 1e-12))
        confidence = 1.0 - entropy / np.log(len(fused_probs))
        
        return FusionResult(
            final_class=final_class,
            fused_probabilities=fused_probs,
            confidence=float(confidence),
            eye_confidence=float(eye_conf),
            gill_confidence=float(gill_conf),
            eye_class=eye_result.predicted_class,
            gill_class=gill_result.predicted_class
        )
    
    def make_decision(self, fusion_result: FusionResult) -> DecisionOutcome:
        """
        Convert fusion result into business decision.
        
        Args:
            fusion_result: Result from fuse()
        
        Returns:
            DecisionOutcome with grade and action
        """
        final_class = fusion_result.final_class
        confidence = fusion_result.confidence
        
        # Map freshness class to business decision
        if final_class == FreshnessClass.HIGHLY_FRESH:
            grade = "A"
            is_acceptable = True
            action = "Premium quality - Sell as fresh/sashimi grade"
        elif final_class == FreshnessClass.FRESH:
            if confidence >= 0.7:
                grade = "B"
                is_acceptable = True
                action = "Good quality - Sell as fresh"
            else:
                grade = "C"
                is_acceptable = True
                action = "Acceptable - Use for cooking/processing"
        else:  # NOT_FRESH
            grade = "REJECT"
            is_acceptable = False
            action = "Reject - Do not sell"
        
        return DecisionOutcome(
            is_acceptable=is_acceptable,
            freshness_grade=grade,
            recommended_action=action,
            confidence=confidence,
            fusion_result=fusion_result
        )