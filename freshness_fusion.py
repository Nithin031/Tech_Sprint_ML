import numpy as np
from dataclasses import dataclass
from freshness_classifiers import ClassificationResult, FreshnessClass

# ================= DATA STRUCTURES =================

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

# ================= FUSION LOGIC =================

class FreshnessFusion:
    """
    Fuses eye and gill freshness classifications using dynamic weighted voting.
    Optimized for 8GB RAM environments by using purely NumPy-based operations.
    """
    
    def __init__(self, eye_weight=0.55, gill_weight=0.45):
        """
        Args:
            eye_weight: Base priority for eye (Eye is usually more reliable for sashimi grade)
            gill_weight: Base priority for gill (Gill is better for binary fresh/not-fresh)
        """
        self.eye_weight = eye_weight
        self.gill_weight = gill_weight
        
        # Initial weight normalization
        total = eye_weight + gill_weight
        self.eye_weight /= total
        self.gill_weight /= total
    
    def fuse(self, eye_result: ClassificationResult, gill_result: ClassificationResult) -> FusionResult:
        """
        Weighted probability fusion. 
        Higher individual confidence leads to higher influence on the final result.
        """
        # Extract 3-class probability arrays from your ONNX outputs
        eye_probs = eye_result.probabilities
        gill_probs = gill_result.probabilities
        
        # Dynamic Weighting Logic
        # Result = (Weight_E * Conf_E * Probs_E + Weight_G * Conf_G * Probs_G)
        eye_conf = eye_result.confidence
        gill_conf = gill_result.confidence
        
        dyn_e_weight = self.eye_weight * eye_conf
        dyn_g_weight = self.gill_weight * gill_conf
        
        total_dyn_weight = dyn_e_weight + dyn_g_weight + 1e-12 # Prevent div by zero
        
        # Final normalized weighted probabilities
        fused_probs = (dyn_e_weight * eye_probs + dyn_g_weight * gill_probs) / total_dyn_weight
        
        # Determine final classification index
        final_idx = int(np.argmax(fused_probs))
        final_class = FreshnessClass(final_idx)
        
        # Fused Confidence calculation (Inverse Entropy)
        # Higher entropy = lower confidence
        entropy = -np.sum(fused_probs * np.log(fused_probs + 1e-12))
        fused_confidence = 1.0 - entropy / np.log(len(fused_probs))
        
        return FusionResult(
            final_class=final_class,
            fused_probabilities=fused_probs,
            confidence=float(fused_confidence),
            eye_confidence=float(eye_conf),
            gill_confidence=float(gill_conf),
            eye_class=eye_result.predicted_class,
            gill_class=gill_result.predicted_class
        )
    
    def make_decision(self, fusion_result: FusionResult) -> DecisionOutcome:
        """
        Business Logic: Maps AI probabilities to commercial grades.
        """
        final_class = fusion_result.final_class
        conf = fusion_result.confidence
        
        # --- GRADE A: PREMIUM ---
        if final_class == FreshnessClass.HIGHLY_FRESH:
            grade = "A"
            is_acceptable = True
            action = "Premium Quality - Safe for raw consumption/sashimi."
            
        # --- GRADE B/C: MARKET QUALITY ---
        elif final_class == FreshnessClass.FRESH:
            if conf >= 0.75:
                grade = "B"
                is_acceptable = True
                action = "Good Quality - Sell as fresh market grade."
            else:
                grade = "C"
                is_acceptable = True
                action = "Acceptable - Recommended for cooking/processing."
                
        # --- REJECT: SPOILED ---
        else:
            grade = "REJECT"
            is_acceptable = False
            action = "SPOILED - DO NOT SELL. Potential health risk."
        
        return DecisionOutcome(
            is_acceptable=is_acceptable,
            freshness_grade=grade,
            recommended_action=action,
            confidence=conf,
            fusion_result=fusion_result
        )