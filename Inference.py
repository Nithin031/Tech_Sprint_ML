import argparse
import torch
from PIL import Image

from yolo_confidence_gate import YOLOConfidenceGate, JobOutcome
from part_cropper import YOLOPartCropper, CropStatus

# ✅ CORRECT IMPORTS
from freshness_classifiers import (
    EyeFreshnessClassifier,
    GillFreshnessClassifier,
    FreshnessClass
)

from freshness_fusion import FreshnessFusion, DecisionOutcome


# ================= CONFIG =================

YOLO_MODEL_PATH = r"D:\Hackathon\Tech Sprint\yolov.pt"
EYE_CLASSIFIER_PATH = r"D:\Hackathon\Tech Sprint\eye_effnet_best (1).pt"
GILL_CLASSIFIER_PATH = r"D:\Hackathon\Tech Sprint\gill_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ================= PIPELINE =================

def run_inference(eye_img_path, gill_img_path, verbose=False):

    print("\n=== FISH FRESHNESS INFERENCE ===\n")

    # ---------- LOAD IMAGES ----------
    try:
        eye_img = Image.open(eye_img_path).convert("RGB")
        gill_img = Image.open(gill_img_path).convert("RGB")
    except Exception as e:
        print("FAILED: Could not load input images")
        print(e)
        return

    # ---------- YOLO CONFIDENCE GATE ----------
    gate = YOLOConfidenceGate(
        model_path=YOLO_MODEL_PATH,
        device=DEVICE
    )

    if gate.run(eye_img, gill_img) == JobOutcome.FAILED:
        print("FAILED: Eye/Gill validation failed at YOLO gate")
        return

    # ---------- PART CROPPING ----------
    cropper = YOLOPartCropper(
        model_path=YOLO_MODEL_PATH,
        device=DEVICE
    )

    crop_status, eye_crop, gill_crop = cropper.crop_parts(
        eye_image=eye_img,
        gill_image=gill_img
    )

    if crop_status == CropStatus.FAILED:
        print("FAILED: Could not crop eye or gill")
        return

    # ---------- LOAD CLASSIFIERS (✅ FIXED) ----------
    eye_classifier = EyeFreshnessClassifier(
        model_path=EYE_CLASSIFIER_PATH,
        device=DEVICE
    )

    gill_classifier = GillFreshnessClassifier(
        model_path=GILL_CLASSIFIER_PATH,
        device=DEVICE
    )

    # ---------- CLASSIFY ----------
    eye_result = eye_classifier.classify(eye_crop)
    gill_result = gill_classifier.classify(gill_crop)

    # ---------- FUSION ----------
    fusion = FreshnessFusion()
    fusion_result = fusion.fuse(eye_result, gill_result)
    
    # ✅ FIXED: Get business decision from fusion result
    decision = fusion.make_decision(fusion_result)

    # ---------- OUTPUT ----------
    print("FINAL RESULT")
    print("------------")
    print(f"Freshness Class  : {fusion_result.final_class.name}")
    print(f"Freshness Grade  : {decision.freshness_grade}")
    print(f"Confidence       : {fusion_result.confidence * 100:.1f}%")
    print(f"Acceptable       : {'YES' if decision.is_acceptable else 'NO'}")
    print(f"Recommendation   : {decision.recommended_action}")

    # ---------- VERBOSE ----------
    if verbose:
        print("\nDETAILED BREAKDOWN")
        print("------------------")
        print(f"Eye Classification:")
        print(f"  • Class      : {eye_result.predicted_class.name}")
        print(f"  • Confidence : {eye_result.confidence * 100:.1f}%")
        print(f"  • Reliable   : {eye_result.is_reliable}")
        
        print(f"\nGill Classification:")
        print(f"  • Class      : {gill_result.predicted_class.name}")
        print(f"  • Confidence : {gill_result.confidence * 100:.1f}%")
        print(f"  • Reliable   : {gill_result.is_reliable}")
        
        print(f"\nFusion:")
        print(f"  • Eye weight  : {fusion.eye_weight:.2f}")
        print(f"  • Gill weight : {fusion.gill_weight:.2f}")

        print("\nProbabilities")
        print("-------------")
        print("Class          | Eye    | Gill   | Fused")
        print("---------------|--------|--------|--------")
        for i, cls in enumerate([FreshnessClass.NOT_FRESH, FreshnessClass.FRESH, FreshnessClass.HIGHLY_FRESH]):
            print(f"{cls.name:14s} | {eye_result.probabilities[i]:.3f}  | {gill_result.probabilities[i]:.3f}  | {fusion_result.fused_probabilities[i]:.3f}")

    print("\nInference completed.\n")


# ================= CLI =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye_image")
    parser.add_argument("gill_image")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_inference(
        args.eye_image,
        args.gill_image,
        verbose=args.verbose
    )