import argparse
import sys
import os

# Custom modules
from Input import ImageInput, ImageType, InputJobProcessor
from part_cropper import YOLOPartCropper, CropStatus
from freshness_classifiers import EyeFreshnessClassifier, GillFreshnessClassifier
from freshness_fusion import FreshnessFusion

def run_inference(eye_path, gill_path):
    print("\n=== FISH FRESHNESS INFERENCE PIPELINE ===\n")

    # ---------------------------------------------------------
    # STEP A: INPUT VALIDATION
    # ---------------------------------------------------------
    print("[1/4] Step A: Validating Input Images...")
    try:
        with open(eye_path, "rb") as f: e_data = f.read()
        with open(gill_path, "rb") as f: g_data = f.read()

        e_in = ImageInput(ImageType.EYE, e_data, eye_path, 0.0)
        g_in = ImageInput(ImageType.GILL, g_data, gill_path, 0.0)

        processor = InputJobProcessor()
        results = processor.process(e_in, g_in)

        # Print Warnings (like low resolution) but keep going
        for itype in [ImageType.EYE, ImageType.GILL]:
            for w in results[itype].warnings:
                print(f"⚠️  {itype.value} WARNING: {w}")

        eye_img = results[ImageType.EYE].corrected_image
        gill_img = results[ImageType.GILL].corrected_image

    except ValueError as e:
        print(f"❌ STEP A FAILED: {e}")
        return

    # ---------------------------------------------------------
    # STEP B: DETECTION & DEBUG CROP
    # ---------------------------------------------------------
    print("[2/4] Step B: Detecting Regions (YOLO)...")
    
    # Use your ONNX model with a reasonable threshold
    cropper = YOLOPartCropper(model_path="yolov.onnx", min_conf=0.25)
    status, eye_crop, gill_crop = cropper.crop_parts(eye_img, gill_img)

    # Handle Warning (Fallback) vs Failed
    if status == CropStatus.WARNING:
        print("⚠️  STEP B WARNING: Detection low confidence. Using Fallback Center Crops.")
    elif status == CropStatus.FAILED:
        print("❌ STEP B FAILED: Could not crop images.")
        return

    # ✅ DEBUG: Save the crops to disk so you can see them!
    print("      -> Saving debug crops...")
    try:
        eye_crop.save("debug_eye_crop.jpg")
        gill_crop.save("debug_gill_crop.jpg")
        print("      ✅ Saved 'debug_eye_crop.jpg' & 'debug_gill_crop.jpg'")
    except Exception as e:
        print(f"      ⚠️ Could not save debug images: {e}")

    # ---------------------------------------------------------
    # STAGE 3: CLASSIFY
    # ---------------------------------------------------------
    print("[3/4] Classifying Freshness...")
    
    # Load ONNX classifiers
    eye_clf = EyeFreshnessClassifier("eye_freshness.onnx")
    gill_clf = GillFreshnessClassifier("gill_freshness.onnx")

    eye_res = eye_clf.classify(eye_crop)
    gill_res = gill_clf.classify(gill_crop)

    # ---------------------------------------------------------
    # STAGE 4: FUSION & DECISION
    # ---------------------------------------------------------
    print("[4/4] Fusing Results...")
    fusion = FreshnessFusion()
    f_res = fusion.fuse(eye_res, gill_res)
    decision = fusion.make_decision(f_res)

    print("\n" + "="*30)
    print(f"FINAL GRADE    : {decision.freshness_grade}")
    print(f"CONFIDENCE     : {f_res.confidence * 100:.1f}%")
    print(f"ACTION         : {decision.recommended_action}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eye")
    parser.add_argument("gill")
    args = parser.parse_args()
    
    run_inference(args.eye, args.gill)