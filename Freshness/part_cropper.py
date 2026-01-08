import numpy as np
import onnxruntime as ort
from PIL import Image
from enum import Enum

class CropStatus(Enum):
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"  # ✅ Added this so Inference.py doesn't crash
    FAILED = "FAILED"

class YOLOPartCropper:
    """
    Crops EYE and GILL regions using ONE YOLOv8 ONNX model.
    FALLBACK MODE ENABLED: If detection fails, takes center crop.
    """

    EYE_CLASSES  = {"eye", "mata"}
    GILL_CLASSES = {"gill", "insang"}

    def __init__(self, model_path: str, min_conf: float = 0.25):
        # Load ONNX Session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.min_conf = min_conf
        
        # Update this mapping if your YOLO classes are different!
        self.names = {0: "eye", 1: "gill"} 

    def _preprocess(self, img: Image.Image):
        """Resize and Normalize for YOLO"""
        shape = self.session.get_inputs()[0].shape
        self.imgsz = (shape[2], shape[3])
        
        img_resized = img.resize(self.imgsz)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)[np.newaxis, ...]
        return img_array

    def _get_fallback_crop(self, image: Image.Image):
        """Returns the center 50% of the image if detection fails."""
        w, h = image.size
        left = w * 0.25
        top = h * 0.25
        right = w * 0.75
        bottom = h * 0.75
        return image.crop((left, top, right, bottom))

    def _crop_single(self, image: Image.Image, target_classes: set):
        img_data = self._preprocess(image)
        
        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: img_data})[0]
        output = np.squeeze(outputs).T 
        
        best_box = None
        best_conf = 0.0

        for row in output:
            scores = row[4:] 
            cls_id = np.argmax(scores)
            conf = scores[cls_id]
            cls_name = self.names.get(int(cls_id), "unknown")

            if cls_name in target_classes and conf > best_conf:
                best_conf = conf
                best_box = row[:4]

        # --- FALLBACK LOGIC ---
        if best_box is None or best_conf < self.min_conf:
            # Return center crop and 'False' (meaning not a real detection)
            return self._get_fallback_crop(image), False 

        # Scale boxes back to original image
        orig_w, orig_h = image.size
        cx, cy, w, h = best_box
        scale_x = orig_w / self.imgsz[0]
        scale_y = orig_h / self.imgsz[1]

        x1 = int((cx - w/2) * scale_x)
        y1 = int((cy - h/2) * scale_y)
        x2 = int((cx + w/2) * scale_x)
        y2 = int((cy + h/2) * scale_y)

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(orig_w, x2), min(orig_h, y2)

        return image.crop((x1, y1, x2, y2)), True

    def crop_parts(self, eye_image: Image.Image, gill_image: Image.Image):
        eye_crop, eye_ok = self._crop_single(eye_image, self.EYE_CLASSES)
        gill_crop, gill_ok = self._crop_single(gill_image, self.GILL_CLASSES)

        # If either part used fallback, return WARNING so we know
        if not eye_ok or not gill_ok:
            return CropStatus.WARNING, eye_crop, gill_crop
        
        return CropStatus.SUCCESS, eye_crop, gill_crop