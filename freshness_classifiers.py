import numpy as np
import onnxruntime as ort
from PIL import Image
from enum import Enum
from dataclasses import dataclass

# ================= ENUM & DATA =================

class FreshnessClass(Enum):
    NOT_FRESH = 0
    FRESH = 1
    HIGHLY_FRESH = 2

@dataclass
class ClassificationResult:
    predicted_class: FreshnessClass
    probabilities: np.ndarray
    confidence: float
    entropy: float
    is_reliable: bool

# ================= BASE (ONNX OPTIMIZED) =================

class _BaseClassifier:
    IMG_SIZE = 224

    def __init__(self, model_path: str, reliability_threshold=0.7):
        # ✅ Load the ONNX Session
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.reliability_threshold = reliability_threshold

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Replaces torchvision.transforms for ONNX compatibility"""
        # 1. Resize
        img = image.resize((self.IMG_SIZE, self.IMG_SIZE), Image.BILINEAR)
        
        # 2. Convert to Array & Normalize (0-1)
        img_data = np.array(img).astype(np.float32) / 255.0
        
        # 3. Standard ImageNet Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        
        # 4. HWC to NCHW (1, 3, 224, 224)
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)
        return img_data

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def _postprocess(self, probs: np.ndarray, class_map):
        # Entropy & Confidence logic
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        confidence = 1.0 - entropy / np.log(len(probs))
        pred_idx = int(np.argmax(probs))

        return ClassificationResult(
            predicted_class=class_map(pred_idx),
            probabilities=probs,
            confidence=float(confidence),
            entropy=float(entropy),
            is_reliable=confidence >= self.reliability_threshold
        )

# ================= EYE (3-CLASS ONNX) =================

class EyeFreshnessClassifier(_BaseClassifier):
    def __init__(self, model_path: str, device="unused"):
        # We ignore device="cuda" here to force CPU efficiency on 8GB RAM
        super().__init__(model_path)

    def classify(self, image: Image.Image) -> ClassificationResult:
        # 1. Preprocess
        x = self._preprocess(image)

        # 2. ONNX Inference
        logits = self.session.run(None, {self.input_name: x})[0]
        
        # 3. Softmax & Postprocess
        probs = self._softmax(logits)[0]

        return self._postprocess(
            probs,
            lambda i: FreshnessClass(i)
        )

# ================= GILL (2-CLASS ONNX) =================

class GillFreshnessClassifier(_BaseClassifier):
    def __init__(self, model_path: str, device="unused"):
        super().__init__(model_path)

    def classify(self, image: Image.Image) -> ClassificationResult:
        # 1. Preprocess
        x = self._preprocess(image)

        # 2. ONNX Inference
        logits = self.session.run(None, {self.input_name: x})[0]
        
        # 3. Softmax
        probs2 = self._softmax(logits)[0]

        # 🔥 LIFT TO 3-CLASS SPACE
        # Keeps your business logic intact: Gill can't see "Highly Fresh"
        probs3 = np.array([
            probs2[0],   # NOT_FRESH
            probs2[1],   # FRESH
            0.0          # HIGHLY_FRESH
        ], dtype=np.float32)

        return self._postprocess(
            probs3,
            lambda i: FreshnessClass(i)
        )