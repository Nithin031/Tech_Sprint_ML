from enum import Enum
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# ================= ENUM =================

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


# ================= BASE =================

class _BaseClassifier:
    IMG_SIZE = 224

    def __init__(self, device="cpu", reliability_threshold=0.7):
        self.device = device
        self.reliability_threshold = reliability_threshold

        self.transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _postprocess(self, probs: np.ndarray, class_map):
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


# ================= EYE (3-CLASS) =================

class EyeFreshnessClassifier(_BaseClassifier):

    def __init__(self, model_path: str, device="cpu"):
        super().__init__(device)

        # ✅ MATCHES TRAINING: EfficientNet-B2 with ImageNet weights
        self.model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, 3
        )

        # ✅ Handle wrapped checkpoint format
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state_dict if wrapped in dictionary
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    def classify(self, image: Image.Image) -> ClassificationResult:
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return self._postprocess(
            probs,
            lambda i: FreshnessClass(i)
        )


# ================= GILL (2-CLASS) =================

class GillFreshnessClassifier(_BaseClassifier):

    def __init__(self, model_path: str, device="cpu"):
        super().__init__(device)

        # ✅ MATCHES TRAINING: EfficientNet-B2 with ImageNet weights
        self.model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, 2
        )

        # ✅ Handle wrapped checkpoint format
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state_dict if wrapped in dictionary
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    def classify(self, image: Image.Image) -> ClassificationResult:
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs2 = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # 🔥 LIFT TO 3-CLASS SPACE
        probs3 = np.array([
            probs2[0],   # NOT_FRESH
            probs2[1],   # FRESH
            0.0          # HIGHLY_FRESH (gill cannot infer)
        ], dtype=np.float32)

        return self._postprocess(
            probs3,
            lambda i: FreshnessClass(i)
        )