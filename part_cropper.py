from enum import Enum
from ultralytics import YOLO
from PIL import Image


# ================= ENUM =================

class CropStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


# ================= PART CROPPER =================

class YOLOPartCropper:
    """
    Crops EYE and GILL regions using ONE YOLO model (best.pt).

    Supports dataset-specific class names:
    - eye   → "eye", "mata"
    - gill  → "gill", "insang"
    """

    # ---- CLASS NAME NORMALIZATION ----
    EYE_CLASSES  = {"eye", "mata"}
    GILL_CLASSES = {"gill", "insang"}

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        min_conf: float = 0.40
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.min_conf = min_conf

    # ---------------- PUBLIC API ----------------

    def crop_parts(self, eye_image: Image.Image, gill_image: Image.Image):
        """
        Returns:
            (CropStatus, eye_crop, gill_crop)
        """

        eye_crop = self._crop_single(
            image=eye_image,
            target_classes=self.EYE_CLASSES
        )
        if eye_crop is None:
            return CropStatus.FAILED, None, None

        gill_crop = self._crop_single(
            image=gill_image,
            target_classes=self.GILL_CLASSES
        )
        if gill_crop is None:
            return CropStatus.FAILED, None, None

        return CropStatus.SUCCESS, eye_crop, gill_crop

    # ---------------- INTERNAL ----------------

    def _crop_single(self, image: Image.Image, target_classes: set):
        """
        Crops the highest-confidence box from target_classes.
        Returns PIL.Image or None.
        """

        result = self.model(image, verbose=False)[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None

        names = result.names
        best_box = None
        best_conf = 0.0

        for cls_id, conf, box in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist()
        ):
            cls_name = names[int(cls_id)]

            if cls_name in target_classes and conf > best_conf:
                best_conf = float(conf)
                best_box = box

        if best_box is None or best_conf < self.min_conf:
            return None

        x1, y1, x2, y2 = map(int, best_box)
        return image.crop((x1, y1, x2, y2))
