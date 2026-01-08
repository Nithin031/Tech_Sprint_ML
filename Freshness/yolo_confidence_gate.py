from enum import Enum
from ultralytics import YOLO


# ==================== ENUM ====================

class JobOutcome(Enum):
    PROCEED = "PROCEED"
    FAILED = "FAILED"


# ==================== YOLO CONFIDENCE GATE ====================

class YOLOConfidenceGate:
    """
    Single YOLOv model (best.pt) with eye/gill classes.

    Supports multilingual / dataset-specific labels:
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
        eye_min_conf: float = 0.40,
        gill_min_conf: float = 0.40,
        eye_max_conf_in_gill: float = 0.20,
        gill_max_conf_in_eye: float = 0.20,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)

        self.eye_min_conf = eye_min_conf
        self.gill_min_conf = gill_min_conf
        self.eye_max_conf_in_gill = eye_max_conf_in_gill
        self.gill_max_conf_in_eye = gill_max_conf_in_eye

    # ---------------- CORE API ----------------

    def run(self, eye_image, gill_image) -> JobOutcome:
        """
        eye_image  : PIL.Image (declared EYE input)
        gill_image : PIL.Image (declared GILL input)
        """

        eye_res  = self.model(eye_image, verbose=False)[0]
        gill_res = self.model(gill_image, verbose=False)[0]

        eye_eye_conf, eye_gill_conf   = self._max_conf_per_class(eye_res)
        gill_eye_conf, gill_gill_conf = self._max_conf_per_class(gill_res)

        # ---------- HARD FAIL RULES ----------

        # EYE image checks
        if eye_eye_conf < self.eye_min_conf:
            return JobOutcome.FAILED

        if eye_gill_conf > self.gill_max_conf_in_eye:
            return JobOutcome.FAILED

        # GILL image checks
        if gill_gill_conf < self.gill_min_conf:
            return JobOutcome.FAILED

        if gill_eye_conf > self.eye_max_conf_in_gill:
            return JobOutcome.FAILED

        return JobOutcome.PROCEED

    # ---------------- HELPERS ----------------

    def _max_conf_per_class(self, result):
        """
        Returns:
            (max_eye_confidence, max_gill_confidence)
        """
        max_eye = 0.0
        max_gill = 0.0

        if result.boxes is None or len(result.boxes) == 0:
            return max_eye, max_gill

        names = result.names

        for cls_id, conf in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist()
        ):
            cls_name = names[int(cls_id)]

            # ---- NORMALIZED CLASS CHECK ----
            if cls_name in self.EYE_CLASSES:
                max_eye = max(max_eye, float(conf))
            elif cls_name in self.GILL_CLASSES:
                max_gill = max(max_gill, float(conf))

        return max_eye, max_gill
