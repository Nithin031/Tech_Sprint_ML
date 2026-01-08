import io
import numpy as np
import cv2
from PIL import Image, ExifTags
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

# ==================== ENUMS ====================

class MLJobStatus(Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"

class FreshnessLevel(Enum):
    FRESH = "FRESH"
    MODERATE = "MODERATE"
    SPOILED = "SPOILED"
    PENDING = "PENDING"

class ImageType(Enum):
    EYE = "EYE"
    GILL = "GILL"

class ValidationError(Enum):
    INVALID_ASPECT_RATIO = "Aspect ratio invalid"
    TOO_BLURRY = "Image too blurry"
    TOO_DARK = "Image too dark"
    TOO_BRIGHT = "Image overexposed"
    CORRUPT_FILE = "Image file corrupted or unreadable"
    UNSUPPORTED_FORMAT = "Image format not supported"
    FILE_TOO_LARGE = "File size exceeds limit"
    MISSING_IMAGE = "Required image missing" # Added to prevent missing enum error

# ==================== DATA STRUCTURES ====================

@dataclass
class ImageMetadata:
    width: int
    height: int
    aspect_ratio: float
    file_size_kb: float
    format: str
    has_exif: bool
    orientation: int
    blur_score: float
    brightness_mean: float
    contrast_std: float

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[str]
    metadata: Optional[ImageMetadata]
    corrected_image: Optional[Image.Image]

@dataclass
class ImageInput:
    image_type: ImageType
    raw_data: bytes
    filename: str
    upload_timestamp: float

# ==================== CONFIGURATION ====================

class ValidationConfig:
    MIN_WIDTH = 224      # Adjusted for EfficientNet
    MIN_HEIGHT = 224

    MIN_ASPECT_RATIO = 0.5
    MAX_ASPECT_RATIO = 2.0

    MIN_BLUR_SCORE = 100.0

    MIN_BRIGHTNESS = 30
    MAX_BRIGHTNESS = 240
    MIN_CONTRAST = 20

    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP"}

# ==================== IMAGE VALIDATOR ====================

class ImageValidator:
    """
    Validates a single image and outputs a YOLO-ready PIL image.
    Optimized for robustness: Low resolution is now a WARNING.
    """

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def validate(self, image_input: ImageInput) -> ValidationResult:
        errors: List[ValidationError] = []
        warnings: List[str] = []

        # 1. File Size Check
        file_size_kb = len(image_input.raw_data) / 1024
        if len(image_input.raw_data) > self.config.MAX_FILE_SIZE_BYTES:
            return ValidationResult(False, [ValidationError.FILE_TOO_LARGE], [], None, None)

        # 2. Decode Image
        try:
            pil_image = Image.open(io.BytesIO(image_input.raw_data))
            img_format = pil_image.format
            # Force format check if needed, mostly handled by PIL exceptions
            if img_format not in self.config.SUPPORTED_FORMATS:
                 # Soft pass for standard PIL formats, but you can restrict here
                 pass
        except Exception:
            return ValidationResult(False, [ValidationError.CORRUPT_FILE], [], None, None)

        # 3. Orientation & Color Correction
        pil_image, orientation = self._correct_orientation(pil_image)
        has_exif = orientation != 1

        # Convert to numpy for stats (ensure RGB)
        img_np = np.array(pil_image.convert("RGB"))

        height, width = img_np.shape[:2]
        aspect_ratio = width / height

        # ✅ FIXED: Resolution Check -> WARNING only
        if width < self.config.MIN_WIDTH or height < self.config.MIN_HEIGHT:
            warnings.append(f"Low Resolution: {width}x{height}. Recommended is 800x600.")

        # 4. Critical Logic Checks (Errors)
        if not (self.config.MIN_ASPECT_RATIO <= aspect_ratio <= self.config.MAX_ASPECT_RATIO):
            errors.append(ValidationError.INVALID_ASPECT_RATIO)

        blur_score = self._detect_blur(img_np)
        if blur_score < self.config.MIN_BLUR_SCORE:
            errors.append(ValidationError.TOO_BLURRY)

        brightness_mean, contrast_std = self._analyze_lighting(img_np)
        if brightness_mean < self.config.MIN_BRIGHTNESS:
            errors.append(ValidationError.TOO_DARK)
        elif brightness_mean > self.config.MAX_BRIGHTNESS:
            errors.append(ValidationError.TOO_BRIGHT)
        
        if contrast_std < self.config.MIN_CONTRAST:
            warnings.append("Low contrast detected")

        # 5. Compile Metadata
        metadata = ImageMetadata(
            width=width,
            height=height,
            aspect_ratio=aspect_ratio,
            file_size_kb=file_size_kb,
            format=str(img_format),
            has_exif=has_exif,
            orientation=orientation,
            blur_score=blur_score,
            brightness_mean=brightness_mean,
            contrast_std=contrast_std,
        )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
            corrected_image=pil_image if len(errors) == 0 else None,
        )

    # ---------------- INTERNAL HELPERS ----------------

    def _correct_orientation(self, pil_image: Image.Image) -> Tuple[Image.Image, int]:
        orientation = 1
        try:
            exif = pil_image._getexif()
            if exif:
                for tag, value in exif.items():
                    if ExifTags.TAGS.get(tag) == "Orientation":
                        orientation = value
                        break
                if orientation == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                elif orientation == 8:
                    pil_image = pil_image.rotate(90, expand=True)
        except Exception:
            pass
        return pil_image, orientation

    def _detect_blur(self, img_np: np.ndarray) -> float:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _analyze_lighting(self, img_np: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return float(np.mean(gray)), float(np.std(gray))

# ==================== ENFORCEMENT LAYER ====================

class InputJobProcessor:
    """
    Enforces compulsory EYE and GILL images.
    """
    def __init__(self):
        self.validator = ImageValidator()

    def process(self, eye_input: Optional[ImageInput], gill_input: Optional[ImageInput]) -> Dict[ImageType, ValidationResult]:
        if eye_input is None:
            raise ValueError("EYE image is compulsory")
        if gill_input is None:
            raise ValueError("GILL image is compulsory")

        eye_result = self.validator.validate(eye_input)
        gill_result = self.validator.validate(gill_input)

        if not eye_result.is_valid:
            raise ValueError(f"EYE image invalid: {[e.value for e in eye_result.errors]}")

        if not gill_result.is_valid:
            raise ValueError(f"GILL image invalid: {[e.value for e in gill_result.errors]}")

        return {
            ImageType.EYE: eye_result,
            ImageType.GILL: gill_result,
        }

if __name__ == "__main__":
    print("INPUT.py ready. Enforces compulsory EYE and GILL images.")