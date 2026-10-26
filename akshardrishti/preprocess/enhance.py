"""Image preprocessing for scanned Indian government documents.

Order matters and is deliberate:
    1. grayscale      -- everything downstream is single-channel
    2. resize/DPI     -- do this BEFORE deskew so angle estimation is cheap
    3. deskew         -- before denoise, so interpolation blur is then cleaned
    4. denoise        -- before CLAHE, so we don't amplify noise
    5. CLAHE          -- last contrast step
    6. binarize       -- optional, off by default

Why binarization is off by default: YOLO and TrOCR are both trained on natural
grayscale/RGB images and lose accuracy on hard-thresholded input. Binarization
helps a CTC/CRNN branch on clean printed text. So it belongs at the *line crop*
stage for the CRNN, not on the whole page.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    image: np.ndarray
    applied: list[str] = field(default_factory=list)
    rotation_deg: float = 0.0
    orientation_rotation_cw: int = 0
    scale: float = 1.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.image.shape[:2]


# --------------------------------------------------------------------- deskew


def estimate_skew_hough(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Estimate page skew in degrees via Hough lines on text edges.

    Positive angle = content rotated counter-clockwise; rotate by ``-angle``
    to correct. Returns 0.0 when it cannot find a confident answer -- a wrong
    deskew is far more damaging than none, so we fail closed.
    """
    if gray.ndim != 2:
        raise ValueError("estimate_skew_hough expects a single-channel image")

    # Downscale: skew estimation does not need full resolution and this keeps
    # a 4000px scan from costing a second.
    h, w = gray.shape
    scale = 1000.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray

    # Dilate horizontally so words merge into line-shaped blobs; the dominant
    # edge direction of those blobs is the text baseline direction.
    thresh = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    merged = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(merged, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 720, threshold=80,
        minLineLength=max(30, small.shape[1] // 8), maxLineGap=20,
    )
    if lines is not None and lines.ndim == 3:
        lines = lines[:, 0]
    if lines is None or len(lines) < 5:
        log.debug("deskew: too few Hough lines (%s), skipping", 0 if lines is None else len(lines))
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep near-horizontal lines only; verticals are rules and table borders.
        if abs(ang) <= max_angle:
            angles.append(ang)

    if len(angles) < 5:
        log.debug("deskew: too few near-horizontal lines (%d), skipping", len(angles))
        return 0.0

    # Median is robust to the handful of diagonal strokes that survive.
    angle = float(np.median(angles))
    if abs(angle) < 0.1 or abs(angle) > max_angle:
        return 0.0
    return angle


def rotate_bound(image: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    """Rotate without cropping corners; expands canvas to fit."""
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(
        image, m, (nw, nh),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
    )


def rotate_page_quadrants(image: np.ndarray, rotation_cw: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an image clockwise by a right angle and return its pixel transform.

    The 3x3 matrix maps original pixel-index coordinates (x, y, 1) into the
    rotated image. Bounding boxes can be transformed by mapping all four
    corners and taking the resulting axis-aligned bounds.
    """
    if rotation_cw not in (0, 90, 180, 270):
        raise ValueError("rotation_cw must be one of 0, 90, 180, or 270")
    if image is None or image.size == 0:
        raise ValueError("cannot rotate an empty page")
    h, w = image.shape[:2]
    transforms = {
        0: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        90: np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64),
        180: np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], dtype=np.float64),
        270: np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], dtype=np.float64),
    }
    rotate_codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotated = image.copy() if rotation_cw == 0 else cv2.rotate(image, rotate_codes[rotation_cw])
    return rotated, transforms[rotation_cw]


def transform_bbox_quadrants(box: tuple[float, float, float, float], transform: np.ndarray) -> tuple[float, float, float, float]:
    """Transform an xyxy pixel-coordinate box using a quadrant rotation matrix."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], dtype=np.float64)
    mapped = corners @ transform.T
    return (float(mapped[:, 0].min()), float(mapped[:, 1].min()),
            float(mapped[:, 0].max()), float(mapped[:, 1].max()))


def detect_orientation_tesseract_osd(gray: np.ndarray, min_confidence: float = 8.0) -> tuple[int, float | None]:
    """Use Tesseract OSD's reported clockwise correction when confidence passes.

    No local orientation score is synthesized. Missing Tesseract, OSD data, or
    low confidence fails closed and returns an unrotated decision.
    """
    if gray.ndim != 2:
        raise ValueError("detect_orientation_tesseract_osd expects grayscale input")
    import pytesseract

    result = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
    rotation = int(result.get("rotate", 0)) % 360
    confidence = float(result.get("orientation_conf", 0.0))
    if rotation not in (0, 90, 180, 270):
        log.warning("orientation OSD returned unsupported rotation %s; leaving page unchanged", rotation)
        return 0, confidence
    if confidence < min_confidence:
        log.info("orientation OSD confidence %.2f is below %.2f; leaving page unchanged", confidence, min_confidence)
        return 0, confidence
    return rotation, confidence


def deskew(gray: np.ndarray, max_angle: float = 15.0) -> tuple[np.ndarray, float]:
    angle = estimate_skew_hough(gray, max_angle=max_angle)
    if angle == 0.0:
        return gray, 0.0
    return rotate_bound(gray, angle), angle


# ------------------------------------------------------------------- contrast


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation.

    The right tool for scans with uneven illumination -- a photographed page
    that is bright on one side and dim on the other, which is extremely common
    in the government-document sources we target.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, method: str = "fastnlmeans", strength: int = 7) -> np.ndarray:
    if method == "fastnlmeans":
        return cv2.fastNlMeansDenoising(gray, None, h=float(strength), templateWindowSize=7, searchWindowSize=21)
    if method == "bilateral":
        return cv2.bilateralFilter(gray, d=7, sigmaColor=float(strength) * 10, sigmaSpace=float(strength) * 10)
    if method == "median":
        k = strength if strength % 2 == 1 else strength + 1
        return cv2.medianBlur(gray, min(k, 9))
    raise ValueError(f"unknown denoise method: {method!r}")


# ----------------------------------------------------------------- binarize


def sauvola_binarize(gray: np.ndarray, window: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """Sauvola adaptive threshold -- better than Otsu on uneven scans.

    Implemented with integral images so it stays fast on full pages.
    """
    if window % 2 == 0:
        window += 1
    img = gray.astype(np.float64)
    mean = cv2.boxFilter(img, ddepth=-1, ksize=(window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sq_mean = cv2.boxFilter(img ** 2, ddepth=-1, ksize=(window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 0.0))
    threshold = mean * (1.0 + k * ((std / r) - 1.0))
    return ((img > threshold) * 255).astype(np.uint8)


def binarize(gray: np.ndarray, method: str = "sauvola", window: int = 25) -> np.ndarray:
    if method == "sauvola":
        return sauvola_binarize(gray, window=window)
    if method == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if method == "adaptive":
        w = window if window % 2 == 1 else window + 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, w, 10)
    raise ValueError(f"unknown binarize method: {method!r}")


# ------------------------------------------------------------------ pipeline


def resize_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=interp), scale


def preprocess_page(image: np.ndarray, cfg: dict | None = None) -> PreprocessResult:
    """Run the configured preprocessing chain over one page image.

    ``cfg`` is the ``preprocess:`` block of pipeline.yaml (a plain dict).
    """
    cfg = cfg or {}
    applied: list[str] = []

    if image is None or image.size == 0:
        raise ValueError("preprocess_page received an empty image")

    if not cfg.get("enabled", True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return PreprocessResult(image=gray, applied=["grayscale"], rotation_deg=0.0, scale=1.0)

    # 1. grayscale
    if image.ndim == 3:
        out = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        applied.append("grayscale")
    else:
        out = image.copy()

    # 2. resize
    scale = 1.0
    max_side = int(cfg.get("max_side", 2048))
    if max_side > 0:
        out, scale = resize_max_side(out, max_side)
        if scale != 1.0:
            applied.append(f"resize(x{scale:.3f})")

    # 3. Optional right-angle orientation correction. Disabled by default.
    orientation_rotation = 0
    orientation_confidence = None
    orientation_cfg = cfg.get("orientation", {}) or {}
    if orientation_cfg.get("enabled", False):
        method = orientation_cfg.get("method", "tesseract_osd")
        if method != "tesseract_osd":
            raise ValueError(f"unknown page orientation method: {method!r}")
        try:
            orientation_rotation, orientation_confidence = detect_orientation_tesseract_osd(
                out, min_confidence=float(orientation_cfg.get("min_confidence", 8.0))
            )
            if orientation_rotation:
                out, _ = rotate_page_quadrants(out, orientation_rotation)
                applied.append(f"orientation_correction(cw={orientation_rotation}deg)")
        except Exception as exc:
            # A requested but unavailable/failed orientation provider must not
            # guess a rotation. Log the failure and preserve the page as-is.
            log.warning("orientation correction unavailable (%s); leaving page unchanged", exc)

    # 4. deskew
    rotation = 0.0
    ds = cfg.get("deskew", {}) or {}
    if ds.get("enabled", True):
        out, rotation = deskew(out, max_angle=float(ds.get("max_angle", 15.0)))
        if rotation != 0.0:
            applied.append(f"deskew({rotation:+.2f}deg)")

    # 5. denoise
    dn = cfg.get("denoise", {}) or {}
    if dn.get("enabled", True):
        out = denoise(out, method=dn.get("method", "fastnlmeans"), strength=int(dn.get("strength", 7)))
        applied.append(f"denoise({dn.get('method', 'fastnlmeans')})")

    # 6. CLAHE
    cl = cfg.get("clahe", {}) or {}
    if cl.get("enabled", True):
        out = apply_clahe(out, clip_limit=float(cl.get("clip_limit", 2.0)), tile_grid=tuple(cl.get("tile_grid", (8, 8))))
        applied.append("clahe")

    # 7. binarize (off by default)
    bn = cfg.get("binarize", {}) or {}
    if bn.get("enabled", False):
        out = binarize(out, method=bn.get("method", "sauvola"), window=int(bn.get("window", 25)))
        applied.append(f"binarize({bn.get('method', 'sauvola')})")

    return PreprocessResult(
        image=out, applied=applied, rotation_deg=rotation,
        orientation_rotation_cw=orientation_rotation, scale=scale,
    )


def preprocess_line_crop(crop: np.ndarray, target_height: int = 32, binarize_crop: bool = True) -> np.ndarray:
    """Prepare a single text-line crop for a CTC/CRNN recognizer.

    This is where binarization earns its keep -- on a tight line crop of printed
    Devanagari it sharpens the shirorekha and the conjuncts hanging beneath it.
    """
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("empty line crop")
    scale = target_height / h
    new_w = max(8, int(round(w * scale)))
    crop = cv2.resize(crop, (new_w, target_height), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    if binarize_crop:
        crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return crop
