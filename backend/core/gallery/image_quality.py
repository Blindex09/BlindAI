"""Analise rapida de iluminacao, nitidez e movimento entre quadros para fotos acessiveis."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image as PILImage


def _rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    """Converte quadro RGB uint8/float em luminancia BT.601 (H, W) float32."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def _resize_gray_for_sharpness(gray: np.ndarray, target_width: int) -> np.ndarray:
    """Redimensiona um recorte em tons de cinza para tornar a variancia do Laplaciano comparavel."""
    h, w = gray.shape[:2]
    if w <= target_width or w < 12 or h < 12:
        return gray
    new_h = max(8, int(round(h * (target_width / float(w)))))
    pil = PILImage.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    resized = pil.resize((target_width, new_h), PILImage.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Variancia do Laplaciano 2D discreto em NumPy puro (maior = imagem mais nitida)."""
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return 0.0
    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    return float(np.var(lap))


def _clamp_roi(
    box: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    fx, fy, fw, fh = box
    x0 = max(0, min(width - 1, int(round(fx * width))))
    y0 = max(0, min(height - 1, int(round(fy * height))))
    x1 = max(x0, min(width, int(round((fx + fw) * width))))
    y1 = max(y0, min(height, int(round((fy + fh) * height))))
    if (x1 - x0) < 8 or (y1 - y0) < 8:
        return None
    return x0, y0, x1, y1


def analyze_quality(
    rgb: np.ndarray,
    face_box: Optional[Tuple[float, float, float, float]] = None,
) -> Dict[str, Any]:
    """Avalia brilho, contraluz, sombra lateral e nitidez de um quadro RGB."""
    luma = _rgb_to_luma(rgb)
    h, w = luma.shape[:2]
    brightness = float(np.mean(luma))
    dark_ratio = float(np.mean(luma < 30.0))
    overexposed_ratio = float(np.mean(luma > 245.0))

    roi_rect = _clamp_roi(face_box, w, h) if face_box is not None else None
    if roi_rect is not None:
        x0, y0, x1, y1 = roi_rect
        roi = luma[y0:y1, x0:x1]
        roi_brightness = float(np.mean(roi))
        bg_mask = np.ones((h, w), dtype=bool)
        bg_mask[y0:y1, x0:x1] = False
        background_brightness = float(np.mean(luma[bg_mask])) if np.any(bg_mask) else roi_brightness
        backlit = bool(roi_brightness < 78.0 and (background_brightness - roi_brightness) > 80.0)
        mid_x = max(1, roi.shape[1] // 2)
        left_mean = float(np.mean(roi[:, :mid_x]))
        right_mean = float(np.mean(roi[:, mid_x:])) if roi.shape[1] > mid_x else left_mean
        side_shadow_diff = float(abs(left_mean - right_mean))
        side_shadow = bool(side_shadow_diff > 45.0)
        sharp_roi = _resize_gray_for_sharpness(roi, 160)
    else:
        my0, my1 = int(h * 0.15), max(int(h * 0.85), int(h * 0.15) + 1)
        mx0, mx1 = int(w * 0.15), max(int(w * 0.85), int(w * 0.15) + 1)
        roi = luma[my0:my1, mx0:mx1]
        roi_brightness = float(np.mean(roi)) if roi.size else brightness
        background_brightness = brightness
        backlit = False
        side_shadow_diff = 0.0
        side_shadow = False
        sharp_roi = _resize_gray_for_sharpness(roi if roi.size else luma, 320)

    sharpness = _laplacian_variance(sharp_roi)

    if backlit:
        lighting_status = "backlit"
    elif roi_brightness < 48.0:
        lighting_status = "too_dark"
    elif roi_brightness > 232.0 or overexposed_ratio > 0.35:
        lighting_status = "too_bright"
    elif side_shadow:
        lighting_status = "side_shadow"
    else:
        lighting_status = "ok"

    if sharpness < 35.0:
        sharpness_status = "blurry"
    elif sharpness < 85.0:
        sharpness_status = "acceptable"
    else:
        sharpness_status = "sharp"

    lighting_phrases = {
        "too_dark": "O ambiente está muito escuro.",
        "backlit": "Há muita luz atrás de você deixando o rosto escuro.",
        "too_bright": "A imagem está estourada de tanta luz.",
        "side_shadow": "Um lado do rosto está bem mais escuro que o outro.",
        "ok": "Iluminação boa.",
    }
    sharp_phrases = {
        "blurry": "A foto saiu um pouco borrada.",
        "acceptable": "Nitidez razoável.",
        "sharp": "Foto bem nítida.",
    }
    if lighting_status == "ok" and sharpness_status != "blurry":
        spoken_summary = f"{lighting_phrases[lighting_status]} {sharp_phrases[sharpness_status]}"
    else:
        spoken_summary = f"{lighting_phrases[lighting_status]} {sharp_phrases[sharpness_status]}".strip()

    return {
        "brightness": round(brightness, 2),
        "roi_brightness": round(roi_brightness, 2),
        "face_brightness": round(roi_brightness, 2) if roi_rect is not None else None,
        "background_brightness": round(background_brightness, 2),
        "dark_ratio": round(dark_ratio, 4),
        "overexposed_ratio": round(overexposed_ratio, 4),
        "backlit": backlit,
        "side_shadow": side_shadow,
        "side_shadow_diff": round(side_shadow_diff, 2),
        "sharpness": round(sharpness, 2),
        "lighting_status": lighting_status,
        "sharpness_status": sharpness_status,
        "spoken_summary": spoken_summary,
    }


def frame_motion_diff(prev_rgb: Optional[np.ndarray], curr_rgb: Optional[np.ndarray]) -> float:
    """Diferenca media absoluta (0 a 255) entre dois quadros consecutivos em escala reduzida."""
    if prev_rgb is None or curr_rgb is None or prev_rgb.shape != curr_rgb.shape:
        return 255.0
    g0 = _rgb_to_luma(prev_rgb[::4, ::4])
    g1 = _rgb_to_luma(curr_rgb[::4, ::4])
    return float(np.mean(np.abs(g1 - g0)))
