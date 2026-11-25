"""Detector facial offline com YuNet ONNX (OpenCV Zoo) usando apenas onnxruntime, NumPy e Pillow."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

BACKEND_DIR = Path(__file__).resolve().parents[2]
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)

_SESSION_LOCK = threading.Lock()
_SESSIONS: Dict[str, Any] = {}


def yunet_model_path() -> Path:
    """Caminho local do modelo YuNet ONNX (reaproveita models/vision do backend se ja baixado)."""
    default_path = BACKEND_DIR / "data" / "models" / "vision" / YUNET_FILENAME
    data_env = os.environ.get("BLIND_DATA_DIR")
    if data_env:
        custom = Path(data_env) / "models" / "vision" / YUNET_FILENAME
        if custom.is_file() or not default_path.is_file():
            return custom
    return default_path


def ensure_yunet_model(timeout: float = 30.0) -> Path:
    """Garante que o modelo YuNet ONNX esteja baixado localmente antes da deteccao."""
    target = yunet_model_path()
    if target.is_file() and target.stat().st_size > 100_000:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".onnx.part")
    import httpx

    last_err: Optional[Exception] = None
    for url in YUNET_URLS:
        try:
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
            if tmp.stat().st_size > 100_000:
                os.replace(tmp, target)
                return target
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
    raise RuntimeError(
        "Não consegui baixar o modelo local de detecção facial YuNet (sem conexão ou erro na rede). "
        f"Detalhe: {last_err}"
    )


@dataclass
class FaceDetection:
    """Deteccao facial normalizada no intervalo [0, 1] com 5 pontos de referencia."""

    x: float
    y: float
    w: float
    h: float
    cx: float
    cy: float
    score: float
    right_eye: Tuple[float, float]
    left_eye: Tuple[float, float]
    nose: Tuple[float, float]
    right_mouth: Tuple[float, float]
    left_mouth: Tuple[float, float]
    eye_mid_x: float
    eye_mid_y: float
    roll_deg: float
    yaw_ratio: float
    mouth_mid_y: float
    pitch_ratio: float
    head_top_y: float
    chin_bottom_y: float
    margin_left: float
    margin_right: float
    margin_top: float
    margin_bottom: float

    @classmethod
    def from_normalized(
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        score: float,
        right_eye: Tuple[float, float],
        left_eye: Tuple[float, float],
        nose: Tuple[float, float],
        right_mouth: Tuple[float, float],
        left_mouth: Tuple[float, float],
    ) -> "FaceDetection":
        e_left_img, e_right_img = (
            (right_eye, left_eye) if right_eye[0] <= left_eye[0] else (left_eye, right_eye)
        )
        eye_span_x = max(e_right_img[0] - e_left_img[0], 1e-5)
        eye_mid_x = (e_left_img[0] + e_right_img[0]) / 2.0
        eye_mid_y = (e_left_img[1] + e_right_img[1]) / 2.0
        roll_deg = math.degrees(math.atan2(e_right_img[1] - e_left_img[1], eye_span_x))
        yaw_ratio = (nose[0] - e_left_img[0]) / eye_span_x
        mouth_mid_y = (right_mouth[1] + left_mouth[1]) / 2.0
        pitch_ratio = (nose[1] - eye_mid_y) / max(mouth_mid_y - eye_mid_y, 1e-5)
        head_top_y = y - 0.20 * h
        chin_bottom_y = y + h
        return cls(
            x=float(x),
            y=float(y),
            w=float(w),
            h=float(h),
            cx=float(x + w / 2.0),
            cy=float(y + h / 2.0),
            score=float(score),
            right_eye=(float(right_eye[0]), float(right_eye[1])),
            left_eye=(float(left_eye[0]), float(left_eye[1])),
            nose=(float(nose[0]), float(nose[1])),
            right_mouth=(float(right_mouth[0]), float(right_mouth[1])),
            left_mouth=(float(left_mouth[0]), float(left_mouth[1])),
            eye_mid_x=float(eye_mid_x),
            eye_mid_y=float(eye_mid_y),
            roll_deg=float(roll_deg),
            yaw_ratio=float(yaw_ratio),
            mouth_mid_y=float(mouth_mid_y),
            pitch_ratio=float(pitch_ratio),
            head_top_y=float(head_top_y),
            chin_bottom_y=float(chin_bottom_y),
            margin_left=float(x),
            margin_right=float(1.0 - (x + w)),
            margin_top=float(head_top_y),
            margin_bottom=float(1.0 - chin_bottom_y),
        )


def _nms_xywh(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, top_k: int) -> List[int]:
    """Non-Maximum Suppression guloso em NumPy puro para caixas [x1, y1, w, h]."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    areas = np.maximum(0.0, boxes[:, 2]) * np.maximum(0.0, boxes[:, 3])
    order = np.argsort(scores)[::-1]
    kept: List[int] = []
    while order.size > 0 and len(kept) < top_k:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return kept


def decode_yunet_outputs(
    outputs_by_name: Dict[str, np.ndarray],
    pad_w: int,
    pad_h: int,
    orig_w: int,
    orig_h: int,
    scale_x: float,
    scale_y: float,
    score_threshold: float = 0.62,
    nms_threshold: float = 0.30,
    top_k: int = 20,
) -> List[FaceDetection]:
    """Decodifica os 12 tensores brutos do YuNet seguindo exatamente o FaceDetectorYNImpl do OpenCV."""
    if orig_w <= 0 or orig_h <= 0:
        return []
    box_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []
    kps_parts: List[np.ndarray] = []

    for stride in (8, 16, 32):
        cols = pad_w // stride
        rows = pad_h // stride
        if cols <= 0 or rows <= 0:
            continue
        cls_arr = np.clip(np.asarray(outputs_by_name[f"cls_{stride}"], dtype=np.float32).reshape(-1), 0.0, 1.0)
        obj_arr = np.clip(np.asarray(outputs_by_name[f"obj_{stride}"], dtype=np.float32).reshape(-1), 0.0, 1.0)
        scores = np.sqrt(cls_arr * obj_arr)
        idx = np.where(scores >= score_threshold)[0]
        if idx.size == 0:
            continue

        bbox_arr = np.asarray(outputs_by_name[f"bbox_{stride}"], dtype=np.float32).reshape(-1, 4)[idx]
        kps_arr = np.asarray(outputs_by_name[f"kps_{stride}"], dtype=np.float32).reshape(-1, 10)[idx]
        r = (idx // cols).astype(np.float32)
        c = (idx % cols).astype(np.float32)

        cx = (c + bbox_arr[:, 0]) * stride
        cy = (r + bbox_arr[:, 1]) * stride
        bw = np.exp(np.clip(bbox_arr[:, 2], -10.0, 10.0)) * stride
        bh = np.exp(np.clip(bbox_arr[:, 3], -10.0, 10.0)) * stride
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0

        decoded_kps = np.empty((idx.size, 10), dtype=np.float32)
        for n in range(5):
            decoded_kps[:, 2 * n] = (kps_arr[:, 2 * n] + c) * stride
            decoded_kps[:, 2 * n + 1] = (kps_arr[:, 2 * n + 1] + r) * stride

        box_parts.append(np.stack([x1, y1, bw, bh], axis=1))
        score_parts.append(scores[idx])
        kps_parts.append(decoded_kps)

    if not box_parts:
        return []

    all_boxes = np.vstack(box_parts)
    all_scores = np.concatenate(score_parts)
    all_kps = np.vstack(kps_parts)
    kept = _nms_xywh(all_boxes, all_scores, iou_threshold=nms_threshold, top_k=top_k)

    detections: List[FaceDetection] = []
    inv_w = scale_x / float(orig_w)
    inv_h = scale_y / float(orig_h)
    for i in kept:
        bx, by, bw, bh = all_boxes[i]
        k = all_kps[i]
        det = FaceDetection.from_normalized(
            x=float(bx) * inv_w,
            y=float(by) * inv_h,
            w=float(bw) * inv_w,
            h=float(bh) * inv_h,
            score=float(all_scores[i]),
            right_eye=(float(k[0]) * inv_w, float(k[1]) * inv_h),
            left_eye=(float(k[2]) * inv_w, float(k[3]) * inv_h),
            nose=(float(k[4]) * inv_w, float(k[5]) * inv_h),
            right_mouth=(float(k[6]) * inv_w, float(k[7]) * inv_h),
            left_mouth=(float(k[8]) * inv_w, float(k[9]) * inv_h),
        )
        detections.append(det)

    detections.sort(key=lambda d: max(d.w * d.h, 0.0) * d.score, reverse=True)
    return detections


class YuNetDetector:
    """Wrapper leve em onnxruntime para o YuNet ONNX, com cache de sessao CPU."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        score_threshold: float = 0.62,
        nms_threshold: float = 0.30,
        target_max_side: int = 320,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.target_max_side = int(target_max_side)
        self._session: Any = None
        self._input_name: str = ""
        self._output_names: List[str] = []
        self._fixed_hw: Optional[Tuple[int, int]] = None

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        path = self.model_path if self.model_path is not None else ensure_yunet_model()
        key = str(path.resolve())
        with _SESSION_LOCK:
            sess = _SESSIONS.get(key)
            if sess is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 1
                opts.intra_op_num_threads = 2
                sess = ort.InferenceSession(key, sess_options=opts, providers=["CPUExecutionProvider"])
                _SESSIONS[key] = sess
        self._session = sess
        inp = sess.get_inputs()[0]
        self._input_name = inp.name
        self._output_names = [o.name for o in sess.get_outputs()]
        shape = list(inp.shape or [])
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int) and shape[2] > 0 and shape[3] > 0:
            self._fixed_hw = (shape[2], shape[3])
        else:
            self._fixed_hw = None

    def _prepare_input(self, rgb: np.ndarray) -> Tuple[np.ndarray, int, int, int, int, float, float]:
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("Quadro RGB invalido para deteccao facial.")
        orig_h, orig_w = arr.shape[:2]
        pil = PILImage.fromarray(arr[:, :, :3].astype(np.uint8, copy=False))
        if self._fixed_hw is not None:
            in_h, in_w = self._fixed_hw
            fit = min(in_w / float(orig_w), in_h / float(orig_h))
            rw = max(1, int(round(orig_w * fit)))
            rh = max(1, int(round(orig_h * fit)))
            resized = pil.resize((rw, rh), PILImage.Resampling.BILINEAR) if (rw, rh) != (orig_w, orig_h) else pil
            bgr = np.asarray(resized, dtype=np.float32)[:, :, ::-1]
            canvas = np.zeros((in_h, in_w, 3), dtype=np.float32)
            canvas[:rh, :rw, :] = bgr
            pad_w, pad_h = in_w, in_h
            scale_x = orig_w / float(rw)
            scale_y = orig_h / float(rh)
        else:
            scale = min(1.0, self.target_max_side / float(max(orig_w, orig_h)))
            rw = max(32, int(round(orig_w * scale)))
            rh = max(32, int(round(orig_h * scale)))
            resized = pil.resize((rw, rh), PILImage.Resampling.BILINEAR) if (rw, rh) != (orig_w, orig_h) else pil
            bgr = np.asarray(resized, dtype=np.float32)[:, :, ::-1]
            pad_w = ((rw - 1) // 32 + 1) * 32
            pad_h = ((rh - 1) // 32 + 1) * 32
            if pad_w != rw or pad_h != rh:
                canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
                canvas[:rh, :rw, :] = bgr
            else:
                canvas = bgr
            scale_x = orig_w / float(rw)
            scale_y = orig_h / float(rh)
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[np.newaxis, :, :, :], dtype=np.float32)
        return blob, pad_w, pad_h, orig_w, orig_h, scale_x, scale_y

    def detect(self, rgb: np.ndarray) -> List[FaceDetection]:
        """Detecta rostos no quadro RGB e devolve lista ordenada pelo rosto principal primeiro."""
        if rgb is None or np.asarray(rgb).size == 0:
            return []
        self._ensure_session()
        blob, pad_w, pad_h, orig_w, orig_h, scale_x, scale_y = self._prepare_input(rgb)
        raw_outputs = self._session.run(self._output_names, {self._input_name: blob})
        outputs_by_name = dict(zip(self._output_names, raw_outputs))
        return decode_yunet_outputs(
            outputs_by_name=outputs_by_name,
            pad_w=pad_w,
            pad_h=pad_h,
            orig_w=orig_w,
            orig_h=orig_h,
            scale_x=scale_x,
            scale_y=scale_y,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )
