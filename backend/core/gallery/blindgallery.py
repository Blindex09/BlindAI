"""Orquestrador do BlindPhoto e BlindGallery: enquadramento guiado, captura em rajada e visualizacao."""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

from . import content_safety
from .camera import CameraStream, VoiceCoach, list_cameras
from .face_engine import FaceDetection, YuNetDetector, ensure_yunet_model
from .framing_coach import (
    BurstFrame,
    FramingEvaluator,
    ObjectStabilityTracker,
    StillnessTracker,
    resolve_stability_mode,
    select_best_burst_frame,
)
from .gallery_store import (
    _classify_ext,
    check_blocked_file,
    confirm_staged_photo,
    discard_staged_photo,
    get_item,
    is_blocked_path,
    is_verified_safe,
    list_albums,
    list_items,
    mark_file_safe,
    remove_item,
    save_captured_photo,
    save_note,
    scan_folders,
    search_items,
    stage_captured_photo,
)
from .image_quality import analyze_quality, frame_motion_diff

VALID_CAPTURE_MODES = ("portrait", "object", "money", "text", "general")
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def resolve_capture_mode(mode: str) -> str:
    m = (mode or "general").strip().lower()
    aliases = {
        "portrait": "portrait",
        "retrato": "portrait",
        "selfie": "portrait",
        "rosto": "portrait",
        "pessoa": "portrait",
        "object": "object",
        "objeto": "object",
        "money": "money",
        "dinheiro": "money",
        "nota": "money",
        "notas": "money",
        "cedula": "money",
        "cédula": "money",
        "moeda": "money",
        "moedas": "money",
        "text": "text",
        "texto": "text",
        "remedio": "text",
        "remédio": "text",
        "rotulo": "text",
        "rótulo": "text",
        "documento": "text",
        "general": "general",
        "geral": "general",
    }
    if m not in aliases:
        raise ValueError("Modo inválido. Use: portrait, object, money, text ou general.")
    return aliases[m]


def _encode_preview_jpeg(rgb: np.ndarray, max_side: int = 1568, quality: int = 88) -> bytes:
    arr = np.asarray(rgb, dtype=np.uint8)
    pil = PILImage.fromarray(arr[:, :, :3]).convert("RGB")
    pil.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _face_box(det: Optional[FaceDetection]) -> Optional[Tuple[float, float, float, float]]:
    if det is None:
        return None
    return (det.x, det.y, det.w, det.h)


def _lighting_score(quality: Dict[str, Any]) -> float:
    status = quality.get("lighting_status", "ok")
    if status == "ok":
        return 1.0
    if status == "side_shadow":
        return 0.72
    if status == "backlit":
        return 0.45
    return 0.30


def _capture_mode_note(mode: str, question: str, quality: Dict[str, Any], staged: Dict[str, Any]) -> str:
    q_extra = f" Pergunta da pessoa: {question.strip()}." if (question or "").strip() else ""
    safety_gate = (
        f"ATENCAO OBRIGATORIA DE SEGURANCA: esta foto ainda esta apenas em area temporaria ({staged['pending_id']}), "
        "NAO foi salva na pasta Imagens nem na galeria. Primeiro confira visualmente se ha nudez, sexo explicito "
        "ou violencia grafica extrema: se houver qualquer conteudo improprio, chame IMMEDIATAMENTE "
        f"blindgallery_discard_capture(pending_id='{staged['pending_id']}', reason='unsafe') e avise que o Blind "
        "nao salva midias sexuais ou violentas. Se estiver segura, descreva abaixo e so entao salve na galeria "
        f"chamando blindgallery_confirm_save(pending_id='{staged['pending_id']}', safe_verified=True, title=..., "
        "description=...) sem incluir dados sensiveis na nota."
    )
    quality_note = f" Qualidade medida na foto: {quality.get('spoken_summary', '')}"
    if mode == "money":
        return (
            "Foto de dinheiro capturada pelo BlindPhoto. "
            + safety_gate
            + " Identifique com atenção cada cédula ou moeda do Real "
            "(ou moeda estrangeira), diga o valor em números e repita por extenso (ex: cinquenta reais), some o total "
            "se houver mais de uma nota e diga o grau de certeza. Se estiver dobrada, cortada, borrada ou escura, avise "
            "sem inventar valor. Siga a habilidade dados-sensiveis (se a pessoa preferir privacidade, fale com falar_em_segredo)."
            + quality_note
            + q_extra
        )
    if mode == "text":
        return (
            "Foto de texto, rótulo ou remédio capturada pelo BlindPhoto. "
            + safety_gate
            + " Leia palavra por palavra sem resumir; se for "
            "remédio, dose, validade ou documento, repita os números separados e diga o grau de certeza sem recusar "
            "(habilidade dados-sensiveis e descrevendo-imagens). Se alguma parte estiver cortada ou borrada, avise e peça outra foto."
            + quality_note
            + q_extra
        )
    if mode == "object":
        return (
            "Foto de objeto capturada pelo BlindPhoto. "
            + safety_gate
            + " Diga logo o que é o objeto, cor, posição aproximada e qualquer "
            "texto ou marca visível. Se estiver cortado ou escuro, avise com clareza."
            + quality_note
            + q_extra
        )
    if mode == "portrait":
        return (
            "Retrato/selfie capturado pelo BlindPhoto. "
            + safety_gate
            + " Descreva como a pessoa aparece (enquadramento, cabeça inteira, "
            "expressão, roupa, iluminação) seguindo descrevendo-imagens. Nunca tente identificar ninguém pelo rosto."
            + quality_note
            + q_extra
        )
    return (
        "Foto capturada pelo BlindPhoto. "
        + safety_gate
        + " Descreva para uma pessoa cega seguindo descrevendo-imagens."
        + quality_note
        + q_extra
    )


def check_position(
    camera_name: str = "",
    stability_mode: str = "normal",
    mirror_horizontal: bool = False,
    speak: bool = True,
) -> Dict[str, Any]:
    """Abre a câmera rapidamente, analisa posição do rosto, cortes e luz, e devolve orientação."""
    stab = resolve_stability_mode(stability_mode)
    ensure_yunet_model()
    detector = YuNetDetector()
    evaluator = FramingEvaluator(stability_mode=stab, mirror_horizontal=mirror_horizontal)

    samples: List[Tuple[List[FaceDetection], Dict[str, Any], Dict[str, Any]]] = []
    cam_used = ""
    with CameraStream(camera_name=camera_name) as cam:
        cam_used = cam.camera_name
        cam.wait_ready(warmup_frames=5, timeout=5.0)
        last_id = -1
        for _ in range(5):
            pkt = cam.get_latest(last_id=last_id, timeout=1.0)
            if pkt is None:
                break
            fid, ts, rgb = pkt
            last_id = fid
            dets = detector.detect(rgb)
            primary = dets[0] if dets else None
            qual = analyze_quality(rgb, face_box=_face_box(primary))
            framing = evaluator.evaluate(primary, ts, qual)
            samples.append((dets, qual, framing))

    if not samples:
        raise RuntimeError("Não consegui ler quadros da câmera para verificar a posição.")

    last_dets, last_qual, last_framing = samples[-1]
    spoken = last_framing["spoken_instruction"]
    if speak:
        with VoiceCoach(enabled=True, audio_cues=False) as coach:
            coach.say(spoken, priority="high", min_interval=0.0)
            coach.close(wait=True, drain=True, timeout=4.0)

    return {
        "ok": True,
        "camera": cam_used,
        "stability_mode": stab,
        "face_detected": last_framing["face_detected"],
        "faces_count": len(last_dets),
        "status": last_framing["status"],
        "issue_code": last_framing["issue_code"],
        "ready_to_capture": bool(last_framing["is_centered"]),
        "position": last_framing["filtered"],
        "lighting": {
            "status": last_qual["lighting_status"],
            "brightness": last_qual["brightness"],
            "face_brightness": last_qual["face_brightness"],
            "backlit": last_qual["backlit"],
            "side_shadow": last_qual["side_shadow"],
        },
        "sharpness": {
            "status": last_qual["sharpness_status"],
            "score": last_qual["sharpness"],
        },
        "spoken_guidance": spoken,
    }


def _collect_burst_portrait(
    cam: CameraStream,
    initial_id: int,
    initial_rgb: np.ndarray,
    detector: YuNetDetector,
    evaluator: FramingEvaluator,
    count: int = 6,
) -> Tuple[int, BurstFrame]:
    candidates: List[BurstFrame] = []
    last_id = initial_id
    for idx in range(count):
        if idx == 0:
            rgb = initial_rgb
            ts = time.monotonic()
        else:
            pkt = cam.get_latest(last_id=last_id, timeout=0.6)
            if pkt is None or pkt[0] == last_id:
                break
            last_id, ts, rgb = pkt
        dets = detector.detect(rgb)
        primary = dets[0] if dets else None
        qual = analyze_quality(rgb, face_box=_face_box(primary))
        framing = evaluator.evaluate(primary, ts, qual)
        pose_ok = 1.0
        if primary is not None:
            pose_ok = max(0.0, 1.0 - abs(primary.roll_deg) / 45.0 - abs(primary.yaw_ratio - 0.5) * 0.8)
        candidates.append(
            BurstFrame(
                rgb=rgb,
                sharpness=qual["sharpness"],
                framing_score=framing["framing_score"],
                pose_score=pose_ok,
                lighting_score=_lighting_score(qual),
                face_detected=primary is not None,
            )
        )
    return select_best_burst_frame(candidates)


def _collect_burst_general(
    cam: CameraStream,
    initial_id: int,
    initial_rgb: np.ndarray,
    count: int = 5,
) -> Tuple[int, BurstFrame]:
    candidates: List[BurstFrame] = []
    last_id = initial_id
    for idx in range(count):
        if idx == 0:
            rgb = initial_rgb
        else:
            pkt = cam.get_latest(last_id=last_id, timeout=0.6)
            if pkt is None or pkt[0] == last_id:
                break
            last_id, _, rgb = pkt
        qual = analyze_quality(rgb, face_box=None)
        candidates.append(
            BurstFrame(
                rgb=rgb,
                sharpness=qual["sharpness"],
                framing_score=1.0,
                pose_score=1.0,
                lighting_score=_lighting_score(qual),
                face_detected=False,
            )
        )
    return select_best_burst_frame(candidates)


def capture_photo(
    mode: str = "general",
    question: str = "",
    guided: bool = True,
    stability_mode: str = "normal",
    hold_seconds: float = 2.0,
    timeout_seconds: float = 45.0,
    speak_guidance: bool = True,
    audio_cues: bool = True,
    mirror_horizontal: bool = False,
    camera_name: str = "",
    album: str = "",
    save_folder: str = "",
) -> Tuple[Dict[str, Any], bytes]:
    """Tira uma foto acessível com guia de voz e coloca em área temporária até confirmar que não é imprópria."""
    content_safety.assert_safe_text(question, album, save_folder)
    cap_mode = resolve_capture_mode(mode)
    stab = resolve_stability_mode(stability_mode)
    timeout_s = max(5.0, min(float(timeout_seconds or 45.0), 120.0))
    hold_s = max(0.6, min(float(hold_seconds or 2.0), 6.0))

    detector: Optional[YuNetDetector] = None
    if cap_mode == "portrait":
        ensure_yunet_model()
        detector = YuNetDetector()

    selected_rgb: Optional[np.ndarray] = None
    selected_qual: Dict[str, Any] = {}
    best_seen: Optional[BurstFrame] = None
    best_seen_score: float = -1.0
    timed_out = False
    cam_used = ""
    guidance_log: List[str] = []

    with (
        CameraStream(camera_name=camera_name) as cam,
        VoiceCoach(enabled=speak_guidance and guided, audio_cues=audio_cues and guided) as coach,
    ):
        cam_used = cam.camera_name
        first_pkt = cam.wait_ready(warmup_frames=6, timeout=6.0)
        if first_pkt is None:
            raise RuntimeError("A câmera abriu, mas não enviou nenhum quadro.")

        last_id, start_ts, curr_rgb = first_pkt

        if not guided:
            if cap_mode == "portrait" and detector is not None:
                evaluator = FramingEvaluator(stability_mode=stab, mirror_horizontal=mirror_horizontal)
                _, chosen = _collect_burst_portrait(cam, last_id, curr_rgb, detector, evaluator, count=3)
                selected_rgb = chosen.rgb
                dets = detector.detect(selected_rgb)
                selected_qual = analyze_quality(selected_rgb, face_box=_face_box(dets[0] if dets else None))
            else:
                _, chosen = _collect_burst_general(cam, last_id, curr_rgb, count=3)
                selected_rgb = chosen.rgb
                selected_qual = analyze_quality(selected_rgb, face_box=None)

        elif cap_mode == "portrait" and detector is not None:
            evaluator = FramingEvaluator(stability_mode=stab, mirror_horizontal=mirror_horizontal)
            stillness = StillnessTracker(stability_mode=stab, hold_seconds=hold_s)
            deadline = start_ts + timeout_s
            prev_state = ""

            while True:
                now_ts = time.monotonic()
                dets = detector.detect(curr_rgb)
                primary = dets[0] if dets else None
                qual = analyze_quality(curr_rgb, face_box=_face_box(primary))
                framing = evaluator.evaluate(primary, now_ts, qual)
                track = stillness.update(primary, framing, now_ts)

                cand_score = (
                    (0.5 if primary is not None else 0.0)
                    + framing["framing_score"] * 0.4
                    + min(qual["sharpness"] / 200.0, 0.2)
                )
                if cand_score > best_seen_score:
                    best_seen_score = cand_score
                    best_seen = BurstFrame(
                        rgb=curr_rgb,
                        sharpness=qual["sharpness"],
                        framing_score=framing["framing_score"],
                        lighting_score=_lighting_score(qual),
                        face_detected=primary is not None,
                    )

                cue = track.get("spoken_cue", "")
                sound = track.get("sound_cue", "")
                state = track.get("state", "")
                if cue and (not guidance_log or guidance_log[-1] != cue):
                    guidance_log.append(cue)

                if state == "CAPTURE_NOW":
                    coach.play_cue("shutter")
                    coach.say("Foto tirada!", priority="high", min_interval=0.0)
                    _, chosen = _collect_burst_portrait(cam, last_id, curr_rgb, detector, evaluator, count=6)
                    selected_rgb = chosen.rgb
                    dets_final = detector.detect(selected_rgb)
                    selected_qual = analyze_quality(selected_rgb, face_box=_face_box(dets_final[0] if dets_final else None))
                    break

                if sound:
                    coach.play_cue(sound)
                if cue:
                    prio = "high" if (state in ("STEADY_HOLD", "COUNTDOWN") or prev_state != state) else "normal"
                    coach.say(cue, priority=prio, min_interval=1.55)
                prev_state = state

                if now_ts >= deadline:
                    timed_out = True
                    coach.play_cue("shutter")
                    coach.say("Tempo esgotado, salvando o melhor quadro que consegui.", priority="high", min_interval=0.0)
                    selected_rgb = best_seen.rgb if best_seen is not None else curr_rgb
                    dets_final = detector.detect(selected_rgb)
                    selected_qual = analyze_quality(selected_rgb, face_box=_face_box(dets_final[0] if dets_final else None))
                    break

                pkt = cam.get_latest(last_id=last_id, timeout=0.5)
                if pkt is not None and pkt[0] > last_id:
                    last_id, _, curr_rgb = pkt
                else:
                    time.sleep(0.03)

            coach.close(wait=True, drain=True, timeout=1.6)

        else:
            tracker = ObjectStabilityTracker(stability_mode=stab, hold_seconds=min(hold_s, 1.2))
            deadline = start_ts + min(timeout_s, 20.0)
            prev_rgb: Optional[np.ndarray] = None
            if cap_mode == "money":
                coach.say("Mostre a nota ou moeda na frente da câmera.", priority="high", min_interval=0.0)
            elif cap_mode == "text":
                coach.say("Aponte o texto ou rótulo para a câmera e segure firme.", priority="high", min_interval=0.0)
            elif cap_mode == "object":
                coach.say("Mostre o objeto para a câmera e segure firme.", priority="high", min_interval=0.0)

            while True:
                now_ts = time.monotonic()
                qual = analyze_quality(curr_rgb, face_box=None)
                motion = frame_motion_diff(prev_rgb, curr_rgb) if prev_rgb is not None else 0.0
                prev_rgb = curr_rgb
                track = tracker.update(motion, qual, now_ts)

                cand_score = qual["sharpness"] + (_lighting_score(qual) * 20.0)
                if cand_score > best_seen_score:
                    best_seen_score = cand_score
                    best_seen = BurstFrame(rgb=curr_rgb, sharpness=qual["sharpness"], lighting_score=_lighting_score(qual))

                if track["state"] == "CAPTURE_NOW":
                    coach.play_cue("shutter")
                    coach.say("Foto tirada!", priority="high", min_interval=0.0)
                    _, chosen = _collect_burst_general(cam, last_id, curr_rgb, count=5)
                    selected_rgb = chosen.rgb
                    selected_qual = analyze_quality(selected_rgb, face_box=None)
                    break

                if track.get("spoken_cue"):
                    coach.say(track["spoken_cue"], priority="normal", min_interval=1.8)
                    if not guidance_log or guidance_log[-1] != track["spoken_cue"]:
                        guidance_log.append(track["spoken_cue"])

                if now_ts >= deadline:
                    timed_out = True
                    coach.play_cue("shutter")
                    coach.say("Salvando o melhor quadro que consegui.", priority="high", min_interval=0.0)
                    selected_rgb = best_seen.rgb if best_seen is not None else curr_rgb
                    selected_qual = analyze_quality(selected_rgb, face_box=None)
                    break

                pkt = cam.get_latest(last_id=last_id, timeout=0.5)
                if pkt is not None and pkt[0] > last_id:
                    last_id, _, curr_rgb = pkt
                else:
                    time.sleep(0.03)

            coach.close(wait=True, drain=True, timeout=1.6)

    if selected_rgb is None:
        raise RuntimeError("Falha ao capturar imagem da câmera.")

    staged = stage_captured_photo(
        rgb=selected_rgb,
        mode=cap_mode,
        album=album,
        save_folder=save_folder,
        quality=selected_qual,
    )
    preview_bytes = _encode_preview_jpeg(selected_rgb, max_side=1568, quality=88)
    note = _capture_mode_note(cap_mode, question, selected_qual, staged)
    if timed_out:
        note = (
            "Aviso: o tempo limite terminou antes da confirmação total de centralização/estabilidade; "
            "confira se a imagem ficou nítida e bem enquadrada antes de confirmar. " + note
        )

    payload: Dict[str, Any] = {
        "ok": True,
        "saved": False,
        "status": "pending_safety_review",
        "safety_review_required": True,
        "pending_id": staged["pending_id"],
        "album": staged["album"],
        "mode": cap_mode,
        "stability_mode": stab,
        "camera": cam_used,
        "guided": bool(guided),
        "timed_out": timed_out,
        "size_spoken": staged["size_spoken"],
        "dimensions": f"{staged['width']}x{staged['height']}",
        "quality": selected_qual,
        "guidance_steps": guidance_log[-5:],
        "vision_instructions": note,
    }
    return payload, preview_bytes


def _extract_video_frame_rgb(video_path: Path) -> np.ndarray:
    """Extrai um quadro representativo de um video via ffmpeg para inspecao visual."""
    with tempfile.TemporaryDirectory(prefix="blindgallery_") as tmp_dir:
        out_jpg = Path(tmp_dir) / "frame.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "1",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(out_jpg),
        ]
        subprocess.run(cmd, capture_output=True, timeout=20, creationflags=_NO_WINDOW)
        if not out_jpg.is_file():
            cmd_fallback = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(out_jpg),
            ]
            subprocess.run(cmd_fallback, capture_output=True, timeout=20, creationflags=_NO_WINDOW)
        if not out_jpg.is_file():
            raise RuntimeError("Não consegui extrair um quadro desse vídeo.")
        with PILImage.open(out_jpg) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _load_image_rgb(image_path: Path) -> np.ndarray:
    with PILImage.open(image_path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def prepare_view_image(
    item_id_or_path: str,
    question: str = "",
    review_flagged: bool = False,
) -> Tuple[Dict[str, Any], bytes]:
    """Carrega uma foto (ou quadro de video) da galeria para o modelo ver, com suporte a conferencia de falso alarme."""
    content_safety.assert_safe_text(question)
    try:
        item = get_item(item_id_or_path)
    except content_safety.FilenameFlaggedError:
        if not review_flagged:
            raise content_safety.FilenameFlaggedError(
                "Esse arquivo ficou em espera porque o nome parece suspeito. Para conferir visualmente se é um "
                "falso alarme e liberar na galeria caso esteja tudo bem, chame blindgallery_view com review_flagged=True."
            )
        p = Path(item_id_or_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Não encontrei o arquivo '{item_id_or_path}'.")
        if check_blocked_file(p) is not None:
            raise content_safety.ContentSafetyError(content_safety.BLOCKED_MESSAGE)
        mtype = _classify_ext(p.suffix)
        if mtype == "audio":
            raise ValueError(
                "Esse item é um áudio, não uma imagem. Use blindaudio ou blindaudio_listen para ouvir ou transcrever."
            )
        if mtype is None:
            raise ValueError(f"Formato de arquivo não suportado pela galeria: {p.suffix or '(sem extensão)'}")
        rgb = _extract_video_frame_rgb(p) if mtype == "video" else _load_image_rgb(p)
        preview_bytes = _encode_preview_jpeg(rgb, max_side=1568, quality=85)
        q_txt = (question or "").strip()
        note = (
            "CONFERENCIA DE FALSO ALARME: este arquivo ficou em espera pelo nome e ainda NAO entrou na galeria. "
            + (f"Pergunta da pessoa: {q_txt}. " if q_txt else "")
            + "Primeiro olhe a imagem de verdade: se houver nudez, sexo explicito ou violencia grafica extrema, "
            + f"chame imediatamente blindgallery_remove(item='{str(p)}', block_unsafe=True, reason='unsafe') "
            + "para bloquear pela impressao digital do conteudo. Se for uma foto ou video comum e seguro, descreva "
            + f"e libere chamando blindgallery_mark_safe(file_path='{str(p)}', safe_verified=True, "
            + "title='Mídia verificada', description=...). Nunca identifique ninguém pelo rosto."
        )
        return {
            "ok": True,
            "status": "pending_visual_review",
            "in_gallery": False,
            "file_path": str(p),
            "media_type": mtype,
            "vision_instructions": note,
        }, preview_bytes

    p = Path(item["file_path"])
    if not p.is_file():
        raise FileNotFoundError(f"O arquivo '{item['filename']}' não está mais na pasta original.")
    if check_blocked_file(p) is not None:
        remove_item(item["id"], block_unsafe=False, reason="blocked_content")
        raise content_safety.ContentSafetyError(content_safety.BLOCKED_MESSAGE)
    if not content_safety.check_filename_safety(p)["safe"] and not is_verified_safe(p):
        remove_item(item["id"], block_unsafe=False, reason="content_changed")
        raise content_safety.FilenameFlaggedError(
            "O conteúdo desse arquivo mudou desde a aprovação anterior e o nome parece suspeito. "
            "Chame blindgallery_view com review_flagged=True para conferir de novo."
        )
    if item["media_type"] == "audio":
        raise ValueError(
            "Esse item é um áudio, não uma imagem. Use blindaudio ou blindaudio_listen para ouvir ou transcrever."
        )
    rgb = _extract_video_frame_rgb(p) if item["media_type"] == "video" else _load_image_rgb(p)

    preview_bytes = _encode_preview_jpeg(rgb, max_side=1568, quality=85)
    q_txt = (question or "").strip()
    note = (
        f"Mídia '{item['filename']}' (id {item['id']}) entregue para você descrever seguindo descrevendo-imagens. "
        + (f"Pergunta da pessoa: {q_txt}. " if q_txt else "")
        + "REGRA DE SEGURANCA: se a imagem contiver nudez, sexo explicito ou violencia grafica extrema, NAO grave "
        + f"nenhuma nota e chame imediatamente blindgallery_remove(item='{item['id']}', block_unsafe=True, reason='unsafe') "
        + "para gravar a impressao digital do conteudo e impedir que volte mesmo com outro nome. "
        + "Se for segura e ainda nao houver descrição salva (sem dado sensível), grave uma descrição curta com "
        + f"blindgallery_save_note(item='{item['id']}', description=...). Nunca identifique ninguém pelo rosto."
    )
    payload = {
        "ok": True,
        "item": item,
        "vision_instructions": note,
    }
    return payload, preview_bytes


__all__ = [
    "VALID_CAPTURE_MODES",
    "capture_photo",
    "check_position",
    "confirm_capture_save",
    "discard_capture",
    "list_cameras",
    "list_gallery",
    "list_gallery_albums",
    "mark_gallery_file_safe",
    "prepare_view_image",
    "remove_gallery_item",
    "resolve_capture_mode",
    "save_gallery_note",
    "scan_gallery",
    "search_gallery",
]

scan_gallery = scan_folders
list_gallery = list_items
search_gallery = search_items
save_gallery_note = save_note
list_gallery_albums = list_albums
confirm_capture_save = confirm_staged_photo
discard_capture = discard_staged_photo
remove_gallery_item = remove_item
mark_gallery_file_safe = mark_file_safe
