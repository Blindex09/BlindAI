"""Testes do BlindPhoto e BlindGallery: deteccao facial YuNet, histerese, estabilidade, qualidade, galeria SQLite e MCP."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from blind_mcp import agent_loop, seguranca
from blind_mcp.servers import _audit, blind, gallery
from core.gallery import blindgallery, content_safety, face_engine, framing_coach, gallery_store, image_quality


def _make_face(
    cx: float = 0.50,
    cy: float = 0.45,
    w: float = 0.30,
    h: float = 0.38,
    roll_offset: float = 0.0,
    yaw_shift: float = 0.0,
    score: float = 0.96,
) -> face_engine.FaceDetection:
    eye_dx = w * 0.22
    eye_y = cy - h * 0.12
    re = (cx - eye_dx, eye_y - roll_offset)
    le = (cx + eye_dx, eye_y + roll_offset)
    nose = (cx + yaw_shift * (2.0 * eye_dx), cy + h * 0.04)
    rm = (cx - w * 0.16, cy + h * 0.24)
    lm = (cx + w * 0.16, cy + h * 0.24)
    return face_engine.FaceDetection.from_normalized(
        x=cx - w / 2.0,
        y=cy - h / 2.0,
        w=w,
        h=h,
        score=score,
        right_eye=re,
        left_eye=le,
        nose=nose,
        right_mouth=rm,
        left_mouth=lm,
    )


# ----------------------------------------------------------------------------- qualidade de imagem
def test_analyze_quality_detects_dark_bright_backlit_shadow_and_blur():
    dark = np.full((120, 160, 3), 15, dtype=np.uint8)
    assert image_quality.analyze_quality(dark)["lighting_status"] == "too_dark"

    bright = np.full((120, 160, 3), 248, dtype=np.uint8)
    assert image_quality.analyze_quality(bright)["lighting_status"] == "too_bright"

    backlit = np.full((120, 160, 3), 230, dtype=np.uint8)
    backlit[30:90, 50:110, :] = 35
    q_back = image_quality.analyze_quality(backlit, face_box=(50 / 160, 30 / 120, 60 / 160, 60 / 120))
    assert q_back["backlit"] is True and q_back["lighting_status"] == "backlit"

    shadow = np.full((120, 160, 3), 130, dtype=np.uint8)
    shadow[24:96, 40:80, :] = 65
    shadow[24:96, 80:120, :] = 195
    q_side = image_quality.analyze_quality(shadow, face_box=(0.25, 0.20, 0.50, 0.60))
    assert q_side["side_shadow"] is True and q_side["lighting_status"] == "side_shadow"

    smooth = np.full((120, 160, 3), 128, dtype=np.uint8)
    assert image_quality.analyze_quality(smooth)["sharpness_status"] == "blurry"

    checker = np.full((120, 160, 3), 90, dtype=np.uint8)
    checker[::2, ::2] = 200
    assert image_quality.analyze_quality(checker)["sharpness_status"] == "sharp"
    assert image_quality.frame_motion_diff(checker, checker) == 0.0
    assert image_quality.frame_motion_diff(checker, dark) > 5.0


# ----------------------------------------------------------------------------- filtro 1-euro, histerese e estabilidade
def test_one_euro_filter_smooths_jitter_and_tracks_step():
    flt = framing_coach.OneEuroFilter(min_cutoff=1.0, beta=0.05)
    noisy = [0.5 + (0.02 if i % 2 == 0 else -0.02) for i in range(25)]
    out_noisy = [flt.update(v, i * 0.05) for i, v in enumerate(noisy)]
    assert np.var(out_noisy[10:]) < np.var(noisy[10:]) * 0.4

    # Movimento grande real deve acompanhar rapidamente
    jumped = [flt.update(0.82, 1.3 + i * 0.05) for i in range(6)]
    assert jumped[-1] > 0.72


def test_framing_evaluator_directions_hysteresis_and_lighting():
    ev = framing_coach.FramingEvaluator(stability_mode="normal", mirror_horizontal=False)
    ev_mir = framing_coach.FramingEvaluator(stability_mode="normal", mirror_horizontal=True)

    # Sem rosto
    no_f = ev.evaluate(None, 0.0, {"lighting_status": "ok"})
    assert not no_f["face_detected"] and no_f["issue_code"] == "no_face"

    # Rosto à esquerda na câmera sem espelho -> mover para a esquerda
    r_left = ev.evaluate(_make_face(cx=0.22), 0.1, {"lighting_status": "ok"})
    assert r_left["issue_code"] == "move_left" and "esquerda" in r_left["spoken_instruction"]

    # Modo espelhado inverte esquerda/direita
    r_mir = ev_mir.evaluate(_make_face(cx=0.22), 0.1, {"lighting_status": "ok"})
    assert r_mir["issue_code"] == "move_right" and "direita" in r_mir["spoken_instruction"]

    # Corte no topo da cabeça tem prioridade
    ev.reset_filters()
    r_cut = ev.evaluate(_make_face(cx=0.50, cy=0.25, h=0.45), 0.2, {"lighting_status": "ok"})
    assert r_cut["issue_code"] == "cut_top" and "alto da cabeça" in r_cut["spoken_instruction"]

    # Muito perto (sem cortar bordas) / muito longe / cima / baixo / virado / inclinado
    ev.reset_filters()
    assert ev.evaluate(_make_face(cy=0.54, h=0.62), 0.3, {"lighting_status": "ok"})["issue_code"] == "too_close"
    ev.reset_filters()
    assert ev.evaluate(_make_face(h=0.14), 0.4, {"lighting_status": "ok"})["issue_code"] == "too_far"
    ev.reset_filters()
    assert ev.evaluate(_make_face(cy=0.26, h=0.30), 0.5, {"lighting_status": "ok"})["issue_code"] == "move_down"
    ev.reset_filters()
    assert ev.evaluate(_make_face(cy=0.68, h=0.30), 0.6, {"lighting_status": "ok"})["issue_code"] == "move_up"
    ev.reset_filters()
    assert ev.evaluate(_make_face(yaw_shift=0.26), 0.7, {"lighting_status": "ok"})["issue_code"] == "turn_face"
    ev.reset_filters()
    assert ev.evaluate(_make_face(roll_offset=0.045), 0.8, {"lighting_status": "ok"})["issue_code"] == "tilt_head"

    # Histerese: depois de centralizar (<=0.085), pequeno desvio até 0.11 continua centralizado (cx_out=0.13)
    hyst = framing_coach.FramingEvaluator(stability_mode="normal", mirror_horizontal=False)
    for i in range(10):
        res_c = hyst.evaluate(_make_face(cx=0.50), i * 0.08, {"lighting_status": "ok"})
    assert res_c["is_centered"]
    res_hover = hyst.evaluate(_make_face(cx=0.605), 0.9, {"lighting_status": "too_dark"})
    assert res_hover["is_centered"]  # nao oscila na borda
    assert "escuro" in res_hover["spoken_instruction"]
    # Mas segundo aviso seguido nao repete o alerta de luz infinitamente
    res_hover2 = hyst.evaluate(_make_face(cx=0.605), 1.0, {"lighting_status": "too_dark"})
    assert "escuro" not in res_hover2["spoken_instruction"]


def test_stillness_tracker_grace_for_involuntary_twitch_and_drift_reset():
    ev = framing_coach.FramingEvaluator(stability_mode="gentle")
    st = framing_coach.StillnessTracker(stability_mode="gentle", hold_seconds=1.2)

    t = 0.0
    last = {}
    for _ in range(10):
        det = _make_face(cx=0.50, cy=0.45)
        fr = ev.evaluate(det, t, {"lighting_status": "ok"})
        last = st.update(det, fr, t)
        t += 0.08
    assert last["state"] in ("STEADY_HOLD", "COUNTDOWN")
    saved_hold = last["hold_elapsed"]
    assert saved_hold > 0.0

    # Pequeno espasmo involuntario de 2 quadros ainda dentro da zona externa: usa tolerancia (nao zera!)
    for dx in (0.045, -0.040):
        det_spasm = _make_face(cx=0.50 + dx, cy=0.45)
        fr = ev.evaluate(det_spasm, t, {"lighting_status": "ok"})
        res_spasm = st.update(det_spasm, fr, t)
        assert res_spasm["hold_elapsed"] >= saved_hold
        t += 0.08

    # Volta a estabilizar e conclui contagem ate CAPTURE_NOW
    for _ in range(20):
        det = _make_face(cx=0.50, cy=0.45)
        fr = ev.evaluate(det, t, {"lighting_status": "ok"})
        last = st.update(det, fr, t)
        t += 0.08
        if last["state"] == "CAPTURE_NOW":
            break
    assert last["state"] == "CAPTURE_NOW"

    # Desvio sustentado para longe da zona externa reseta com aviso calmo
    ev2 = framing_coach.FramingEvaluator(stability_mode="normal")
    st2 = framing_coach.StillnessTracker(stability_mode="normal", hold_seconds=1.5)
    for i in range(12):
        det = _make_face(cx=0.50, cy=0.45)
        st2.update(det, ev2.evaluate(det, i * 0.08, {"lighting_status": "ok"}), i * 0.08)
    cues_far = []
    for j in range(12):
        det_far = _make_face(cx=0.18, cy=0.45)
        ts = 1.0 + j * 0.08
        res_far = st2.update(det_far, ev2.evaluate(det_far, ts, {"lighting_status": "ok"}), ts)
        cues_far.append(res_far["spoken_cue"])
    assert any("mexeu" in c for c in cues_far)
    assert res_far["state"] == "GUIDING" and res_far["hold_elapsed"] == 0.0


def test_select_best_burst_frame_prefers_sharp_and_centered():
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    f_blurry = framing_coach.BurstFrame(rgb=dummy, sharpness=18.0, framing_score=0.98)
    f_off = framing_coach.BurstFrame(rgb=dummy, sharpness=120.0, framing_score=0.25)
    f_best = framing_coach.BurstFrame(rgb=dummy, sharpness=135.0, framing_score=0.96, pose_score=0.95, lighting_score=1.0)
    idx, chosen = framing_coach.select_best_burst_frame([f_blurry, f_off, f_best])
    assert idx == 2 and chosen is f_best


# ----------------------------------------------------------------------------- decodificacao YuNet + ONNX real
def test_decode_yunet_outputs_math_and_nms():
    pad_w, pad_h = 640, 640
    outs = {}
    for s in (8, 16, 32):
        n = (pad_w // s) * (pad_h // s)
        outs[f"cls_{s}"] = np.zeros((1, n, 1), dtype=np.float32)
        outs[f"obj_{s}"] = np.zeros((1, n, 1), dtype=np.float32)
        outs[f"bbox_{s}"] = np.zeros((1, n, 4), dtype=np.float32)
        outs[f"kps_{s}"] = np.zeros((1, n, 10), dtype=np.float32)

    # Ancora no stride 16: linha r=20, coluna c=20 (centro 320,320 no pad 640x640)
    cols_16 = pad_w // 16
    idx16 = 20 * cols_16 + 20
    outs["cls_16"][0, idx16, 0] = 0.95
    outs["obj_16"][0, idx16, 0] = 0.95
    outs["bbox_16"][0, idx16, :] = [0.0, 0.0, float(np.log(160.0 / 16.0)), float(np.log(200.0 / 16.0))]
    # olhos simetricos, nariz exatamente no meio (yaw=0.5), roll ~ 0
    outs["kps_16"][0, idx16, :] = [-2.0, -2.0, 2.0, -2.0, 0.0, 0.5, -1.5, 3.0, 1.5, 3.0]

    # Ancora sobreposta mais fraca no stride 8 perto do mesmo ponto -> deve ser suprimida pelo NMS
    cols_8 = pad_w // 8
    idx8 = 40 * cols_8 + 40
    outs["cls_8"][0, idx8, 0] = 0.72
    outs["obj_8"][0, idx8, 0] = 0.72
    outs["bbox_8"][0, idx8, :] = [0.0, 0.0, float(np.log(156.0 / 8.0)), float(np.log(196.0 / 8.0))]
    outs["kps_8"][0, idx8, :] = [-4.0, -4.0, 4.0, -4.0, 0.0, 1.0, -3.0, 6.0, 3.0, 6.0]

    dets = face_engine.decode_yunet_outputs(outs, pad_w, pad_h, 640, 640, 1.0, 1.0)
    assert len(dets) == 1
    d = dets[0]
    assert d.score == pytest.approx(0.95, abs=0.01)
    assert d.cx == pytest.approx(0.50, abs=0.01)
    assert d.cy == pytest.approx(0.50, abs=0.01)
    assert d.w == pytest.approx(160.0 / 640.0, abs=0.02)
    assert d.h == pytest.approx(200.0 / 640.0, abs=0.02)
    assert d.yaw_ratio == pytest.approx(0.50, abs=0.02)
    assert abs(d.roll_deg) < 1.0


def test_real_yunet_onnx_loads_and_runs_on_blank_frame():
    model_file = face_engine.yunet_model_path()
    if not model_file.is_file():
        pytest.skip("Modelo YuNet ONNX ainda nao baixado neste ambiente.")
    detector = face_engine.YuNetDetector(model_path=model_file)
    assert detector.detect(np.zeros((360, 640, 3), dtype=np.uint8)) == []


# ----------------------------------------------------------------------------- galeria SQLite e seguranca
def test_gallery_store_capture_scan_search_notes_and_security(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path / "data"))
    pics_dir = tmp_path / "Pictures"
    pics_dir.mkdir()
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(tmp_path))

    # Bloqueio de seguranca fora das pastas permitidas
    monkeypatch.delenv("BLIND_PASTAS_EXTRAS", raising=False)
    outside = Path.home() / "Desktop"
    with pytest.raises(seguranca.BloqueioDeSeguranca):
        gallery_store.save_captured_photo(
            np.full((40, 40, 3), 120, dtype=np.uint8),
            mode="general",
            save_folder=str(outside),
        )

    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(pics_dir))
    rgb = np.full((60, 80, 3), 140, dtype=np.uint8)
    s1 = gallery_store.save_captured_photo(rgb, mode="money", album="Finanças", save_folder=str(pics_dir))
    s2 = gallery_store.save_captured_photo(rgb, mode="money", album="Finanças", save_folder=str(pics_dir))
    assert Path(s1["file_path"]).is_file() and Path(s2["file_path"]).is_file()
    assert s1["file_path"] != s2["file_path"]  # nunca sobrescreve

    # Salva nota com acento e busca sem acento
    updated = gallery_store.save_note(
        s1["id"],
        title="Nota de cinquenta reais",
        description="Cédula de cinquenta reais bem nítida sobre o café da manhã.",
        tags="dinheiro, manhã",
        favorite=True,
    )
    assert updated["has_description"] is True and updated["favorite"] is True

    found = gallery_store.search_items("cafe cinquenta")
    assert found["count"] == 1 and found["items"][0]["id"] == s1["id"]

    # Dado sensivel bloqueado na anotacao da galeria
    with pytest.raises(ValueError, match="dado sensível"):
        gallery_store.save_note(s2["id"], description="CPF 123.456.789-09 do cliente")

    # Scan indexa midias existentes preservando notas ja salvas
    ext_img = pics_dir / "ferias_praia.jpg"
    PILImage.new("RGB", (100, 80), color=(10, 100, 200)).save(ext_img)
    ext_aud = pics_dir / "recado.mp3"
    ext_aud.write_bytes(b"ID3")
    scan = gallery_store.scan_folders([str(pics_dir)])
    assert scan["added"] == 2 and scan["already_indexed"] == 2
    assert gallery_store.get_item(s1["id"])["title"] == "Nota de cinquenta reais"

    alb = gallery_store.list_albums()
    assert alb["total_items"] == 4 and alb["by_type"]["image"] == 3 and alb["by_type"]["audio"] == 1


# ----------------------------------------------------------------------------- trava de seguranca (sexo e violencia)
def test_content_safety_blocks_sexual_and_violence_without_false_positives():
    for bad in (
        "video porno explicito",
        "pornografia amadora",
        "fotos de sexo explicito",
        "cenas nsfw xxx",
        "gore chacina real",
        "decapitacao brutal",
        "violencia extrema sangrenta",
        "explicit sex tape",
        "graphic violence",
    ):
        r = content_safety.check_text_safety(bad)
        assert not r["safe"] and r["categories"], bad

    for safe_txt in (
        "documento com campo sexo masculino e sexo feminino",
        "sexta-feira dia vinte e tres",
        "sexto andar do predio",
        "exame de sangue e hemograma completo",
        "doacao de sangue no hospital",
        "nota de cinquenta reais bem nitida",
        "remedio dipirona 500mg comprimido",
        "faca de cozinha em cima da mesa",
    ):
        assert content_safety.check_text_safety(safe_txt)["safe"], safe_txt

    assert not content_safety.check_filename_safety(r"C:\pasta\video_porno_hd.mp4")["safe"]
    assert not content_safety.check_filename_safety(r"C:\pasta\gore-acidente.jpg")["safe"]
    assert content_safety.check_filename_safety(r"C:\pasta\ficha_sexo_masculino.jpg")["safe"]

    yt_18 = content_safety.check_youtube_metadata({"title": "Aula comum", "age_limit": 18})
    assert not yt_18["safe"] and "adult_18" in yt_18["categories"]
    yt_ok = content_safety.check_youtube_metadata({"title": "Receita de bolo simples", "age_limit": 0})
    assert yt_ok["safe"]


def test_staged_capture_safety_flow_confirm_discard_scan_and_block_remove(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path / "data"))
    pics_dir = tmp_path / "Pictures"
    pics_dir.mkdir()
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(pics_dir))

    rgb = np.full((60, 80, 3), 145, dtype=np.uint8)

    # 1. Stage nao salva na pasta final nem na tabela media_items
    staged = gallery_store.stage_captured_photo(rgb, mode="object", album="Casa", save_folder=str(pics_dir))
    pid = staged["pending_id"]
    assert list(pics_dir.iterdir()) == []
    assert gallery_store.list_items()["total"] == 0

    # Sem safe_verified=True recusa salvar
    with pytest.raises(content_safety.ContentSafetyError):
        asyncio.run(gallery.blindgallery_confirm_save(pid, safe_verified=False, description="Teste"))
    assert list(pics_dir.iterdir()) == []

    # Descricao impropria tambem recusa
    with pytest.raises(content_safety.ContentSafetyError):
        asyncio.run(gallery.blindgallery_confirm_save(pid, safe_verified=True, description="cena de pornografia"))
    assert list(pics_dir.iterdir()) == []

    # Com safe_verified=True e descricao segura salva e apaga staging
    conf = asyncio.run(
        gallery.blindgallery_confirm_save(
            pid,
            safe_verified=True,
            title="Caneca azul",
            description="Caneca azul em cima da mesa",
            tags="casa, caneca",
            favorite=True,
        )
    )
    assert conf["saved"] is True and Path(conf["item"]["file_path"]).parent == pics_dir
    assert not (gallery_store._staging_dir() / f"{pid}.jpg").exists()

    # 2. Discard apaga temporario sem deixar nada definitivo
    rgb_bad_cap = _textured_rgb(seed=55)
    staged2 = gallery_store.stage_captured_photo(rgb_bad_cap, mode="general", save_folder=str(pics_dir))
    pid2 = staged2["pending_id"]
    disc = asyncio.run(gallery.blindgallery_discard_capture(pid2, reason="unsafe"))
    assert disc["discarded"] is True and disc["removed_files"] == 2
    assert not (gallery_store._staging_dir() / f"{pid2}.jpg").exists()
    assert gallery_store.list_items()["total"] == 1

    # 3. Scan pula arquivos improprios e remove(block_unsafe=True) impede reindexar/reabrir
    unsafe1 = pics_dir / "porno_teste.mp4"
    unsafe1.write_bytes(b"fake")
    unsafe2 = pics_dir / "violencia_extrema.jpg"
    PILImage.new("RGB", (50, 50), color=(200, 10, 10)).save(unsafe2)
    normal = pics_dir / "cachorro_quintal.jpg"
    PILImage.new("RGB", (50, 50), color=(40, 180, 40)).save(normal)

    scan = asyncio.run(gallery.blindgallery_scan(folder=str(pics_dir)))
    assert scan["added"] == 1 and scan["skipped_unsafe"] == 2

    rem = asyncio.run(gallery.blindgallery_remove(str(normal), block_unsafe=True, reason="unsafe"))
    assert rem["removed"] is True and rem["blocked"] is True and normal.is_file()  # nao apaga disco
    scan_after = asyncio.run(gallery.blindgallery_scan(folder=str(pics_dir)))
    assert scan_after["added"] == 0 and scan_after["skipped_unsafe"] == 3
    with pytest.raises(content_safety.ContentSafetyError):
        asyncio.run(gallery.blindgallery_view(str(normal)))


def _textured_rgb(seed: int = 1, width: int = 160, height: int = 120) -> np.ndarray:
    """Gera imagem deterministica com formas macro reais para testes estaveis de dHash."""
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    f1 = 2.0 + (seed % 5) * 1.1
    f2 = 1.5 + ((seed // 3) % 4) * 1.3
    phase = (seed % 7) * 0.7
    r = np.clip(125 + 95 * np.sin(xs * f1 + ys * f2 + phase), 0, 255)
    g = np.clip(115 + 85 * np.cos(xs * (f2 + 0.8) - ys * f1 + phase), 0, 255)
    b = np.clip(128 + 90 * np.sin(xs * 2.1 - ys * (f2 + 1.2) - phase), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def test_fingerprint_blocks_renamed_moved_copied_and_resaved_media(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path / "data"))
    pics_dir = tmp_path / "Pictures"
    pics_dir.mkdir()
    other_dir = pics_dir / "OutraPasta"
    other_dir.mkdir()
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(pics_dir))

    bad_arr = _textured_rgb(seed=42)
    distinct_arr = _textured_rgb(seed=99)[::-1, :, :]  # padrao oposto: dHash diferente
    assert content_safety.compute_image_dhash(bad_arr) != ""
    assert not content_safety.dhash_matches(
        content_safety.compute_image_dhash(bad_arr),
        content_safety.compute_image_dhash(distinct_arr),
    )

    bad_orig = pics_dir / "foto_comum_inicial.jpg"
    PILImage.fromarray(bad_arr).save(bad_orig, quality=92)
    assert asyncio.run(gallery.blindgallery_scan(folder=str(pics_dir)))["added"] == 1

    # Bloqueia por conteudo improprio: grava SHA-256 + dHash
    rem = asyncio.run(gallery.blindgallery_remove(str(bad_orig), block_unsafe=True, reason="unsafe"))
    assert rem["blocked"] is True and rem["fingerprint_blocked"] is True

    # 1. Mesmo arquivo copiado para outra pasta com nome bonito -> barrado por SHA-256
    renamed_copy = other_dir / "flor_primavera_bonita.jpg"
    renamed_copy.write_bytes(bad_orig.read_bytes())
    assert gallery_store.check_blocked_file(renamed_copy) in ("sha256", "dhash")

    # 2. Mesma imagem regravada com tamanho e compressao diferentes (outro SHA-256!) -> barrada por dHash
    resaved_copy = other_dir / "paisagem_linda_regravada.jpg"
    PILImage.fromarray(bad_arr).resize((96, 72)).save(resaved_copy, quality=52)
    assert gallery_store.check_blocked_file(resaved_copy) == "dhash"

    # Nem scan, nem view(review_flagged=True), nem mark_safe, nem nova captura da mesma cena passam
    scan = asyncio.run(gallery.blindgallery_scan(folder=str(other_dir)))
    assert scan["added"] == 0 and scan["skipped_blocked"] == 2
    with pytest.raises(content_safety.ContentSafetyError):
        asyncio.run(gallery.blindgallery_view(str(renamed_copy), review_flagged=True))
    with pytest.raises(content_safety.ContentSafetyError):
        asyncio.run(gallery.blindgallery_view(str(resaved_copy), review_flagged=True))
    with pytest.raises(content_safety.ContentSafetyError):
        asyncio.run(gallery.blindgallery_mark_safe(str(resaved_copy), safe_verified=True, title="Tentativa"))
    with pytest.raises(content_safety.ContentSafetyError):
        gallery_store.stage_captured_photo(bad_arr, mode="general", save_folder=str(pics_dir))

    # Imagem realmente diferente entra normalmente
    ok_file = other_dir / "paisagem_diferente.jpg"
    PILImage.fromarray(distinct_arr).save(ok_file, quality=90)
    assert asyncio.run(gallery.blindgallery_scan(folder=str(other_dir)))["added"] == 1

    # Captura descartada como impropria tambem grava impressao digital antes de apagar
    staged_bad = gallery_store.stage_captured_photo(distinct_arr, mode="general", save_folder=str(pics_dir))
    disc = asyncio.run(gallery.blindgallery_discard_capture(staged_bad["pending_id"], reason="unsafe_violence"))
    assert disc["discarded"] is True and disc["fingerprint_blocked"] is True
    with pytest.raises(content_safety.ContentSafetyError):
        gallery_store.stage_captured_photo(distinct_arr, mode="general", save_folder=str(pics_dir))


def test_flagged_filename_review_and_mark_safe_false_alarm(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path / "data"))
    pics_dir = tmp_path / "Pictures"
    pics_dir.mkdir()
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(pics_dir))

    innocent = pics_dir / "pelada_de_domingo.jpg"
    PILImage.fromarray(_textured_rgb(seed=7)).save(innocent, quality=90)

    # Scan coloca arquivo com nome feio em espera (nao entra na galeria ainda)
    scan = asyncio.run(gallery.blindgallery_scan(folder=str(pics_dir)))
    assert scan["added"] == 0 and scan["skipped_suspicious_name"] == 1
    assert scan["flagged_files"][0]["filename"] == "pelada_de_domingo.jpg"
    assert gallery_store.list_items()["total"] == 0

    # View normal recusa avisando que precisa conferencia visual
    with pytest.raises(content_safety.FilenameFlaggedError):
        asyncio.run(gallery.blindgallery_view(str(innocent)))

    # View com review_flagged=True entrega imagem para conferencia sem indexar ainda
    rev = asyncio.run(gallery.blindgallery_view(str(innocent), review_flagged=True))
    assert len(rev) == 2 and rev[1].type == "image"
    rev_meta = json.loads(rev[0].text)
    assert rev_meta["status"] == "pending_visual_review" and rev_meta["in_gallery"] is False
    assert gallery_store.list_items()["total"] == 0

    # Sem safe_verified=True recusa liberar
    with pytest.raises(content_safety.ContentSafetyError):
        asyncio.run(gallery.blindgallery_mark_safe(str(innocent), safe_verified=False))

    # Aprova como falso alarme com titulo limpo
    approved = asyncio.run(
        gallery.blindgallery_mark_safe(
            str(innocent),
            safe_verified=True,
            title="Futebol no domingo",
            description="Foto do campo de futebol com amigos",
            album="Esporte",
            favorite=True,
        )
    )
    assert approved["verified_safe"] is True and approved["item"]["title"] == "Futebol no domingo"
    assert gallery_store.list_items()["total"] == 1

    # Novo scan e view normal funcionam normalmente enquanto for o mesmo arquivo
    assert asyncio.run(gallery.blindgallery_scan(folder=str(pics_dir)))["already_indexed"] == 1
    assert len(asyncio.run(gallery.blindgallery_view(approved["item"]["id"]))) == 2

    # Se trocarem os bytes daquele caminho por outro conteudo depois, perde a aprovacao automaticamente
    PILImage.fromarray(_textured_rgb(seed=303)).save(innocent, quality=90)
    assert not gallery_store.is_verified_safe(innocent)
    with pytest.raises(content_safety.FilenameFlaggedError):
        asyncio.run(gallery.blindgallery_view(approved["item"]["id"]))


# ----------------------------------------------------------------------------- servidor MCP e integracao Blind
def test_mcp_tools_capture_view_and_integrations(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path / "data"))
    pics_dir = tmp_path / "Pictures"
    pics_dir.mkdir()
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(pics_dir))

    frames = iter(
        [
            _textured_rgb(seed=10),
            _textured_rgb(seed=20),
            _textured_rgb(seed=37)[::-1, :, :],
        ]
    )
    monkeypatch.setattr(
        "core.gallery.camera.list_cameras",
        lambda: [{"index": 0, "name": "Câmera Teste", "alternative_name": "", "default": True}],
    )
    monkeypatch.setattr(
        "core.gallery.blindgallery.check_position",
        lambda *a, **k: {"ok": True, "face_detected": True, "ready_to_capture": True, "spoken_guidance": "Rosto bem centralizado."},
    )

    def _fake_capture(*args, **kwargs):
        rgb_frame = next(frames)
        staged = gallery_store.stage_captured_photo(rgb_frame, mode="money", album="Carteira", save_folder=str(pics_dir))
        return (
            {
                "ok": True,
                "saved": False,
                "status": "pending_safety_review",
                "safety_review_required": True,
                "pending_id": staged["pending_id"],
                "album": staged["album"],
                "mode": "money",
                "vision_instructions": "Leia o valor em voz alta.",
            },
            blindgallery._encode_preview_jpeg(rgb_frame),
        )

    monkeypatch.setattr("core.gallery.blindgallery.capture_photo", _fake_capture)

    cams = asyncio.run(gallery.blindphoto_cameras())
    assert cams["count"] == 1 and cams["cameras"][0]["name"] == "Câmera Teste"

    pos = asyncio.run(gallery.blindphoto_check_position(speak=False))
    assert pos["ready_to_capture"] is True

    # Fluxo padrao: entrega imagem + pending_id (sem salvar ainda) e confirma em seguida
    cap = asyncio.run(gallery.blindphoto_capture(mode="money", save_folder=str(pics_dir)))
    assert len(cap) == 2 and cap[0].type == "text" and cap[1].type == "image"
    cap_meta = json.loads(cap[0].text)
    assert cap_meta["ok"] is True and cap_meta["saved"] is False and cap_meta["pending_id"]
    assert gallery_store.list_items()["total"] == 0

    confirmed = asyncio.run(
        gallery.blindgallery_confirm_save(
            cap_meta["pending_id"],
            safe_verified=True,
            title="Nota azul",
            description="Nota dobrada",
            favorite=True,
        )
    )
    item_id = confirmed["item"]["id"]

    viewed = asyncio.run(gallery.blindgallery_view(item_id, question="qual nota e essa?"))
    assert len(viewed) == 2 and viewed[1].type == "image"

    noted = asyncio.run(gallery.blindgallery_save_note(item_id, title="Nota azul atualizada"))
    assert noted["ok"] is True and noted["item"]["title"] == "Nota azul atualizada"

    listed = asyncio.run(gallery.blindgallery_list(favorites_only=True))
    assert listed["total"] == 1

    # Fluxo describe_with_vision=True: UNSAFE descarta e nao devolve imagem; SAFE salva direto
    from core.vision import smart_vision_router as svr_mod

    async def _fake_unsafe(**kwargs):
        return "UNSAFE_SEXUAL conteudo improprio"

    monkeypatch.setattr(svr_mod.smart_vision_router, "describe_multimodal", _fake_unsafe)
    cap_unsafe = asyncio.run(gallery.blindphoto_capture(mode="general", save_folder=str(pics_dir), describe_with_vision=True))
    assert len(cap_unsafe) == 1
    meta_u = json.loads(cap_unsafe[0].text)
    assert meta_u["ok"] is False and meta_u["blocked"] is True and meta_u["saved"] is False
    assert gallery_store.list_items()["total"] == 1

    async def _fake_safe(**kwargs):
        return "SAFE\nNota de dez reais limpa."

    monkeypatch.setattr(svr_mod.smart_vision_router, "describe_multimodal", _fake_safe)
    cap_safe = asyncio.run(gallery.blindphoto_capture(mode="money", save_folder=str(pics_dir), describe_with_vision=True))
    assert len(cap_safe) == 2
    meta_s = json.loads(cap_safe[0].text)
    assert meta_s["ok"] is True and meta_s["saved"] is True and meta_s["auto_description"] == "Nota de dez reais limpa."
    assert gallery_store.list_items()["total"] == 2

    # Integracao no blind comecar, _audit e agent_loop
    r = asyncio.run(blind.comecar("blindphoto tirar selfie e abrir blindgallery"))
    hab_names = {h["nome"] for h in r["habilidades_do_pedido"]}
    assert "usando-blindgallery" in hab_names
    assert set(r["nomes_exatos"]["blind-gallery"]) == {
        "mcp__blind-gallery__blindphoto_cameras",
        "mcp__blind-gallery__blindphoto_check_position",
        "mcp__blind-gallery__blindphoto_capture",
        "mcp__blind-gallery__blindgallery_confirm_save",
        "mcp__blind-gallery__blindgallery_discard_capture",
        "mcp__blind-gallery__blindgallery_mark_safe",
        "mcp__blind-gallery__blindgallery_scan",
        "mcp__blind-gallery__blindgallery_list",
        "mcp__blind-gallery__blindgallery_search",
        "mcp__blind-gallery__blindgallery_view",
        "mcp__blind-gallery__blindgallery_save_note",
        "mcp__blind-gallery__blindgallery_remove",
        "mcp__blind-gallery__blindgallery_albums",
    }
    assert "blindgallery" in r["guia"] and "blindphoto" in r["guia"]

    for tool_name in (
        "blindphoto_cameras",
        "blindphoto_check_position",
        "blindphoto_capture",
        "blindgallery_confirm_save",
        "blindgallery_discard_capture",
        "blindgallery_mark_safe",
        "blindgallery_scan",
        "blindgallery_list",
        "blindgallery_search",
        "blindgallery_view",
        "blindgallery_save_note",
        "blindgallery_remove",
        "blindgallery_albums",
    ):
        h = _audit.humanize({"server": "blind-gallery", "tool": tool_name, "ok": True, "terminou": True, "ms": 20})
        assert h["origem"] == "BlindGallery e BlindPhoto" and h["estado"] == "deu_certo"

    assert "blind-gallery" in agent_loop.select_relevant_servers("tira uma selfie no blindphoto")
    assert "blind-gallery" in agent_loop.select_relevant_servers("procura na galeria uma foto")
    assert agent_loop.wants_tools("abrir blindgallery")
