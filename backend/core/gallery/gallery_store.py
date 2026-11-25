"""Catalogo acessivel em SQLite e armazenamento seguro de fotos e midias do BlindGallery."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image as PILImage

from blind_mcp import cofre, seguranca

from . import content_safety

BACKEND_DIR = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".flac", ".aac"}
ALL_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS

_FORBIDDEN_WIN_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED_WIN_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


def _db_path() -> Path:
    data_dir = Path(os.environ.get("BLIND_DATA_DIR") or (BACKEND_DIR / "data"))
    return data_dir / "blind_gallery.db"


def _known_folder_path(kind: str) -> Path:
    guid = seguranca._KNOWN.get(kind)
    kf = seguranca._known_folder(guid) if guid else None
    return Path(kf) if kf else (Path.home() / kind)


def default_photos_dir() -> Path:
    """Pasta padrao de fotos do BlindPhoto dentro de Imagens (Pictures)."""
    return _known_folder_path("Pictures") / "BlindGallery"


def default_scan_folders() -> List[Path]:
    """Pastas de midia do usuario (Imagens, Videos, Musicas e Downloads) sem duplicatas."""
    seen: List[Path] = []
    for kind in ("Pictures", "Videos", "Music", "Downloads"):
        p = _known_folder_path(kind)
        if p.exists() and p not in seen:
            seen.append(p)
    return seen


def _fold(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", (text or "").lower()) if unicodedata.category(ch) != "Mn"
    )


def _norm_path(p: str | os.PathLike[str]) -> str:
    s = os.path.expandvars(os.path.expanduser(str(p)))
    try:
        s = os.path.realpath(s)
    except OSError:
        s = os.path.abspath(s)
    return os.path.normcase(os.path.normpath(s))


def _safe_stem(name: str, fallback: str = "foto") -> str:
    limpo = _FORBIDDEN_WIN_RE.sub(" ", name or "")
    limpo = re.sub(r"\s+", " ", limpo).strip(" .")
    if len(limpo) > 80:
        limpo = limpo[:80].rstrip(" .")
    if not limpo or limpo.upper() in _RESERVED_WIN_NAMES:
        return fallback
    return limpo


def _unique_path(folder: Path, stem: str, ext: str = ".jpg") -> Path:
    ext_norm = ext if ext.startswith(".") else f".{ext}"
    candidato = folder / f"{stem}{ext_norm}"
    if not candidato.exists():
        return candidato
    idx = 2
    while True:
        tentativa = folder / f"{stem} ({idx}){ext_norm}"
        if not tentativa.exists():
            return tentativa
        idx += 1


def _spoken_date(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d/%m/%Y às %H:%M")
    except Exception:  # noqa: BLE001
        return iso_str or ""


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} bytes"
    if num_bytes < 1024 * 1024:
        return f"{round(num_bytes / 1024)} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB".replace(".", ",")


def _classify_ext(ext: str) -> Optional[str]:
    e = (ext or "").lower()
    if e in IMAGE_EXTS:
        return "image"
    if e in VIDEO_EXTS:
        return "video"
    if e in AUDIO_EXTS:
        return "audio"
    return None


def _connect() -> sqlite3.Connection:
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS media_items (
            id TEXT PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            norm_path TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            album TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            capture_mode TEXT NOT NULL DEFAULT '',
            quality_json TEXT NOT NULL DEFAULT '{}',
            favorite INTEGER NOT NULL DEFAULT 0,
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_media_type ON media_items(media_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_album ON media_items(album)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS blocked_media (
            norm_path TEXT PRIMARY KEY,
            sha256 TEXT NOT NULL DEFAULT '',
            dhash TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            blocked_at TEXT NOT NULL
        )
        """
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(blocked_media)").fetchall()}
    if "sha256" not in cols:
        conn.execute("ALTER TABLE blocked_media ADD COLUMN sha256 TEXT NOT NULL DEFAULT ''")
    if "dhash" not in cols:
        conn.execute("ALTER TABLE blocked_media ADD COLUMN dhash TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_blocked_sha256 ON blocked_media(sha256)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS verified_safe_media (
            sha256 TEXT PRIMARY KEY,
            dhash TEXT NOT NULL DEFAULT '',
            original_name TEXT NOT NULL DEFAULT '',
            verified_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _staging_dir() -> Path:
    data_dir = Path(os.environ.get("BLIND_DATA_DIR") or (BACKEND_DIR / "data"))
    return data_dir / "gallery_staging"


def _valid_pending_id(pending_id: str) -> str:
    pid = (pending_id or "").strip()
    if not re.fullmatch(r"[a-f0-9]{8,32}", pid):
        raise ValueError("Identificador temporário da captura inválido.")
    return pid


def purge_expired_staging(max_age_seconds: float = 600.0) -> int:
    """Remove capturas temporarias antigas nao confirmadas para nao deixar midia solta."""
    sdir = _staging_dir()
    if not sdir.is_dir():
        return 0
    cutoff = time.time() - max(60.0, float(max_age_seconds))
    removed = 0
    for p in list(sdir.iterdir()):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def check_blocked_file(path: str | os.PathLike[str]) -> Optional[str]:
    """Confere se o arquivo esta bloqueado por caminho, SHA-256 exato ou dHash perceptual de imagem."""
    p = Path(path).expanduser()
    np_str = _norm_path(p)
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM blocked_media WHERE norm_path = ?", (np_str,)).fetchone() is not None:
            return "path"
        if p.is_file():
            sha = content_safety.compute_file_sha256(p)
            if sha and conn.execute("SELECT 1 FROM blocked_media WHERE sha256 = ?", (sha,)).fetchone() is not None:
                return "sha256"
            if _classify_ext(p.suffix) == "image":
                dh = content_safety.compute_image_dhash(p)
                if dh:
                    rows = conn.execute("SELECT dhash FROM blocked_media WHERE dhash != ''").fetchall()
                    for row in rows:
                        if content_safety.dhash_matches(dh, row["dhash"]):
                            return "dhash"
    return None


def check_blocked_rgb(rgb: np.ndarray) -> Optional[str]:
    """Confere se um quadro RGB recem-capturado casa com dHash ja bloqueado."""
    dh = content_safety.compute_image_dhash(rgb)
    if not dh:
        return None
    with _connect() as conn:
        rows = conn.execute("SELECT dhash FROM blocked_media WHERE dhash != ''").fetchall()
    for row in rows:
        if content_safety.dhash_matches(dh, row["dhash"]):
            return "dhash"
    return None


def is_blocked_path(path: str | os.PathLike[str]) -> bool:
    """True se o arquivo esta bloqueado por caminho, SHA-256 ou dHash."""
    return check_blocked_file(path) is not None


def is_verified_safe(path: str | os.PathLike[str]) -> bool:
    """True se o hash exato do arquivo foi conferido e aprovado visualmente como seguro."""
    if check_blocked_file(path) is not None:
        return False
    p = Path(path).expanduser()
    if not p.is_file():
        return False
    sha = content_safety.compute_file_sha256(p)
    if not sha:
        return False
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM verified_safe_media WHERE sha256 = ?", (sha,)).fetchone()
    return row is not None


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    tags: List[str] = []
    quality: Dict[str, Any] = {}
    try:
        tags = json.loads(row["tags_json"] or "[]")
    except Exception:  # noqa: BLE001
        tags = []
    try:
        quality = json.loads(row["quality_json"] or "{}")
    except Exception:  # noqa: BLE001
        quality = {}
    exists = os.path.exists(row["file_path"])
    return {
        "id": row["id"],
        "file_path": row["file_path"],
        "filename": row["filename"],
        "media_type": row["media_type"],
        "source": row["source"],
        "title": row["title"],
        "description": row["description"],
        "has_description": bool((row["description"] or "").strip()),
        "album": row["album"],
        "tags": tags,
        "capture_mode": row["capture_mode"],
        "quality": quality,
        "favorite": bool(row["favorite"]),
        "width": int(row["width"] or 0),
        "height": int(row["height"] or 0),
        "size_bytes": int(row["size_bytes"] or 0),
        "size_spoken": _human_size(int(row["size_bytes"] or 0)),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "date_spoken": _spoken_date(row["created_at"]),
        "file_exists": exists,
    }


def _check_no_sensitive(texts: Iterable[str]) -> None:
    for txt in texts:
        tipos = cofre.detectar(txt or "")
        if tipos:
            raise ValueError(
                f"Detectei dado sensível ({', '.join(tipos)}) na descrição ou título. "
                "Por segurança, não gravo senhas, chaves, cartões, CPF, CNPJ ou boletos na galeria. "
                "Para dados sensíveis lidos da foto, use falar_em_segredo ou cofre_guardar."
            )


def _probe_image_dims(path: Path) -> Tuple[int, int]:
    try:
        with PILImage.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:  # noqa: BLE001
        return 0, 0


def _parse_tags(tags: Any) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, str):
        raw = [t.strip() for t in re.split(r"[,;\n]+", tags) if t.strip()]
    elif isinstance(tags, (list, tuple)):
        raw = [str(t).strip() for t in tags if str(t).strip()]
    else:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for t in raw:
        key = _fold(t)
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def stage_captured_photo(
    rgb: np.ndarray,
    mode: str = "general",
    album: str = "",
    save_folder: str = "",
    title: str = "",
    quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Grava a captura apenas em area temporaria aguardando conferencia contra sexo e violencia."""
    purge_expired_staging()
    content_safety.assert_safe_text(title, album, save_folder)
    if check_blocked_rgb(rgb) is not None:
        raise content_safety.ContentSafetyError(content_safety.BLOCKED_MESSAGE)
    if (save_folder or "").strip():
        probe_dir = Path(save_folder).expanduser()
        seguranca.conferir_gravacao_blind_docs(_unique_path(probe_dir, "blindphoto_probe", ".jpg"))
    if title or album:
        _check_no_sensitive([title, album])

    sdir = _staging_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    pending_id = uuid.uuid4().hex[:16]
    jpg_path = sdir / f"{pending_id}.jpg"
    meta_path = sdir / f"{pending_id}.json"

    arr = np.asarray(rgb, dtype=np.uint8)
    pil = PILImage.fromarray(arr[:, :, :3])
    pil.save(str(jpg_path), format="JPEG", quality=92)
    w, h = pil.size
    size_bytes = int(jpg_path.stat().st_size)

    meta = {
        "pending_id": pending_id,
        "mode": (mode or "general").strip(),
        "album": (album or "").strip(),
        "save_folder": (save_folder or "").strip(),
        "title": (title or "").strip(),
        "quality": quality or {},
        "width": w,
        "height": h,
        "size_bytes": size_bytes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return {
        "pending_id": pending_id,
        "mode": meta["mode"],
        "album": meta["album"],
        "save_folder": meta["save_folder"],
        "width": w,
        "height": h,
        "size_bytes": size_bytes,
        "size_spoken": _human_size(size_bytes),
        "expires_in_seconds": 600,
    }


def confirm_staged_photo(
    pending_id: str,
    safe_verified: bool = False,
    title: str = "",
    description: str = "",
    album: str = "",
    tags: Any = None,
    favorite: bool = False,
    save_folder: str = "",
) -> Dict[str, Any]:
    """Move captura temporaria para a pasta definitiva e galeria somente apos confirmar que e segura."""
    if not safe_verified:
        raise content_safety.ContentSafetyError(
            "A foto só pode ser salva na galeria após confirmar que não contém conteúdo impróprio de sexo ou violência."
        )
    pid = _valid_pending_id(pending_id)
    sdir = _staging_dir()
    jpg_path = sdir / f"{pid}.jpg"
    meta_path = sdir / f"{pid}.json"
    if not jpg_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("Captura temporária não encontrada ou expirada. Tire outra foto se precisar.")
    if check_blocked_file(jpg_path) is not None:
        discard_staged_photo(pid, reason="unsafe_blocked")
        raise content_safety.ContentSafetyError(content_safety.BLOCKED_MESSAGE)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mode = (meta.get("mode") or "general").strip()
    eff_title = (title or "").strip() or (meta.get("title") or "").strip()
    eff_desc = (description or "").strip()
    eff_album = (album or "").strip() or (meta.get("album") or "").strip()
    eff_tags = _parse_tags(tags)
    eff_folder_str = (save_folder or "").strip() or (meta.get("save_folder") or "").strip()

    content_safety.assert_safe_text(eff_title, eff_desc, eff_album, eff_folder_str, " ".join(eff_tags))
    _check_no_sensitive([eff_title, eff_desc, eff_album, " ".join(eff_tags)])

    folder = Path(eff_folder_str).expanduser() if eff_folder_str else default_photos_dir()
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    prefix = _safe_stem(f"blindphoto_{mode}_{stamp}", f"blindphoto_{stamp}")
    target = _unique_path(folder, prefix, ".jpg")
    seguranca.conferir_gravacao_blind_docs(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(jpg_path), str(target))
    try:
        meta_path.unlink()
    except OSError:
        pass

    size_bytes = int(target.stat().st_size)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    item_id = uuid.uuid4().hex[:12]

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO media_items (
                id, file_path, norm_path, filename, media_type, source, title, description,
                album, tags_json, capture_mode, quality_json, favorite, width, height,
                size_bytes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'image', 'blindphoto', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                str(target),
                _norm_path(target),
                target.name,
                eff_title,
                eff_desc,
                eff_album,
                json.dumps(eff_tags, ensure_ascii=False),
                mode,
                json.dumps(meta.get("quality") or {}, ensure_ascii=False),
                int(bool(favorite)),
                int(meta.get("width") or 0),
                int(meta.get("height") or 0),
                size_bytes,
                now_iso,
                now_iso,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM media_items WHERE id = ?", (item_id,)).fetchone()
    return _row_to_dict(row)


def discard_staged_photo(pending_id: str, reason: str = "discarded") -> Dict[str, Any]:
    """Apaga imediatamente a foto temporaria e, se for impropria, grava a impressao digital no bloqueio."""
    pid = _valid_pending_id(pending_id)
    sdir = _staging_dir()
    jpg_path = sdir / f"{pid}.jpg"
    clean_reason = (reason or "discarded").strip()
    low_reason = clean_reason.lower()
    is_unsafe = any(k in low_reason for k in ("unsafe", "impropr", "sexo", "violen"))
    fp_blocked = False
    if is_unsafe and jpg_path.is_file():
        fp = content_safety.fingerprint_media(jpg_path, is_image=True)
        if fp["sha256"] or fp["dhash"]:
            with _connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO blocked_media (norm_path, sha256, dhash, reason, blocked_at) VALUES (?, ?, ?, ?, ?)",
                    (f"staged:{pid}", fp["sha256"], fp["dhash"], clean_reason, time.strftime("%Y-%m-%dT%H:%M:%S")),
                )
                conn.commit()
            fp_blocked = True

    removed_files = 0
    for suffix in (".jpg", ".json"):
        p = sdir / f"{pid}{suffix}"
        if p.exists():
            try:
                p.unlink()
                removed_files += 1
            except OSError:
                pass
    return {
        "ok": True,
        "discarded": True,
        "pending_id": pid,
        "reason": clean_reason,
        "fingerprint_blocked": fp_blocked,
        "removed_files": removed_files,
    }


def save_captured_photo(
    rgb: np.ndarray,
    mode: str = "general",
    album: str = "",
    save_folder: str = "",
    title: str = "",
    quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Salva uma foto capturada pelo BlindPhoto sem sobrescrever nada e registra na galeria."""
    content_safety.assert_safe_text(title, album, save_folder)
    folder = Path(save_folder).expanduser() if (save_folder or "").strip() else default_photos_dir()
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    prefix = _safe_stem(f"blindphoto_{mode}_{stamp}", f"blindphoto_{stamp}")
    target = _unique_path(folder, prefix, ".jpg")
    seguranca.conferir_gravacao_blind_docs(target)

    if title:
        _check_no_sensitive([title, album])

    target.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(rgb, dtype=np.uint8)
    pil = PILImage.fromarray(arr[:, :, :3])
    pil.save(str(target), format="JPEG", quality=92)
    w, h = pil.size
    size_bytes = int(target.stat().st_size)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    item_id = uuid.uuid4().hex[:12]

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO media_items (
                id, file_path, norm_path, filename, media_type, source, title, description,
                album, tags_json, capture_mode, quality_json, favorite, width, height,
                size_bytes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'image', 'blindphoto', ?, '', ?, '[]', ?, ?, 0, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                str(target),
                _norm_path(target),
                target.name,
                (title or "").strip(),
                (album or "").strip(),
                (mode or "general").strip(),
                json.dumps(quality or {}, ensure_ascii=False),
                w,
                h,
                size_bytes,
                now_iso,
                now_iso,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM media_items WHERE id = ?", (item_id,)).fetchone()
    return _row_to_dict(row)


def register_external_file(
    file_path: str | os.PathLike[str],
    source: str = "scan",
    album: str = "",
) -> Tuple[Dict[str, Any], bool]:
    """Registra um arquivo de midia existente se ainda nao estiver na galeria e passar na seguranca."""
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Arquivo de mídia não encontrado: {file_path}")
    if check_blocked_file(p) is not None:
        raise content_safety.ContentSafetyError(content_safety.BLOCKED_MESSAGE)
    fn_check = content_safety.check_filename_safety(p)
    if not fn_check["safe"] and not is_verified_safe(p):
        raise content_safety.FilenameFlaggedError(
            "Esse arquivo ficou em espera porque o nome parece suspeito. Se for um falso alarme, abra pela "
            "galeria com conferência visual para o Blind olhar o conteúdo real e liberar com título limpo."
        )
    if album:
        content_safety.assert_safe_text(album)
    mtype = _classify_ext(p.suffix)
    if mtype is None:
        raise ValueError(f"Formato de arquivo não suportado pela galeria: {p.suffix or '(sem extensão)'}")

    np_str = _norm_path(p)
    with _connect() as conn:
        existing = conn.execute("SELECT * FROM media_items WHERE norm_path = ?", (np_str,)).fetchone()
        if existing is not None:
            return _row_to_dict(existing), False

        st = p.stat()
        w, h = _probe_image_dims(p) if mtype == "image" else (0, 0)
        created_iso = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%dT%H:%M:%S")
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
        item_id = uuid.uuid4().hex[:12]
        eff_album = (album or "").strip() or (p.parent.name if p.parent else "")
        conn.execute(
            """
            INSERT INTO media_items (
                id, file_path, norm_path, filename, media_type, source, title, description,
                album, tags_json, capture_mode, quality_json, favorite, width, height,
                size_bytes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, '', ?, '[]', '', '{}', 0, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                str(p),
                np_str,
                p.name,
                mtype,
                source,
                p.stem.replace("_", " ").replace("-", " "),
                eff_album,
                w,
                h,
                int(st.st_size),
                created_iso,
                now_iso,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM media_items WHERE id = ?", (item_id,)).fetchone()
    return _row_to_dict(row), True


def get_item(item_id_or_path: str) -> Dict[str, Any]:
    """Localiza um item por id, caminho completo ou nome do arquivo."""
    key = (item_id_or_path or "").strip()
    if not key:
        raise ValueError("Informe o id ou o caminho da mídia na galeria.")

    with _connect() as conn:
        row = conn.execute("SELECT * FROM media_items WHERE id = ?", (key,)).fetchone()
        if row is not None:
            return _row_to_dict(row)

        np_key = _norm_path(key)
        row = conn.execute("SELECT * FROM media_items WHERE norm_path = ?", (np_key,)).fetchone()
        if row is not None:
            return _row_to_dict(row)

        by_name = conn.execute(
            "SELECT * FROM media_items WHERE lower(filename) = lower(?) ORDER BY created_at DESC",
            (Path(key).name,),
        ).fetchall()
        if len(by_name) == 1:
            return _row_to_dict(by_name[0])
        if len(by_name) > 1:
            ids = ", ".join(r["id"] for r in by_name[:5])
            raise ValueError(f"Há mais de um arquivo chamado '{Path(key).name}'. Informe o id: {ids}.")

    candidate = Path(key).expanduser()
    if candidate.is_file():
        item, _ = register_external_file(candidate, source="manual")
        return item

    raise FileNotFoundError(f"Não encontrei '{key}' na galeria.")


def list_items(
    media_type: str = "",
    album: str = "",
    favorites_only: bool = False,
    undescribed_only: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """Lista midias da galeria ordenadas da mais recente para a mais antiga."""
    lim = max(1, min(int(limit or 20), 100))
    off = max(0, int(offset or 0))
    mt = (media_type or "").strip().lower()
    if mt in ("all", "todos", "todas", ""):
        mt = ""
    elif mt in ("foto", "fotos", "imagem", "imagens", "image"):
        mt = "image"
    elif mt in ("video", "videos", "vídeo", "vídeos"):
        mt = "video"
    elif mt in ("audio", "audios", "áudio", "áudios"):
        mt = "audio"

    clauses: List[str] = []
    params: List[Any] = []
    if mt:
        clauses.append("media_type = ?")
        params.append(mt)
    if favorites_only:
        clauses.append("favorite = 1")
    if undescribed_only:
        clauses.append("trim(description) = ''")

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM media_items {where} ORDER BY created_at DESC, updated_at DESC",
            params,
        ).fetchall()

    items = [_row_to_dict(r) for r in rows]
    if (album or "").strip():
        want_album = _fold(album)
        items = [it for it in items if want_album in _fold(it["album"])]

    total = len(items)
    sliced = items[off : off + lim]
    return {
        "ok": True,
        "total": total,
        "count": len(sliced),
        "offset": off,
        "items": sliced,
    }


def search_items(
    query: str,
    media_type: str = "",
    album: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    """Busca midias por descricao, titulo, tags, album ou nome de arquivo (sem diferenciar acentos)."""
    q = (query or "").strip()
    if not q:
        raise ValueError("Informe o que procurar na galeria.")
    content_safety.assert_safe_text(q, album)
    _check_no_sensitive([q])

    with _connect() as conn:
        all_rows = conn.execute("SELECT * FROM media_items ORDER BY created_at DESC").fetchall()

    mt_filter = ""
    raw_mt = (media_type or "").strip().lower()
    if raw_mt in ("image", "imagem", "imagens", "foto", "fotos"):
        mt_filter = "image"
    elif raw_mt in ("video", "videos", "vídeo", "vídeos"):
        mt_filter = "video"
    elif raw_mt in ("audio", "audios", "áudio", "áudios"):
        mt_filter = "audio"

    album_fold = _fold(album)
    tokens = [t for t in re.findall(r"[a-z0-9]+", _fold(q)) if t]
    if not tokens:
        return {"ok": True, "query": q, "count": 0, "items": []}

    filtered_total = 0
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for row in all_rows:
        item = _row_to_dict(row)
        if mt_filter and item["media_type"] != mt_filter:
            continue
        if album_fold and album_fold not in _fold(item["album"]):
            continue
        filtered_total += 1
        hay_title = _fold(item["title"])
        hay_desc = _fold(item["description"])
        hay_tags = _fold(" ".join(item["tags"]))
        hay_album = _fold(item["album"])
        hay_file = _fold(item["filename"])
        hay_mode = _fold(item["capture_mode"])
        full = f"{hay_title} {hay_desc} {hay_tags} {hay_album} {hay_file} {hay_mode}"
        if all(tok in full for tok in tokens):
            score = 0
            for tok in tokens:
                if tok in hay_title:
                    score += 4
                if tok in hay_desc:
                    score += 5
                if tok in hay_tags:
                    score += 4
                if tok in hay_album:
                    score += 2
                if tok in hay_file:
                    score += 1
            scored.append((score, item["created_at"], item))

    scored.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    lim = max(1, min(int(limit or 20), 50))
    matches = [pair[2] for pair in scored[:lim]]
    return {
        "ok": True,
        "query": q,
        "count": len(matches),
        "total_matched": len(scored),
        "items": matches,
        "searched_total": filtered_total,
    }


def save_note(
    item_id_or_path: str,
    title: Optional[str] = None,
    description: Optional[str] = None,
    album: Optional[str] = None,
    tags: Any = None,
    favorite: Optional[bool] = None,
    allow_sensitive: bool = False,
) -> Dict[str, Any]:
    """Grava ou atualiza titulo, descricao acessivel, album, tags ou favorito na galeria."""
    item = get_item(item_id_or_path)
    new_title = item["title"] if title is None or title == "" else str(title).strip()
    new_desc = item["description"] if description is None or description == "" else str(description).strip()
    new_album = item["album"] if album is None or album == "" else str(album).strip()
    new_tags = item["tags"] if tags is None or tags == "" else _parse_tags(tags)
    new_fav = item["favorite"] if favorite is None else bool(favorite)

    if not allow_sensitive:
        _check_no_sensitive([new_title, new_desc, new_album, " ".join(new_tags)])
    content_safety.assert_safe_text(new_title, new_desc, new_album, " ".join(new_tags))

    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _connect() as conn:
        conn.execute(
            """
            UPDATE media_items
            SET title = ?, description = ?, album = ?, tags_json = ?, favorite = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                new_title,
                new_desc,
                new_album,
                json.dumps(new_tags, ensure_ascii=False),
                int(new_fav),
                now_iso,
                item["id"],
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM media_items WHERE id = ?", (item["id"],)).fetchone()
    return _row_to_dict(row)


def list_albums() -> Dict[str, Any]:
    """Devolve resumo dos albuns e contagens de fotos, videos, audios e itens descritos."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM media_items").fetchall()

    by_type = {"image": 0, "video": 0, "audio": 0}
    described = 0
    undescribed = 0
    favorites = 0
    albums: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        mt = row["media_type"]
        if mt in by_type:
            by_type[mt] += 1
        has_desc = bool((row["description"] or "").strip())
        if has_desc:
            described += 1
        else:
            undescribed += 1
        if row["favorite"]:
            favorites += 1
        alb = (row["album"] or "").strip() or "Sem álbum"
        entry = albums.setdefault(alb, {"album": alb, "total": 0, "described": 0, "undescribed": 0})
        entry["total"] += 1
        if has_desc:
            entry["described"] += 1
        else:
            entry["undescribed"] += 1

    album_list = sorted(albums.values(), key=lambda a: (-a["total"], a["album"].lower()))
    return {
        "ok": True,
        "total_items": len(rows),
        "by_type": by_type,
        "described": described,
        "undescribed": undescribed,
        "favorites": favorites,
        "albums": album_list,
    }


def mark_file_safe(
    file_path: str,
    safe_verified: bool = False,
    title: str = "",
    description: str = "",
    album: str = "",
    tags: Any = None,
    favorite: bool = False,
) -> Dict[str, Any]:
    """Aprova um arquivo exato apos conferencia visual quando o nome parecia suspeito por engano."""
    if not safe_verified:
        raise content_safety.ContentSafetyError(
            "Só é possível liberar esse arquivo após conferir visualmente que não há conteúdo de sexo nem violência."
        )
    p = Path(file_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Arquivo de mídia não encontrado: {file_path}")
    mtype = _classify_ext(p.suffix)
    if mtype is None:
        raise ValueError(f"Formato de arquivo não suportado pela galeria: {p.suffix or '(sem extensão)'}")
    if check_blocked_file(p) is not None:
        raise content_safety.ContentSafetyError(content_safety.BLOCKED_MESSAGE)

    eff_title = (title or "").strip()
    if not eff_title and not content_safety.check_filename_safety(p)["safe"]:
        eff_title = "Mídia verificada"
    eff_desc = (description or "").strip()
    eff_album = (album or "").strip()
    eff_tags = _parse_tags(tags)
    content_safety.assert_safe_text(eff_title, eff_desc, eff_album, " ".join(eff_tags))
    _check_no_sensitive([eff_title, eff_desc, eff_album, " ".join(eff_tags)])

    fp = content_safety.fingerprint_media(p, is_image=(mtype == "image"))
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    if fp["sha256"]:
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO verified_safe_media (sha256, dhash, original_name, verified_at) VALUES (?, ?, ?, ?)",
                (fp["sha256"], fp["dhash"], p.name, now_iso),
            )
            conn.commit()

    item, _ = register_external_file(p, source="verified_safe", album=eff_album)
    updated = save_note(
        item["id"],
        title=eff_title or item["title"],
        description=eff_desc if eff_desc else None,
        album=eff_album if eff_album else None,
        tags=eff_tags if eff_tags else None,
        favorite=bool(favorite),
    )
    return {"ok": True, "verified_safe": True, "item": updated}


def remove_item(
    item_id_or_path: str,
    block_unsafe: bool = False,
    reason: str = "removed",
) -> Dict[str, Any]:
    """Remove uma midia do catalogo da galeria (sem apagar arquivo do disco) e grava impressao digital se bloqueada."""
    key = (item_id_or_path or "").strip()
    if not key:
        raise ValueError("Informe o id ou o caminho da mídia a remover.")
    item_id = ""
    file_path = ""
    filename = ""
    np_key = ""
    fp: Dict[str, str] = {"sha256": "", "dhash": ""}

    with _connect() as conn:
        row = conn.execute("SELECT * FROM media_items WHERE id = ?", (key,)).fetchone()
        if row is None:
            row = conn.execute("SELECT * FROM media_items WHERE norm_path = ?", (_norm_path(key),)).fetchone()
        if row is not None:
            item_id = row["id"]
            file_path = row["file_path"]
            filename = row["filename"]
            np_key = row["norm_path"]
            conn.execute("DELETE FROM media_items WHERE id = ?", (item_id,))
        else:
            p = Path(key).expanduser()
            if p.exists():
                file_path = str(p.resolve())
                filename = p.name
                np_key = _norm_path(p)
            else:
                raise FileNotFoundError(f"Não encontrei '{key}' na galeria.")

        if block_unsafe and np_key:
            is_img = bool(file_path and _classify_ext(Path(file_path).suffix) == "image")
            if file_path and os.path.isfile(file_path):
                fp = content_safety.fingerprint_media(file_path, is_image=is_img)
            conn.execute(
                "INSERT OR REPLACE INTO blocked_media (norm_path, sha256, dhash, reason, blocked_at) VALUES (?, ?, ?, ?, ?)",
                (np_key, fp["sha256"], fp["dhash"], (reason or "unsafe").strip(), time.strftime("%Y-%m-%dT%H:%M:%S")),
            )
            if fp["sha256"]:
                conn.execute("DELETE FROM verified_safe_media WHERE sha256 = ?", (fp["sha256"],))
        conn.commit()

    return {
        "ok": True,
        "removed": True,
        "id": item_id,
        "file_path": file_path,
        "filename": filename,
        "blocked": bool(block_unsafe),
        "fingerprint_blocked": bool(fp["sha256"] or fp["dhash"]),
        "reason": (reason or "removed").strip(),
    }


def scan_folders(
    folders: Optional[List[str]] = None,
    media_type: str = "all",
    max_files: int = 300,
) -> Dict[str, Any]:
    """Indexa fotos, videos e audios existentes nas pastas de midia preservando descricoes e pulando itens improprios."""
    if folders:
        content_safety.assert_safe_text(*folders)
    lim = max(1, min(int(max_files or 300), 1500))
    mt_req = (media_type or "all").strip().lower()
    if mt_req in ("image", "imagem", "imagens", "foto", "fotos"):
        allowed_exts = IMAGE_EXTS
    elif mt_req in ("video", "videos", "vídeo", "vídeos"):
        allowed_exts = VIDEO_EXTS
    elif mt_req in ("audio", "audios", "áudio", "áudios"):
        allowed_exts = AUDIO_EXTS
    else:
        allowed_exts = ALL_MEDIA_EXTS

    if folders:
        roots = [Path(f).expanduser() for f in folders if str(f).strip()]
    else:
        roots = default_scan_folders()

    found_files: List[Path] = []
    scanned_roots: List[str] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        scanned_roots.append(str(root))
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                p = Path(dirpath) / fn
                if p.suffix.lower() in allowed_exts:
                    found_files.append(p)
                    if len(found_files) >= lim:
                        break
            if len(found_files) >= lim:
                break
        if len(found_files) >= lim:
            break

    added = 0
    already = 0
    skipped_blocked = 0
    skipped_suspicious_name = 0
    flagged_files: List[Dict[str, str]] = []
    for p in found_files:
        try:
            _, created = register_external_file(p, source="scan")
            if created:
                added += 1
            else:
                already += 1
        except content_safety.FilenameFlaggedError:
            skipped_suspicious_name += 1
            if len(flagged_files) < 10:
                flagged_files.append({"filename": p.name, "file_path": str(p)})
        except content_safety.ContentSafetyError:
            skipped_blocked += 1
        except Exception:  # noqa: BLE001
            continue

    summary = list_albums()
    return {
        "ok": True,
        "scanned_folders": scanned_roots,
        "found_files": len(found_files),
        "added": added,
        "already_indexed": already,
        "skipped_blocked": skipped_blocked,
        "skipped_suspicious_name": skipped_suspicious_name,
        "skipped_unsafe": skipped_blocked + skipped_suspicious_name,
        "flagged_files": flagged_files,
        "total_in_gallery": summary["total_items"],
        "undescribed": summary["undescribed"],
    }
