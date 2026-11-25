"""Filtro defensivo contra midias improprias de sexo explicito, nudez ou violencia extrema."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image as PILImage


class ContentSafetyError(ValueError):
    """Levantada quando uma foto, video, busca ou anotacao viola as regras contra sexo ou violencia."""


class FilenameFlaggedError(ContentSafetyError):
    """Levantada quando apenas o nome do arquivo parece suspeito e aguarda conferencia visual."""


DHASH_MAX_DISTANCE = 6
_PARTIAL_HASH_LIMIT = 256 * 1024 * 1024
_PARTIAL_CHUNK_BYTES = 4 * 1024 * 1024


def _fold(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", (text or "").lower()) if unicodedata.category(ch) != "Mn"
    )


_SEXUAL_RE = re.compile(
    r"(?:"
    r"\b(?:porno\w*|porn\w*|xvideos|xnxx|pornhub|onlyfans|hentai|nsfw|nudes?|pelad[oa]s?|erotic[oa]s?|"
    r"putaria|boquete|masturba\w*|orgasm\w*|fetiche|xxx|pedofil\w*|csam|nudity|naked)\b|"
    r"\bnu[as]?\s+explicit\w*\b|"
    r"\bsexo\s+(?:explicit\w*|oral|anal|selvagem|quente|real|amador|pesad\w*)\b|"
    r"\b(?:foto|video|filme|cena|conteudo|midia)s?\s+de\s+sexo\b|"
    r"\bconteudo\s+adulto\b|"
    r"\bproibido\s+para\s+menores\b|"
    r"\bexplicit\s+sex\b|\bsex\s+tape\b|\bhardcore\s+sex\b"
    r")"
)

_VIOLENCE_RE = re.compile(
    r"(?:"
    r"\b(?:gore|chacina|decapit\w*|esquartej\w*|mutila\w*|automutila\w*|tortura\w*|cadaver\w*|"
    r"espancamento|suicidio|massacre|snuff|beheading|dismember\w*)\b|"
    r"\bcorpo\s+ensanguentad\w*\b|"
    r"\bviolencia\s+(?:extrema|grafica|explicita)\b|"
    r"\bbriga\s+sangrenta\b|"
    r"\bassassinato\s+real\b|\bhomicidio\s+real\b|\bexecucao\s+real\b|"
    r"\bse\s+matar\b|\bcomo\s+me\s+matar\b|"
    r"\bgraphic\s+violence\b|\bbloody\s+gore\b|\bself[\s-]?harm\b|\bkill\s+yourself\b"
    r")"
)

BLOCKED_MESSAGE = (
    "Por segurança do Blind, não salvo, pesquiso nem abro fotos, vídeos ou mídias impróprias de sexo ou violência."
)

SAFETY_REVIEW_PROMPT = (
    "Antes de qualquer coisa, verifique se a imagem tem conteúdo impróprio: nudez, sexo explícito, pornografia, "
    "violência gráfica, sangue extremo, mutilação, cadáver ou automutilação. Fotos comuns do dia a dia (rostos, "
    "documentos com campo sexo M/F, remédios, exames médicos comuns, dinheiro, objetos da casa) são permitidas."
)


def check_text_safety(text: str) -> Dict[str, Any]:
    """Verifica se um texto indica conteudo sexual explicito ou violencia grafica sem gerar falsos positivos."""
    folded = _fold(text)
    if not folded.strip():
        return {"safe": True, "categories": [], "message": ""}
    categories: List[str] = []
    if _SEXUAL_RE.search(folded):
        categories.append("sexual")
    if _VIOLENCE_RE.search(folded):
        categories.append("violence")
    if categories:
        return {"safe": False, "categories": categories, "message": BLOCKED_MESSAGE}
    return {"safe": True, "categories": [], "message": ""}


def assert_safe_text(*texts: str, context: str = "") -> None:
    """Levanta ContentSafetyError se algum texto indicar sexo explicito ou violencia."""
    joined = " ".join((t or "") for t in texts if t)
    res = check_text_safety(joined)
    if not res["safe"]:
        raise ContentSafetyError(res["message"])


def check_filename_safety(path: str | Path) -> Dict[str, Any]:
    """Confere o nome do arquivo e da pasta imediata trocando separadores por espaco."""
    p = Path(path)
    parts = [p.stem, p.parent.name if p.parent else ""]
    normalized = re.sub(r"[_\-.]+", " ", " ".join(parts))
    return check_text_safety(normalized)


def check_youtube_metadata(info: Dict[str, Any]) -> Dict[str, Any]:
    """Confere restricao +18 e metadados de video/busca do YouTube contra sexo ou violencia."""
    if not isinstance(info, dict):
        return {"safe": True, "categories": [], "message": ""}
    categories: List[str] = []
    try:
        age_limit = int(info.get("age_limit") or 0)
    except (TypeError, ValueError):
        age_limit = 0
    if age_limit >= 18:
        categories.append("adult_18")

    pieces = [
        str(info.get("title") or ""),
        str(info.get("titulo") or ""),
        str(info.get("description") or ""),
        str(info.get("descricao_resumida") or ""),
        str(info.get("uploader") or ""),
        str(info.get("channel") or ""),
        str(info.get("canal") or ""),
    ]
    for key in ("tags", "categories"):
        val = info.get(key)
        if isinstance(val, (list, tuple)):
            pieces.extend(str(x) for x in val if x)
        elif val:
            pieces.append(str(val))

    text_check = check_text_safety(" ".join(pieces))
    for cat in text_check["categories"]:
        if cat not in categories:
            categories.append(cat)

    if categories:
        return {"safe": False, "categories": categories, "message": BLOCKED_MESSAGE}
    return {"safe": True, "categories": [], "message": ""}


def compute_file_sha256(path: str | Path, full_limit_bytes: int = _PARTIAL_HASH_LIMIT) -> str:
    """Calcula SHA-256 em streaming do arquivo (ou amostragem inicio/meio/fim para videos enormes)."""
    p = Path(path)
    try:
        if not p.is_file():
            return ""
        size = int(p.stat().st_size)
        h = hashlib.sha256()
        with p.open("rb") as fh:
            if size <= full_limit_bytes:
                while True:
                    chunk = fh.read(262144)
                    if not chunk:
                        break
                    h.update(chunk)
                return h.hexdigest()
            h.update(f"partial:{size}:".encode("ascii"))
            h.update(fh.read(_PARTIAL_CHUNK_BYTES))
            mid = max(0, (size // 2) - (_PARTIAL_CHUNK_BYTES // 2))
            fh.seek(mid)
            h.update(fh.read(_PARTIAL_CHUNK_BYTES))
            tail = max(0, size - _PARTIAL_CHUNK_BYTES)
            fh.seek(tail)
            h.update(fh.read(_PARTIAL_CHUNK_BYTES))
            return h.hexdigest()
    except OSError:
        return ""


def compute_image_dhash(source: Any) -> str:
    """Calcula dHash perceptual de 64 bits (16 hex) para reconhecer a imagem mesmo se for regravada ou redimensionada."""
    try:
        close_after = False
        if isinstance(source, (str, Path)):
            img = PILImage.open(source)
            close_after = True
        elif isinstance(source, PILImage.Image):
            img = source
        else:
            arr = np.asarray(source, dtype=np.uint8)
            if arr.ndim < 2:
                return ""
            img = PILImage.fromarray(arr[:, :, :3] if arr.ndim == 3 else arr)
            close_after = True

        try:
            gray = img.convert("L")
            probe = gray.copy()
            probe.thumbnail((128, 128))
            if float(np.std(np.asarray(probe, dtype=np.float32))) < 2.0:
                return ""
            small = gray.resize((9, 8), PILImage.Resampling.LANCZOS)
            mat = np.asarray(small, dtype=np.int16)
            bits_mat = mat[:, 1:] > mat[:, :-1]
            val = 0
            for bit in bits_mat.reshape(-1):
                val = (val << 1) | int(bool(bit))
            return f"{val:016x}"
        finally:
            if close_after:
                img.close()
    except Exception:  # noqa: BLE001
        return ""


def dhash_distance(hash_a: str, hash_b: str) -> int:
    """Distancia de Hamming entre dois hashes dHash hex de 64 bits."""
    a = (hash_a or "").strip().lower()
    b = (hash_b or "").strip().lower()
    if len(a) != 16 or len(b) != 16:
        return 64
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return 64


def dhash_matches(hash_a: str, hash_b: str, max_distance: int = DHASH_MAX_DISTANCE) -> bool:
    """True se os dois hashes perceptuais nao forem vazios e estiverem dentro da distancia permitida."""
    a = (hash_a or "").strip().lower()
    b = (hash_b or "").strip().lower()
    if len(a) != 16 or len(b) != 16:
        return False
    return dhash_distance(a, b) <= int(max_distance)


def fingerprint_media(path: str | Path, is_image: bool = False) -> Dict[str, str]:
    """Extrai SHA-256 do arquivo e dHash perceptual quando for imagem."""
    return {
        "sha256": compute_file_sha256(path),
        "dhash": compute_image_dhash(path) if is_image else "",
    }
