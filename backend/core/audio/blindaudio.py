"""Motor do blindaudio: transcricao acessivel de arquivos locais e videos do YouTube.

Prioriza legendas existentes no YouTube (manuais e depois automaticas no idioma original ou pedido)
e usa o Whisper local (faster-whisper) para arquivos de audio/video do computador ou videos do
YouTube sem legendas. Mantem cache local paginado para continuar transcricoes longas por partes.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[2]

MAX_LOCAL_BYTES = 500 * 1024 * 1024
MAX_YOUTUBE_AUDIO_BYTES = 200 * 1024 * 1024
MAX_WHISPER_YOUTUBE_SECONDS = 3600
MAX_PART_CHARS = 12000
CACHE_MAX_AGE_DAYS = 7
CACHE_MAX_FILES = 60

DEFAULT_AUDIO_MODEL = "small"
DIRECT_AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".m4a", ".opus", ".flac", ".aac"}
MAX_DIRECT_AUDIO_BYTES = 25 * 1024 * 1024
MAX_INLINE_AUDIO_BYTES = 12 * 1024 * 1024
AUDIO_CACHE_MAX_FILES = 30

YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtu.be",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

NON_MEDIA_HINTS: Dict[str, str] = {
    ".pdf": "Esse arquivo e um documento PDF. Use read_document para ler o texto.",
    ".docx": "Esse arquivo e um documento Word. Use read_document ou Blind Commander para ler.",
    ".xlsx": "Esse arquivo e uma planilha Excel. Use read_document para ler.",
    ".xlsm": "Esse arquivo e uma planilha Excel. Use read_document para ler.",
    ".pptx": "Esse arquivo e uma apresentacao PowerPoint. Use read_document para ler.",
    ".txt": "Esse arquivo e de texto simples. Leia diretamente como documento ou arquivo de texto.",
    ".md": "Esse arquivo e de texto Markdown. Leia diretamente como documento ou arquivo de texto.",
    ".jpg": "Esse arquivo e uma imagem. Use blindimage (describe_image ou ver_imagem) para descrever.",
    ".jpeg": "Esse arquivo e uma imagem. Use blindimage (describe_image ou ver_imagem) para descrever.",
    ".png": "Esse arquivo e uma imagem. Use blindimage (describe_image ou ver_imagem) para descrever.",
    ".webp": "Esse arquivo e uma imagem. Use blindimage (describe_image ou ver_imagem) para descrever.",
    ".bmp": "Esse arquivo e uma imagem. Use blindimage (describe_image ou ver_imagem) para descrever.",
    ".gif": "Esse arquivo e uma imagem. Use blindimage (describe_image ou ver_imagem) para descrever.",
}

_WHISPER_LOCK = threading.Lock()
_WHISPER_MODELS: Dict[Tuple[str, str, str, str], Any] = {}
_MODEL_ACTUAL_NAMES: Dict[int, str] = {}

_VTT_TIME_RE = re.compile(
    r"^\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{3})"
)
_CUE_TAG_RE = re.compile(r"<[^>]+>")


def _data_dir() -> Path:
    return Path(os.environ.get("BLIND_DATA_DIR") or (BACKEND_DIR / "data"))


def _cache_dir() -> Path:
    return _data_dir() / "blindaudio"


def _audios_dir() -> Path:
    return _cache_dir() / "audios"


def _model_dir() -> Path:
    return _data_dir() / "models" / "whisper"


# ----------------------------------------------------------------------------- formatacao para voz
def formatar_duracao(segundos: Optional[float]) -> str:
    """Converte segundos em duracao falada em portugues para leitor de tela."""
    if segundos is None or segundos <= 0:
        return "duração desconhecida"
    total = int(round(float(segundos)))
    if total <= 0:
        return "menos de 1 segundo"
    horas, resto = divmod(total, 3600)
    minutos, segs = divmod(resto, 60)

    def _p(val: int, sing: str, plur: str) -> str:
        return f"{val} {sing if val == 1 else plur}"

    partes: List[str] = []
    if horas:
        partes.append(_p(horas, "hora", "horas"))
    if minutos:
        partes.append(_p(minutos, "minuto", "minutos"))
    if segs and not (horas and minutos):
        partes.append(_p(segs, "segundo", "segundos"))
    if not partes:
        return "menos de 1 segundo"
    if len(partes) == 1:
        return partes[0]
    return " e ".join(partes)


def formatar_tempo(segundos: float) -> str:
    """Converte segundos em marca MM:SS ou H:MM:SS."""
    total = max(0, int(segundos))
    h, r = divmod(total, 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ----------------------------------------------------------------------------- validacao e classificacao
def _extract_youtube_id(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").strip().lower()
    parts = [p for p in parsed.path.split("/") if p]
    qs = parse_qs(parsed.query)

    if host in ("youtu.be", "www.youtu.be"):
        return parts[0] if parts else ""

    if "v" in qs and qs["v"]:
        return qs["v"][0]

    if parts and parts[0] == "playlist":
        raise ValueError(
            "Esse link é de uma lista de reprodução do YouTube. Informe o link de um vídeo específico."
        )

    if len(parts) >= 2 and parts[0] in ("shorts", "live", "embed", "v"):
        return parts[1]

    return ""


def classificar_fonte(fonte: str) -> Dict[str, Any]:
    """Valida e classifica a entrada como link do YouTube ou arquivo local."""
    clean = (fonte or "").strip().strip("\"'<>").strip()
    if not clean:
        raise ValueError("Informe o caminho de um arquivo no computador ou um link do YouTube.")
    if clean.startswith("-"):
        raise ValueError("Fonte invalida: o caminho ou link nao pode comecar com '-'.")

    bare_yt = clean.lower().startswith(
        (
            "youtube.com/",
            "www.youtube.com/",
            "m.youtube.com/",
            "music.youtube.com/",
            "youtu.be/",
            "www.youtu.be/",
            "youtube-nocookie.com/",
            "www.youtube-nocookie.com/",
        )
    )
    is_url = (
        "://" in clean
        or bare_yt
        or clean.lower().startswith(("http:", "https:", "file:", "ftp:", "javascript:", "data:"))
    )

    if is_url:
        candidate = f"https://{clean}" if bare_yt else clean
        parsed = urlparse(candidate)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("Apenas links do YouTube com http ou https sao aceitos.")
        if parsed.username or parsed.password:
            raise ValueError("Links com usuario ou senha embutidos nao sao permitidos.")
        host = (parsed.hostname or "").strip().lower()
        if host not in YOUTUBE_HOSTS:
            raise ValueError(
                "Por enquanto o blindaudio aceita arquivos locais do computador e links do YouTube. "
                "Esse endereço não é do YouTube."
            )
        video_id = _extract_youtube_id(candidate)
        if not VIDEO_ID_RE.match(video_id or ""):
            raise ValueError("Nao consegui identificar o video nesse link do YouTube.")
        return {
            "tipo": "youtube",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "video_id": video_id,
        }

    p = Path(clean).expanduser()
    suf = p.suffix.lower()
    if suf in NON_MEDIA_HINTS:
        raise ValueError(NON_MEDIA_HINTS[suf])
    if p.is_dir():
        raise ValueError("Esse caminho e uma pasta, nao um arquivo de audio ou video.")
    if not p.is_file():
        raise FileNotFoundError(f"Nao achei esse arquivo de audio ou video no computador: {clean}")
    size = p.stat().st_size
    if size == 0:
        raise ValueError("Esse arquivo esta vazio e nao tem audio para transcrever.")
    if size > MAX_LOCAL_BYTES:
        raise ValueError(
            "Esse arquivo passa do limite de 500 MB. Informe um trecho menor com inicio_segundos e fim_segundos."
        )
    return {"tipo": "local", "path": str(p.resolve())}


def _validar_intervalo(
    inicio_segundos: Optional[float], fim_segundos: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    ini = float(inicio_segundos) if inicio_segundos is not None else None
    fim = float(fim_segundos) if fim_segundos is not None else None
    if ini is not None and ini < 0:
        raise ValueError("inicio_segundos nao pode ser negativo.")
    if fim is not None and fim <= 0:
        raise ValueError("fim_segundos precisa ser maior que zero.")
    if ini is not None and fim is not None and fim <= ini:
        raise ValueError("fim_segundos precisa ser maior que inicio_segundos.")
    return ini, fim


def _normalize_idioma(idioma: Optional[str]) -> Optional[str]:
    val = (idioma or "").strip().lower()
    if not val or val == "auto":
        return None
    if "-" in val:
        val = val.split("-", 1)[0]
    if "_" in val:
        val = val.split("_", 1)[0]
    return val or None


# ----------------------------------------------------------------------------- Whisper local compartilhado
def _load_whisper_model(model_name: Optional[str] = None) -> Tuple[Any, str]:
    """Carrega o modelo faster-whisper com fallback offline para 'base' se o modelo pedido falhar."""
    name = (
        model_name
        or os.environ.get("BLIND_AUDIO_MODEL")
        or DEFAULT_AUDIO_MODEL
    ).strip()
    device = os.environ.get("BLIND_STT_DEVICE", "auto")
    compute = os.environ.get("BLIND_STT_COMPUTE", "int8")
    model_dir = _model_dir()
    key = (name, str(model_dir), device, compute)
    with _WHISPER_LOCK:
        if key in _WHISPER_MODELS:
            cached = _WHISPER_MODELS[key]
            actual = _MODEL_ACTUAL_NAMES.get(id(cached), name)
            return cached, actual
        from faster_whisper import WhisperModel

        model_dir.mkdir(parents=True, exist_ok=True)
        actual_name = name
        try:
            model = WhisperModel(
                name,
                device=device,
                compute_type=compute,
                download_root=str(model_dir),
                local_files_only=True,
            )
        except Exception:  # noqa: BLE001 - ainda nao baixado localmente: tenta baixar
            try:
                model = WhisperModel(
                    name,
                    device=device,
                    compute_type=compute,
                    download_root=str(model_dir),
                )
            except Exception:  # noqa: BLE001 - offline ou falha no download: tenta 'base' local
                if name == "base":
                    raise
                actual_name = "base"
                base_key = ("base", str(model_dir), device, compute)
                if base_key in _WHISPER_MODELS:
                    cached_base = _WHISPER_MODELS[base_key]
                    _MODEL_ACTUAL_NAMES[id(cached_base)] = "base"
                    return cached_base, "base"
                model = WhisperModel(
                    "base",
                    device=device,
                    compute_type=compute,
                    download_root=str(model_dir),
                    local_files_only=True,
                )
        actual_key = (actual_name, str(model_dir), device, compute)
        _WHISPER_MODELS[actual_key] = model
        _MODEL_ACTUAL_NAMES[id(model)] = actual_name
        try:
            model._blind_model_name = actual_name
        except Exception:  # noqa: BLE001
            pass
        return model, actual_name


def get_whisper_model(model_name: Optional[str] = None) -> Any:
    """Carrega o modelo faster-whisper uma vez por processo, reutilizavel por canais e blindaudio."""
    return _load_whisper_model(model_name)[0]


def baixar_youtube_com_retry(opts: Dict[str, Any], url: str) -> None:
    """Baixa com o yt-dlp. O corte de trecho (download_ranges) usa o ffmpeg e as vezes ele cai de forma intermitente
    ("ffmpeg exited with code ..."): nesse caso tenta mais uma vez antes de desistir."""
    import yt_dlp

    for tentativa in (1, 2):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
            return
        except Exception as exc:  # noqa: BLE001
            if tentativa == 1 and opts.get("download_ranges") and "ffmpeg exited" in str(exc).lower():
                continue
            raise RuntimeError(_traduzir_erro_ytdlp(exc)) from exc


def _recortar_com_ffmpeg(
    path: str, tmp_dir: str, ini: Optional[float], fim: Optional[float]
) -> Optional[str]:
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    out_wav = str(Path(tmp_dir) / "trecho.wav")
    cmd = [ff, "-nostdin", "-y", "-loglevel", "error"]
    if ini is not None:
        cmd += ["-ss", str(ini)]
    cmd += ["-i", path]
    if fim is not None:
        dur = fim - (ini or 0.0)
        cmd += ["-t", str(dur)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", out_wav]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode == 0 and Path(out_wav).is_file() and Path(out_wav).stat().st_size > 44:
        return out_wav
    return None


def transcribe_local_whisper(
    path: str,
    idioma: Optional[str] = "auto",
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
    model_name: Optional[str] = None,
    musica: bool = False,
) -> Dict[str, Any]:
    """Transcreve um arquivo de audio ou video local usando faster-whisper.

    `musica=True` desliga o filtro de voz (VAD), que descarta canto sobre instrumental."""
    ini, fim = _validar_intervalo(inicio_segundos, fim_segundos)
    lang = _normalize_idioma(idioma)
    req_model_name = (
        model_name
        or os.environ.get("BLIND_AUDIO_MODEL")
        or DEFAULT_AUDIO_MODEL
    ).strip()
    model = get_whisper_model(req_model_name)
    used_model_name = (
        getattr(model, "_blind_model_name", None)
        or _MODEL_ACTUAL_NAMES.get(id(model))
        or req_model_name
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        target = path
        used_ffmpeg_trim = False
        if ini is not None or fim is not None:
            trimmed = _recortar_com_ffmpeg(path, tmp_dir, ini, fim)
            if trimmed:
                target = trimmed
                used_ffmpeg_trim = True

        segments, info = model.transcribe(
            target,
            language=lang,
            beam_size=5,
            vad_filter=not musica,
            condition_on_previous_text=False,
        )
        offset = ini if (used_ffmpeg_trim and ini is not None) else 0.0
        segmentos: List[Dict[str, Any]] = []
        for seg in segments:
            txt = re.sub(r"\s+", " ", getattr(seg, "text", "") or "").strip()
            if not txt:
                continue
            s_ini = round(float(getattr(seg, "start", 0.0)) + offset, 2)
            s_fim = round(float(getattr(seg, "end", 0.0)) + offset, 2)
            if not used_ffmpeg_trim:
                if ini is not None and s_fim <= ini:
                    continue
                if fim is not None and s_ini >= fim:
                    continue
            segmentos.append({"inicio": s_ini, "fim": s_fim, "texto": txt})

    text = " ".join(s["texto"] for s in segmentos).strip()
    return {
        "ok": True,
        "text": text,
        "segmentos": segmentos,
        "language": getattr(info, "language", None) or lang or "pt",
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 3),
        "duration_seconds": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
        "model": used_model_name,
    }


# ----------------------------------------------------------------------------- YouTube (metadados, legendas e audio)
class _LoggerMudo:
    """Logger do yt-dlp que descarta tudo. Num servidor MCP por stdio, o stdout e o canal do protocolo: o yt-dlp (barra de
    progresso do download por trecho, "[download] 100% ...") escrevendo nele corrompe a resposta e a chamada fica pendurada."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


def _ydl_base_opts() -> Dict[str, Any]:
    runtimes: Dict[str, Dict[str, Any]] = {}
    if shutil.which("node"):
        runtimes["node"] = {}
    if shutil.which("deno"):
        runtimes["deno"] = {}
    opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _LoggerMudo(),
        "noplaylist": True,
        "socket_timeout": 30,
    }
    if runtimes:
        opts["js_runtimes"] = runtimes
    ff = shutil.which("ffmpeg")
    if ff:
        opts["ffmpeg_location"] = str(Path(ff).parent)
    return opts


def _format_upload_date(raw: Optional[str]) -> str:
    s = (raw or "").strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return s


def _traduzir_erro_ytdlp(exc: Exception) -> str:
    msg = str(exc or "").lower()
    if "private" in msg or "unavailable" in msg:
        return "Esse video do YouTube esta privado ou indisponivel."
    if "sign in" in msg or "age" in msg:
        return "Esse video do YouTube exige login ou confirmacao de idade e nao pode ser aberto sem conta."
    if "live" in msg:
        return "Esse video e uma transmissao ao vivo e nao tem audio fechado para transcrever."
    return f"Nao consegui acessar esse video do YouTube: {str(exc)[:180]}"


def obter_info_youtube(url: str) -> Dict[str, Any]:
    """Busca metadados e legendas disponiveis de um video do YouTube sem baixar o video."""
    import yt_dlp

    opts = _ydl_base_opts()
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_traduzir_erro_ytdlp(exc)) from exc

    if not isinstance(info, dict):
        raise RuntimeError("Nao consegui obter as informacoes desse video do YouTube.")

    dur = float(info.get("duration") or 0.0)
    raw_subs = info.get("subtitles") or {}
    raw_auto = info.get("automatic_captions") or {}
    subtitles = {k: v for k, v in raw_subs.items() if k != "live_chat" and v}
    auto_captions = {k: v for k, v in raw_auto.items() if k != "live_chat" and v}
    capitulos = [
        {
            "titulo": (c.get("title") or "").strip(),
            "inicio": round(float(c.get("start_time") or 0.0), 2),
            "fim": round(float(c.get("end_time") or 0.0), 2),
        }
        for c in (info.get("chapters") or [])
        if isinstance(c, dict)
    ]

    manuais = sorted(subtitles.keys())
    automaticas = sorted(auto_captions.keys())
    return {
        "ok": True,
        "tipo": "youtube",
        "video_id": info.get("id") or _extract_youtube_id(url),
        "url": info.get("webpage_url") or url,
        "titulo": (info.get("title") or "Video do YouTube").strip(),
        "canal": (info.get("uploader") or info.get("channel") or "").strip(),
        "duracao_segundos": round(dur, 2),
        "duracao_falada": formatar_duracao(dur),
        "data_publicacao": _format_upload_date(info.get("upload_date")),
        "idioma_video": (info.get("language") or "").strip(),
        "ao_vivo": bool(info.get("is_live")),
        "capitulos": capitulos,
        "tem_legendas": bool(manuais or automaticas),
        "idiomas_legendas_manuais": manuais,
        "idiomas_legendas_automaticas": automaticas,
        "_subtitles": subtitles,
        "_automatic_captions": auto_captions,
    }


def _pick_track(tracks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for ext in ("json3", "vtt", "srv3", "ttml"):
        for t in tracks:
            if isinstance(t, dict) and (t.get("ext") or "").lower() == ext and t.get("url"):
                return t
    for t in tracks:
        if isinstance(t, dict) and t.get("url"):
            return t
    return None


def _is_translated_track(tracks: List[Dict[str, Any]]) -> bool:
    for t in tracks:
        url = (t.get("url") or "").lower() if isinstance(t, dict) else ""
        if "tlang=" in url:
            return True
    return False


def _match_lang_keys(keys: List[str], target: str) -> List[str]:
    t = (target or "").strip().lower()
    if not t:
        return []
    base = t.split("-", 1)[0].split("_", 1)[0]
    exact: List[str] = []
    orig: List[str] = []
    regional: List[str] = []
    for k in keys:
        kl = k.lower()
        if kl == f"{t}-orig" or kl == f"{base}-orig":
            orig.append(k)
        elif kl == t:
            exact.append(k)
        elif kl == base or kl.startswith(base + "-") or kl.startswith(base + "_"):
            regional.append(k)
    # Em portugues, prefere pt-BR antes de pt-PT
    if base == "pt":
        regional.sort(key=lambda x: (0 if "br" in x.lower() else 1, x))
    return orig + exact + regional


def _escolher_legenda(
    subtitles: Dict[str, List[Dict[str, Any]]],
    auto_captions: Dict[str, List[Dict[str, Any]]],
    idioma_pedido: str = "auto",
    idioma_video: str = "",
) -> Optional[Tuple[str, str, str, str]]:
    """Escolhe a melhor faixa de legenda.

    Retorna (idioma, ext, url, tipo_legenda) onde tipo_legenda e
    'legenda_manual' ou 'legenda_automatica', ou None se nao houver legenda adequada.
    """
    req = (idioma_pedido or "").strip().lower()
    auto_mode = not req or req == "auto"

    man_keys = [k for k in subtitles if k != "live_chat" and subtitles[k]]
    auto_keys = [k for k in auto_captions if k != "live_chat" and auto_captions[k]]

    if auto_mode:
        # 1. Legenda manual: idioma do video -> pt -> en -> qualquer outra manual
        search_order = [idioma_video, "pt", "en"]
        for cand in search_order:
            for k in _match_lang_keys(man_keys, cand):
                tr = _pick_track(subtitles[k])
                if tr:
                    return k, (tr.get("ext") or "vtt").lower(), tr["url"], "legenda_manual"
        for k in man_keys:
            tr = _pick_track(subtitles[k])
            if tr:
                return k, (tr.get("ext") or "vtt").lower(), tr["url"], "legenda_manual"

        # 2. Legenda automatica ORIGINAL (evita traducoes automaticas tlang=)
        orig_keys = [k for k in auto_keys if k.lower().endswith("-orig")]
        non_trans_keys = [k for k in auto_keys if not _is_translated_track(auto_captions[k])]
        pool = orig_keys + [k for k in non_trans_keys if k not in orig_keys]
        for cand in [idioma_video, "pt", "en"]:
            for k in _match_lang_keys(pool, cand):
                tr = _pick_track(auto_captions[k])
                if tr:
                    return k, (tr.get("ext") or "vtt").lower(), tr["url"], "legenda_automatica"
        for k in pool:
            tr = _pick_track(auto_captions[k])
            if tr:
                return k, (tr.get("ext") or "vtt").lower(), tr["url"], "legenda_automatica"
        return None

    # Idioma especifico pedido pela pessoa
    for k in _match_lang_keys(man_keys, req):
        tr = _pick_track(subtitles[k])
        if tr:
            return k, (tr.get("ext") or "vtt").lower(), tr["url"], "legenda_manual"

    matched_auto = _match_lang_keys(auto_keys, req)
    non_trans = [k for k in matched_auto if not _is_translated_track(auto_captions[k])]
    trans = [k for k in matched_auto if k not in non_trans]
    for k in non_trans + trans:
        tr = _pick_track(auto_captions[k])
        if tr:
            return k, (tr.get("ext") or "vtt").lower(), tr["url"], "legenda_automatica"
    return None


def _baixar_texto_legenda(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise ValueError("URL de legenda nao usa HTTPS.")
    host = (parsed.hostname or "").strip().lower()
    allowed = (
        host == "youtube.com"
        or host.endswith(".youtube.com")
        or host == "googlevideo.com"
        or host.endswith(".googlevideo.com")
    )
    if not allowed:
        raise ValueError("Host de legenda inesperado.")
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.text


def _coalesce_segments(raw_segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrupa micro-segmentos em blocos legiveis (~15s ou ate fim de frase)."""
    if not raw_segs:
        return []
    out: List[Dict[str, Any]] = []
    cur_start = float(raw_segs[0]["inicio"])
    cur_end = float(raw_segs[0]["fim"])
    buf: List[str] = []
    cur_len = 0

    for item in raw_segs:
        txt = item["texto"]
        s_ini = float(item["inicio"])
        s_fim = float(item["fim"])
        if not buf:
            cur_start = s_ini
        buf.append(txt)
        cur_len += len(txt) + 1
        cur_end = max(cur_end, s_fim)
        ended_sentence = txt.endswith((".", "!", "?"))
        if (cur_end - cur_start >= 15.0 and ended_sentence) or (cur_end - cur_start >= 22.0) or cur_len >= 280:
            out.append({"inicio": round(cur_start, 2), "fim": round(cur_end, 2), "texto": " ".join(buf)})
            buf = []
            cur_len = 0
    if buf:
        out.append({"inicio": round(cur_start, 2), "fim": round(cur_end, 2), "texto": " ".join(buf)})
    return out


def _parse_json3(raw: str) -> List[Dict[str, Any]]:
    """Extrai segmentos limpos de legenda no formato JSON3 do YouTube."""
    data = json.loads(raw)
    events = data.get("events") or []
    raw_segs: List[Dict[str, Any]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        segs = ev.get("segs") or []
        text = "".join(s.get("utf8", "") for s in segs if isinstance(s, dict))
        text = html.unescape(text).replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        t_ms = float(ev.get("tStartMs") or 0.0)
        d_ms = float(ev.get("dDurationMs") or 0.0)
        ini = round(t_ms / 1000.0, 2)
        fim = round((t_ms + d_ms) / 1000.0, 2)
        if raw_segs and raw_segs[-1]["texto"] == text:
            raw_segs[-1]["fim"] = max(raw_segs[-1]["fim"], fim)
            continue
        raw_segs.append({"inicio": ini, "fim": fim, "texto": text})
    return _coalesce_segments(raw_segs)


def _vtt_to_seconds(h: Optional[str], m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _parse_vtt(raw: str) -> List[Dict[str, Any]]:
    """Extrai segmentos de WebVTT removendo tags e duplicacoes de legenda rolante."""
    lines = raw.splitlines()
    raw_segs: List[Dict[str, Any]] = []
    last_lines: List[str] = []
    i = 0
    while i < len(lines):
        m = _VTT_TIME_RE.match(lines[i])
        if not m:
            i += 1
            continue
        ini = _vtt_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        fim = _vtt_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        i += 1
        cue_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            cleaned = html.unescape(_CUE_TAG_RE.sub("", lines[i]))
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned and (not cue_lines or cleaned != cue_lines[-1]):
                cue_lines.append(cleaned)
            i += 1
        if not cue_lines:
            continue
        # Remove sobreposicao tipica do YouTube (a linha final da legenda anterior aparece como primeira linha da proxima)
        new_lines = [ln for ln in cue_lines if ln not in last_lines]
        last_lines = cue_lines[-2:]
        text = " ".join(new_lines).strip()
        if not text:
            continue
        raw_segs.append({"inicio": round(ini, 2), "fim": round(fim, 2), "texto": text})
    return _coalesce_segments(raw_segs)


def _parse_xml_captions(raw: str) -> List[Dict[str, Any]]:
    """Parser simples para srv3/ttml caso o YouTube nao devolva json3 nem vtt."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(raw)
    raw_segs: List[Dict[str, Any]] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        if tag not in ("p", "text"):
            continue
        txt = "".join(elem.itertext())
        txt = html.unescape(re.sub(r"\s+", " ", txt)).strip()
        if not txt:
            continue
        if "t" in elem.attrib:
            ini = float(elem.attrib["t"]) / 1000.0
            dur = float(elem.attrib.get("d", "2000")) / 1000.0
        else:
            ini = float(elem.attrib.get("start", "0") or 0)
            dur = float(elem.attrib.get("dur", "2") or 2)
        raw_segs.append({"inicio": round(ini, 2), "fim": round(ini + dur, 2), "texto": txt})
    return _coalesce_segments(raw_segs)


def _parse_captions(raw: str, ext: str) -> List[Dict[str, Any]]:
    fmt = (ext or "").lower()
    if fmt == "json3":
        return _parse_json3(raw)
    if fmt == "vtt":
        return _parse_vtt(raw)
    return _parse_xml_captions(raw)


def _transcrever_youtube_whisper(
    url: str,
    info: Dict[str, Any],
    idioma: str,
    inicio_segundos: Optional[float],
    fim_segundos: Optional[float],
) -> Dict[str, Any]:
    """Baixa apenas o audio do video do YouTube para pasta temporaria e transcreve com Whisper local."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    if info.get("ao_vivo"):
        raise ValueError("Esse video esta ao vivo no momento e nao tem legendas disponiveis para transcricao.")

    ini, fim = _validar_intervalo(inicio_segundos, fim_segundos)
    dur = float(info.get("duracao_segundos") or 0.0)
    if dur > MAX_WHISPER_YOUTUBE_SECONDS and ini is None and fim is None:
        raise ValueError(
            f"Esse video tem {formatar_duracao(dur)} e nao possui legendas no YouTube. "
            "Para transcrever pelo audio local, informe inicio_segundos e fim_segundos de um trecho."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        opts = _ydl_base_opts()
        opts.update(
            {
                "format": "bestaudio[ext=m4a]/bestaudio/best",
                "outtmpl": str(Path(tmp_dir) / "audio.%(ext)s"),
                "max_filesize": MAX_YOUTUBE_AUDIO_BYTES,
            }
        )
        used_ranges = False
        if (ini is not None or fim is not None) and shutil.which("ffmpeg"):
            opts["download_ranges"] = download_range_func(
                None, [(ini or 0.0, fim if fim is not None else float("inf"))]
            )
            used_ranges = True

        baixar_youtube_com_retry(opts, url)

        files = [p for p in Path(tmp_dir).iterdir() if p.is_file() and not p.name.endswith(".part")]
        if not files:
            raise RuntimeError("Nao consegui baixar o audio desse video do YouTube.")
        audio_file = max(files, key=lambda p: p.stat().st_size)
        if audio_file.stat().st_size > MAX_YOUTUBE_AUDIO_BYTES:
            raise ValueError("O audio desse video passou do limite de 200 MB.")

        res = transcribe_local_whisper(
            str(audio_file),
            idioma=idioma,
            inicio_segundos=None if used_ranges else ini,
            fim_segundos=None if used_ranges else fim,
        )
        if used_ranges and ini:
            for seg in res["segmentos"]:
                seg["inicio"] = round(seg["inicio"] + ini, 2)
                seg["fim"] = round(seg["fim"] + ini, 2)
        return res


# ----------------------------------------------------------------------------- arquivo local (metadados)
def _probe_ffprobe(path: str) -> Optional[Dict[str, Any]]:
    fp = shutil.which("ffprobe")
    if not fp:
        return None
    try:
        proc = subprocess.run(
            [fp, "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return json.loads(proc.stdout)
    except Exception:  # noqa: BLE001
        return None
    return None


def obter_info_local(path: str) -> Dict[str, Any]:
    """Le formato, tamanho e duracao de um arquivo de audio ou video local."""
    p = Path(path)
    size = p.stat().st_size
    dur = 0.0
    tem_audio = True
    formato = p.suffix.lstrip(".").lower() or "desconhecido"

    probed = _probe_ffprobe(str(p))
    if probed:
        fmt = probed.get("format") or {}
        if fmt.get("duration"):
            try:
                dur = float(fmt["duration"])
            except ValueError:
                pass
        if fmt.get("format_name"):
            formato = str(fmt["format_name"]).split(",", 1)[0]
        streams = probed.get("streams") or []
        if streams:
            tem_audio = any(s.get("codec_type") == "audio" for s in streams)
    else:
        try:
            with wave.open(str(p), "rb") as w:
                if w.getframerate() > 0:
                    dur = w.getnframes() / float(w.getframerate())
                formato = "wav"
        except Exception:  # noqa: BLE001
            try:
                import av

                with av.open(str(p)) as container:
                    if container.duration:
                        dur = float(container.duration) / 1_000_000.0
                    tem_audio = bool(container.streams.audio)
            except Exception:  # noqa: BLE001
                pass

    return {
        "ok": True,
        "tipo": "local",
        "arquivo": p.name,
        "caminho": str(p),
        "formato": formato,
        "tamanho_mb": round(size / (1024 * 1024), 2),
        "duracao_segundos": round(dur, 2),
        "duracao_falada": formatar_duracao(dur),
        "tem_audio": tem_audio,
    }


# ----------------------------------------------------------------------------- cache e paginacao
def _cache_key(parts: Tuple[Any, ...]) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    path = _cache_dir() / f"{key}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _prune_cache() -> None:
    cdir = _cache_dir()
    if not cdir.is_dir():
        return
    cutoff = time.time() - CACHE_MAX_AGE_DAYS * 86400
    files: List[Path] = []
    for f in cdir.glob("*.json"):
        try:
            st = f.stat()
            if st.st_mtime < cutoff:
                f.unlink()
            else:
                files.append(f)
        except OSError:
            pass
    if len(files) > CACHE_MAX_FILES:
        files.sort(key=lambda x: x.stat().st_mtime)
        for old in files[: len(files) - CACHE_MAX_FILES]:
            try:
                old.unlink()
            except OSError:
                pass


def _save_cache(key: str, data: Dict[str, Any]) -> None:
    cdir = _cache_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{key}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    _prune_cache()


def _filtrar_segmentos(
    segmentos: List[Dict[str, Any]], ini: Optional[float], fim: Optional[float]
) -> List[Dict[str, Any]]:
    if ini is None and fim is None:
        return list(segmentos)
    out: List[Dict[str, Any]] = []
    for s in segmentos:
        s_ini = float(s.get("inicio", 0.0))
        s_fim = float(s.get("fim", s_ini))
        if ini is not None and s_fim <= ini:
            continue
        if fim is not None and s_ini >= fim:
            continue
        out.append(s)
    return out


def _paginar_segmentos(
    segmentos: List[Dict[str, Any]], com_tempos: bool, max_chars: Optional[int] = None
) -> List[str]:
    limit = MAX_PART_CHARS if max_chars is None else max_chars
    if not segmentos:
        return [""]
    sep = "\n" if com_tempos else " "
    partes: List[str] = []
    atual: List[str] = []
    atual_len = 0

    for seg in segmentos:
        txt = (seg.get("texto") or "").strip()
        if not txt:
            continue
        linha = f"[{formatar_tempo(float(seg.get('inicio', 0.0)))}] {txt}" if com_tempos else txt
        if len(linha) > limit:
            if atual:
                partes.append(sep.join(atual))
                atual = []
                atual_len = 0
            for start in range(0, len(linha), limit):
                partes.append(linha[start : start + limit])
            continue
        adicional = len(linha) + (len(sep) if atual else 0)
        if atual and atual_len + adicional > limit:
            partes.append(sep.join(atual))
            atual = [linha]
            atual_len = len(linha)
        else:
            atual.append(linha)
            atual_len += adicional
    if atual:
        partes.append(sep.join(atual))
    return partes or [""]


def _aviso_precisao(metodo: str, modelo_whisper: str = "") -> str:
    if metodo == "legenda_manual":
        return "Legenda manual do autor no YouTube."
    if metodo == "legenda_automatica":
        return (
            "Legenda automatica do YouTube. Nomes proprios, termos tecnicos e numeros podem ter pequenos erros."
        )
    suf = f" ({modelo_whisper})" if modelo_whisper else ""
    return (
        f"Transcricao automatica local com Whisper{suf}. Nomes proprios, termos tecnicos e numeros podem ter pequenos erros."
    )


def _prune_audios() -> None:
    adir = _audios_dir()
    if not adir.is_dir():
        return
    cutoff = time.time() - CACHE_MAX_AGE_DAYS * 86400
    files: List[Path] = []
    for f in adir.iterdir():
        if not f.is_file():
            continue
        try:
            st = f.stat()
            if st.st_mtime < cutoff:
                f.unlink()
            else:
                files.append(f)
        except OSError:
            pass
    if len(files) > AUDIO_CACHE_MAX_FILES:
        files.sort(key=lambda x: x.stat().st_mtime)
        for old in files[: len(files) - AUDIO_CACHE_MAX_FILES]:
            try:
                old.unlink()
            except OSError:
                pass


def _converter_para_mp3(
    src_path: str, dst_path: Path, ini: Optional[float], fim: Optional[float]
) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = dst_path.with_suffix(".tmp.mp3")
    cmd = [ff, "-y", "-loglevel", "error"]
    if ini is not None:
        cmd += ["-ss", str(ini)]
    cmd += ["-i", src_path]
    if fim is not None:
        cmd += ["-t", str(fim - (ini or 0.0))]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(tmp_out)]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if proc.returncode == 0 and tmp_out.is_file() and tmp_out.stat().st_size > 0:
        tmp_out.replace(dst_path)
        return True
    try:
        if tmp_out.is_file():
            tmp_out.unlink()
    except OSError:
        pass
    return False


# ----------------------------------------------------------------------------- pontos de entrada publicos
def info_fonte(fonte: str) -> Dict[str, Any]:
    """Devolve metadados de um arquivo de audio/video local ou de um link do YouTube."""
    src = classificar_fonte(fonte)
    if src["tipo"] == "local":
        return obter_info_local(src["path"])
    raw = obter_info_youtube(src["url"])
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def transcrever_fonte(
    fonte: str,
    idioma: str = "auto",
    com_tempos: bool = False,
    continuar: bool = False,
    parte: int = 0,
    forcar_whisper: bool = False,
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
    musica: bool = False,
) -> Dict[str, Any]:
    """Transcreve arquivo local ou video do YouTube, com cache e leitura por partes.

    `musica` vale so para arquivo local: desliga o filtro de voz para musica cantada."""
    src = classificar_fonte(fonte)
    ini, fim = _validar_intervalo(inicio_segundos, fim_segundos)
    idioma_norm = (idioma or "auto").strip().lower() or "auto"
    active_model = (os.environ.get("BLIND_AUDIO_MODEL") or DEFAULT_AUDIO_MODEL).strip()

    if src["tipo"] == "youtube":
        key = _cache_key(
            ("youtube", src["video_id"], idioma_norm, f"whisper:{active_model}" if forcar_whisper else f"auto:{active_model}", ini, fim)
        )
        entry = _load_cache(key)
        if entry is None:
            yt_info = obter_info_youtube(src["url"])
            chosen = (
                None
                if forcar_whisper
                else _escolher_legenda(
                    yt_info["_subtitles"],
                    yt_info["_automatic_captions"],
                    idioma_pedido=idioma_norm,
                    idioma_video=yt_info.get("idioma_video", ""),
                )
            )
            modelo_whisper = ""
            if chosen is not None:
                lang_key, ext, cap_url, metodo = chosen
                raw_cap = _baixar_texto_legenda(cap_url)
                segmentos = _parse_captions(raw_cap, ext)
                lang_out = lang_key.removesuffix("-orig")
            else:
                res = _transcrever_youtube_whisper(src["url"], yt_info, idioma_norm, ini, fim)
                segmentos = res["segmentos"]
                metodo = "whisper_local"
                lang_out = res.get("language") or idioma_norm
                modelo_whisper = res.get("model") or active_model

            entry = {
                "tipo": "youtube",
                "fonte": yt_info["url"],
                "video_id": yt_info["video_id"],
                "titulo": yt_info["titulo"],
                "canal": yt_info["canal"],
                "duracao_segundos": yt_info["duracao_segundos"],
                "duracao_falada": yt_info["duracao_falada"],
                "idioma": lang_out,
                "metodo": metodo,
                "modelo_whisper": modelo_whisper,
                "capitulos": yt_info["capitulos"],
                "segmentos": segmentos,
                "ultima_parte_lida": 0,
            }
            _save_cache(key, entry)
    else:
        p = Path(src["path"])
        st = p.stat()
        key = _cache_key(("local", src["path"], st.st_size, st.st_mtime_ns, idioma_norm, active_model, ini, fim, "musica" if musica else "fala"))
        entry = _load_cache(key)
        if entry is None:
            local_info = obter_info_local(src["path"])
            if not local_info.get("tem_audio", True):
                raise ValueError("Esse arquivo de video nao tem faixa de audio para transcrever.")
            res = transcribe_local_whisper(
                src["path"],
                idioma=idioma_norm,
                inicio_segundos=ini,
                fim_segundos=fim,
                musica=musica,
            )
            dur = float(local_info["duracao_segundos"] or res.get("duration_seconds") or 0.0)
            entry = {
                "tipo": "local",
                "fonte": src["path"],
                "titulo": p.name,
                "canal": "",
                "duracao_segundos": round(dur, 2),
                "duracao_falada": formatar_duracao(dur),
                "idioma": res.get("language") or idioma_norm,
                "metodo": "whisper_local",
                "modelo_whisper": res.get("model") or active_model,
                "capitulos": [],
                "segmentos": res["segmentos"],
                "ultima_parte_lida": 0,
            }
            _save_cache(key, entry)

    segmentos = _filtrar_segmentos(entry.get("segmentos") or [], ini, fim)
    partes = _paginar_segmentos(segmentos, com_tempos=com_tempos)
    total_partes = len(partes)
    total_palavras = sum(len((s.get("texto") or "").split()) for s in segmentos)
    modelo_usado = entry.get("modelo_whisper") or ""
    aviso = _aviso_precisao(entry["metodo"], modelo_usado)

    if parte and int(parte) > 0:
        alvo = int(parte)
        if alvo > total_partes:
            return {
                "ok": True,
                "tipo": entry["tipo"],
                "fonte": entry["fonte"],
                "titulo": entry["titulo"],
                **({"canal": entry["canal"]} if entry.get("canal") else {}),
                "duracao_segundos": entry["duracao_segundos"],
                "duracao_falada": entry["duracao_falada"],
                "idioma": entry["idioma"],
                "metodo": entry["metodo"],
                **({"modelo_whisper": modelo_usado} if entry["metodo"] == "whisper_local" and modelo_usado else {}),
                "aviso_precisao": aviso,
                "total_palavras": total_palavras,
                "parte_atual": total_partes,
                "total_partes": total_partes,
                "tem_mais": False,
                "fim_da_transcricao": True,
                "texto": "",
                "nota": f"A transcricao tem apenas {total_partes} parte(s). Use parte=1 para ouvir do comeco.",
            }
    elif continuar:
        ultima = int(entry.get("ultima_parte_lida") or 0)
        if ultima >= total_partes:
            return {
                "ok": True,
                "tipo": entry["tipo"],
                "fonte": entry["fonte"],
                "titulo": entry["titulo"],
                **({"canal": entry["canal"]} if entry.get("canal") else {}),
                "duracao_segundos": entry["duracao_segundos"],
                "duracao_falada": entry["duracao_falada"],
                "idioma": entry["idioma"],
                "metodo": entry["metodo"],
                **({"modelo_whisper": modelo_usado} if entry["metodo"] == "whisper_local" and modelo_usado else {}),
                "aviso_precisao": aviso,
                "total_palavras": total_palavras,
                "parte_atual": ultima,
                "total_partes": total_partes,
                "tem_mais": False,
                "fim_da_transcricao": True,
                "texto": "",
                "nota": "Voce ja leu ate o final da transcricao. Use parte=1 para reler do comeco.",
            }
        alvo = ultima + 1
    else:
        alvo = 1

    entry["ultima_parte_lida"] = alvo
    _save_cache(key, entry)

    tem_mais = alvo < total_partes
    out: Dict[str, Any] = {
        "ok": True,
        "tipo": entry["tipo"],
        "fonte": entry["fonte"],
        "titulo": entry["titulo"],
        "duracao_segundos": entry["duracao_segundos"],
        "duracao_falada": entry["duracao_falada"],
        "idioma": entry["idioma"],
        "metodo": entry["metodo"],
        "aviso_precisao": aviso,
        "total_palavras": total_palavras,
        "parte_atual": alvo,
        "total_partes": total_partes,
        "tem_mais": tem_mais,
        "texto": partes[alvo - 1],
    }
    if entry["metodo"] == "whisper_local" and modelo_usado:
        out["modelo_whisper"] = modelo_usado
    if entry.get("canal"):
        out["canal"] = entry["canal"]
    if entry.get("capitulos"):
        out["capitulos"] = entry["capitulos"]
    if tem_mais:
        out["para_continuar"] = "Chame blindaudio com continuar=true para ouvir a proxima parte."
    if total_palavras == 0:
        out["nota"] = "Nenhuma fala foi reconhecida nesse audio."
    return out


def preparar_audio_para_ouvir(
    fonte: str,
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
    incluir_bytes: bool = False,
) -> Dict[str, Any]:
    """Prepara um arquivo de audio compacto para modelos que ouvem audio diretamente (sem Whisper)."""
    src = classificar_fonte(fonte)
    ini, fim = _validar_intervalo(inicio_segundos, fim_segundos)

    if src["tipo"] == "local":
        local_info = obter_info_local(src["path"])
        if not local_info.get("tem_audio", True):
            raise ValueError("Esse arquivo de video nao tem faixa de audio para ouvir.")
        p = Path(src["path"])
        dur = float(local_info.get("duracao_segundos") or 0.0)
        if (
            ini is None
            and fim is None
            and p.suffix.lower() in DIRECT_AUDIO_EXTS
            and p.stat().st_size <= MAX_DIRECT_AUDIO_BYTES
        ):
            out_path = p
            convertido = False
        else:
            st = p.stat()
            key = _cache_key(("ouvir-local", src["path"], st.st_size, st.st_mtime_ns, ini, fim))
            out_path = _audios_dir() / f"local-{key}.mp3"
            if not out_path.is_file() or out_path.stat().st_size == 0:
                if not _converter_para_mp3(src["path"], out_path, ini, fim):
                    raise RuntimeError("Nao consegui preparar esse arquivo de audio com o ffmpeg.")
                _prune_audios()
            convertido = True
            if ini is not None or fim is not None:
                end_eff = min(fim, dur) if (fim is not None and dur > 0) else (fim or dur)
                if end_eff > (ini or 0.0):
                    dur = end_eff - (ini or 0.0)
        meta_extra: Dict[str, Any] = {
            "tipo": "local",
            "fonte": src["path"],
            "titulo": p.name,
        }
    else:
        yt_info = obter_info_youtube(src["url"])
        if yt_info.get("ao_vivo"):
            raise ValueError("Esse video esta ao vivo no momento e nao tem arquivo de audio fechado para ouvir.")
        dur = float(yt_info.get("duracao_segundos") or 0.0)
        if dur > MAX_WHISPER_YOUTUBE_SECONDS and ini is None and fim is None:
            raise ValueError(
                f"Esse video tem {formatar_duracao(dur)}. Para baixar so o audio para o modelo ouvir, "
                "informe inicio_segundos e fim_segundos de um trecho."
            )
        key = _cache_key(("ouvir-yt", yt_info["video_id"], ini, fim))
        stem = f"yt-{yt_info['video_id']}-{key[:10]}"
        adir = _audios_dir()
        cached_mp3 = adir / f"{stem}.mp3"
        existing = (
            cached_mp3
            if cached_mp3.is_file() and cached_mp3.stat().st_size > 0
            else next(
                (f for f in adir.glob(f"{stem}.*") if f.is_file() and f.stat().st_size > 0),
                None,
            )
            if adir.is_dir()
            else None
        )

        if existing is not None:
            out_path = existing
            convertido = out_path.suffix.lower() == ".mp3"
            try:
                os.utime(out_path, None)
            except OSError:
                pass
        else:
            import yt_dlp
            from yt_dlp.utils import download_range_func

            with tempfile.TemporaryDirectory() as tmp_dir:
                opts = _ydl_base_opts()
                opts.update(
                    {
                        "format": "bestaudio[ext=m4a]/bestaudio/best",
                        "outtmpl": str(Path(tmp_dir) / "audio.%(ext)s"),
                        "max_filesize": MAX_YOUTUBE_AUDIO_BYTES,
                    }
                )
                used_ranges = False
                if (ini is not None or fim is not None) and shutil.which("ffmpeg"):
                    opts["download_ranges"] = download_range_func(
                        None, [(ini or 0.0, fim if fim is not None else float("inf"))]
                    )
                    used_ranges = True
                baixar_youtube_com_retry(opts, src["url"])

                files = [p for p in Path(tmp_dir).iterdir() if p.is_file() and not p.name.endswith(".part")]
                if not files:
                    raise RuntimeError("Nao consegui baixar o audio desse video do YouTube.")
                dl_file = max(files, key=lambda p: p.stat().st_size)
                if dl_file.stat().st_size > MAX_YOUTUBE_AUDIO_BYTES:
                    raise ValueError("O audio desse video passou do limite de 200 MB.")

                adir.mkdir(parents=True, exist_ok=True)
                conv_ini = None if used_ranges else ini
                conv_fim = None if used_ranges else fim
                if _converter_para_mp3(str(dl_file), cached_mp3, conv_ini, conv_fim):
                    out_path = cached_mp3
                    convertido = True
                else:
                    if ini is not None or fim is not None:
                        raise RuntimeError("Preciso do ffmpeg para recortar um trecho de audio do YouTube.")
                    out_path = adir / f"{stem}{dl_file.suffix or '.m4a'}"
                    shutil.copy2(dl_file, out_path)
                    convertido = False
                _prune_audios()

        if ini is not None or fim is not None:
            end_eff = min(fim, dur) if (fim is not None and dur > 0) else (fim or dur)
            if end_eff > (ini or 0.0):
                dur = end_eff - (ini or 0.0)
        meta_extra = {
            "tipo": "youtube",
            "fonte": yt_info["url"],
            "titulo": yt_info["titulo"],
            "canal": yt_info["canal"],
            "capitulos": yt_info["capitulos"],
        }

    formato = out_path.suffix.lstrip(".").lower() or "mp3"
    size_bytes = out_path.stat().st_size
    mcp_fmt = {
        "mp3": "mpeg",
        "wav": "wav",
        "ogg": "ogg",
        "opus": "ogg",
        "m4a": "mp4",
        "mp4": "mp4",
        "flac": "flac",
        "aac": "aac",
    }.get(formato, "mpeg")

    out: Dict[str, Any] = {
        "ok": True,
        "tipo": meta_extra["tipo"],
        "fonte": meta_extra["fonte"],
        "titulo": meta_extra["titulo"],
        "duracao_segundos": round(dur, 2),
        "duracao_falada": formatar_duracao(dur),
        "arquivo_audio": str(out_path),
        "formato": formato,
        "tamanho_mb": round(size_bytes / (1024 * 1024), 2),
        "convertido": convertido,
        "aviso_privacidade": (
            "O audio sera enviado ao modelo de IA desta conversa (a mesma nuvem que voce ja usa). "
            "Para manter o audio apenas no computador, use blindaudio (Whisper local)."
        ),
        "como_ouvir": (
            "Abra arquivo_audio diretamente para ouvir e transcrever sem Whisper (por exemplo com read_file). "
            "Se o modelo desta conversa nao ouvir audio (como o Claude), use blindaudio."
        ),
    }
    if meta_extra.get("canal"):
        out["canal"] = meta_extra["canal"]
    if meta_extra.get("capitulos"):
        out["capitulos"] = meta_extra["capitulos"]

    if incluir_bytes:
        if size_bytes > MAX_INLINE_AUDIO_BYTES:
            raise ValueError(
                "Esse audio passa de 12 MB para entrega direta no MCP. "
                "Use inicio_segundos e fim_segundos para um trecho menor, ou chame blindaudio_listen "
                "com entregar_no_mcp=false e abra arquivo_audio diretamente."
            )
        out["_audio_bytes"] = out_path.read_bytes()
        out["_audio_format"] = mcp_fmt
    return out
