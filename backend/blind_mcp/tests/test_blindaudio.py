"""Testes do servidor MCP blind-audio e do motor blindaudio (local e YouTube)."""

from __future__ import annotations

import asyncio
import json
import types
import wave
from pathlib import Path

import httpx
import pytest
import yt_dlp

from blind_mcp.servers import _audit, audio, blind, channels
from core.audio import blindaudio


def _write_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return path


# ----------------------------------------------------------------------------- classificar_fonte
def test_classificar_fonte_aceita_varios_formatos_youtube_e_rejeita_invalidos(tmp_path):
    vid = "dQw4w9WgXcQ"
    urls = [
        f"https://www.youtube.com/watch?v={vid}&t=30s",
        f"http://youtube.com/watch?v={vid}",
        f"https://youtu.be/{vid}?si=abc123",
        f"https://www.youtube.com/shorts/{vid}",
        f"https://m.youtube.com/live/{vid}?feature=share",
        f"https://music.youtube.com/watch?v={vid}",
        f"https://www.youtube-nocookie.com/embed/{vid}",
        f"https://www.youtube.com/v/{vid}",
        f"youtube.com/watch?v={vid}",
        f"youtu.be/{vid}",
    ]
    for u in urls:
        res = blindaudio.classificar_fonte(u)
        assert res == {"tipo": "youtube", "url": f"https://www.youtube.com/watch?v={vid}", "video_id": vid}

    with pytest.raises(ValueError, match="lista de reprodução"):
        blindaudio.classificar_fonte("https://www.youtube.com/playlist?list=PL123456")

    with pytest.raises(ValueError, match="não é do YouTube"):
        blindaudio.classificar_fonte("https://vimeo.com/123456")

    with pytest.raises(ValueError, match="nao pode comecar com '-'"):
        blindaudio.classificar_fonte("--exec=boom")

    with pytest.raises(ValueError, match="usuario ou senha"):
        blindaudio.classificar_fonte(f"https://user:pass@youtube.com/watch?v={vid}")

    with pytest.raises(ValueError, match="http ou https"):
        blindaudio.classificar_fonte("file:///c:/audio.mp3")

    wav = _write_wav(tmp_path / "fala teste.wav")
    local = blindaudio.classificar_fonte(f'  "{wav}"  ')
    assert local == {"tipo": "local", "path": str(wav.resolve())}

    with pytest.raises(FileNotFoundError, match="Nao achei"):
        blindaudio.classificar_fonte(str(tmp_path / "nao_existe.mp3"))

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    with pytest.raises(ValueError, match="read_document"):
        blindaudio.classificar_fonte(str(pdf))

    jpg = tmp_path / "foto.jpg"
    jpg.write_bytes(b"\xff\xd8\xff")
    with pytest.raises(ValueError, match="blindimage"):
        blindaudio.classificar_fonte(str(jpg))


# ----------------------------------------------------------------------------- parsers de legenda e selecao
def test_parsers_json3_e_vtt_limpam_tags_e_duplicacoes():
    json3_raw = json.dumps(
        {
            "events": [
                {"tStartMs": 0, "dDurationMs": 2000, "segs": [{"utf8": "Olá &amp; "}, {"utf8": "bem-vindos."}]},
                {"tStartMs": 2000, "dDurationMs": 2000, "segs": [{"utf8": "Hoje vamos testar.\n"}]},
            ]
        }
    )
    segs_json = blindaudio._parse_json3(json3_raw)
    assert len(segs_json) == 1
    assert segs_json[0]["texto"] == "Olá & bem-vindos. Hoje vamos testar."

    vtt_raw = """WEBVTT

00:00:00.000 --> 00:00:02.000
Primeira linha

00:00:02.000 --> 00:00:04.000
Primeira linha
Segunda <00:00:02.500><c>linha</c> com &quot;aspas&quot;

00:00:04.000 --> 00:00:06.000
Segunda linha com &quot;aspas&quot;
Terceira linha.
"""
    segs_vtt = blindaudio._parse_vtt(vtt_raw)
    texto = " ".join(s["texto"] for s in segs_vtt)
    assert texto.count("Primeira linha") == 1
    assert texto.count('Segunda linha com "aspas"') == 1
    assert "Terceira linha." in texto


def test_escolher_legenda_prioriza_manual_e_original_sobre_traducao():
    subs = {
        "en": [{"ext": "vtt", "url": "https://www.youtube.com/api/timedtext?lang=en"}],
    }
    autos = {
        "pt": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=en&tlang=pt"}],
        "en-orig": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=en"}],
    }

    # auto com manual em en -> pega manual
    chosen = blindaudio._escolher_legenda(subs, autos, idioma_pedido="auto", idioma_video="en")
    assert chosen is not None and chosen[0] == "en" and chosen[3] == "legenda_manual"

    # sem manual, auto prefere -orig em vez de tlang=pt
    chosen_orig = blindaudio._escolher_legenda({}, autos, idioma_pedido="auto", idioma_video="en")
    assert chosen_orig is not None and chosen_orig[0] == "en-orig" and chosen_orig[3] == "legenda_automatica"

    # pedido explicito em pt -> usa faixa pt mesmo com tlang= quando nao ha outra
    chosen_pt = blindaudio._escolher_legenda({}, autos, idioma_pedido="pt", idioma_video="en")
    assert chosen_pt is not None and chosen_pt[0] == "pt"

    # match de prefixo pt -> pt-BR antes de pt-PT
    subs_pt = {
        "pt-PT": [{"ext": "vtt", "url": "https://www.youtube.com/api/timedtext?lang=pt-PT"}],
        "pt-BR": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=pt-BR"}],
    }
    chosen_br = blindaudio._escolher_legenda(subs_pt, {}, idioma_pedido="pt", idioma_video="pt")
    assert chosen_br is not None and chosen_br[0] == "pt-BR" and chosen_br[1] == "json3"


# ----------------------------------------------------------------------------- arquivo local (info + blindaudio)
def test_blindaudio_local_info_transcricao_tempos_e_intervalo(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))
    wav = _write_wav(tmp_path / "nota.wav", seconds=4.0)

    info = asyncio.run(audio.blindaudio_info(str(wav)))
    assert info["ok"] and info["tipo"] == "local" and info["arquivo"] == "nota.wav"
    assert info["duracao_segundos"] >= 3.9 and info["tem_audio"] is True

    calls = []

    class FakeModel:
        def transcribe(self, target, **kwargs):
            calls.append((target, kwargs))
            segs = [
                types.SimpleNamespace(start=0.5, end=2.0, text=" Primeiro trecho."),
                types.SimpleNamespace(start=2.5, end=3.8, text=" Segundo trecho."),
            ]
            return segs, types.SimpleNamespace(language="pt", language_probability=0.98, duration=4.0)

    monkeypatch.setattr(blindaudio, "get_whisper_model", lambda model_name=None: FakeModel())

    r = asyncio.run(audio.blindaudio(str(wav), com_tempos=True))
    assert r["ok"] and r["tipo"] == "local" and r["metodo"] == "whisper_local"
    assert "[00:00] Primeiro trecho." in r["texto"]
    assert "[00:02] Segundo trecho." in r["texto"]
    assert r["total_palavras"] == 4
    assert len(calls) == 1

    # Nova chamada com mesmo arquivo vem do cache (nao chama Whisper de novo)
    r_cache = asyncio.run(audio.blindaudio(str(wav), com_tempos=False))
    assert "Primeiro trecho. Segundo trecho." == r_cache["texto"]
    assert len(calls) == 1

    # Teste com recorte de tempo usando segment filtering quando ffmpeg nao recorta ou recorta
    monkeypatch.setattr(blindaudio, "_recortar_com_ffmpeg", lambda *a, **kw: None)
    r_slice = asyncio.run(audio.blindaudio(str(wav), inicio_segundos=2.1, fim_segundos=4.0))
    assert r_slice["texto"] == "Segundo trecho."
    assert len(calls) == 2


# ----------------------------------------------------------------------------- YouTube legendas, paginacao e fallback Whisper
def test_blindaudio_youtube_legendas_capitulos_e_continuacao(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(blindaudio, "MAX_PART_CHARS", 60)

    extract_calls = []
    cap_downloads = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            extract_calls.append((url, download))
            return {
                "id": "dQw4w9WgXcQ",
                "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "title": "Aula Acessivel de Python",
                "uploader": "Canal Teste",
                "duration": 185,
                "upload_date": "20261120",
                "language": "pt",
                "is_live": False,
                "chapters": [{"title": "Introducao", "start_time": 0, "end_time": 60}],
                "subtitles": {
                    "pt-BR": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=pt-BR"}]
                },
                "automatic_captions": {},
            }

    class FakeResp:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url):
            cap_downloads.append(url)
            return FakeResp(
                json.dumps(
                    {
                        "events": [
                            {"tStartMs": 0, "dDurationMs": 25000, "segs": [{"utf8": "Primeira parte bem longa da aula de Python."}]},
                            {"tStartMs": 26000, "dDurationMs": 25000, "segs": [{"utf8": "Segunda parte com mais conteudo interessante."}]},
                            {"tStartMs": 52000, "dDurationMs": 25000, "segs": [{"utf8": "Terceira e ultima parte do video."}]},
                        ]
                    }
                )
            )

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(httpx, "Client", FakeClient)

    vid_url = "https://youtu.be/dQw4w9WgXcQ"
    info = asyncio.run(audio.blindaudio_info(vid_url))
    assert info["ok"] and info["titulo"] == "Aula Acessivel de Python"
    assert info["canal"] == "Canal Teste"
    assert "_subtitles" not in info and "_automatic_captions" not in info
    assert info["capitulos"][0]["titulo"] == "Introducao"

    p1 = asyncio.run(audio.blindaudio(vid_url))
    assert p1["ok"] and p1["metodo"] == "legenda_manual"
    assert p1["parte_atual"] == 1 and p1["total_partes"] >= 2 and p1["tem_mais"] is True
    assert "Primeira parte" in p1["texto"]
    assert len(cap_downloads) == 1

    # continuar=True deve ler do cache sem baixar novamente
    p2 = asyncio.run(audio.blindaudio(vid_url, continuar=True))
    assert p2["parte_atual"] == 2
    assert "Segunda parte" in p2["texto"]
    assert len(cap_downloads) == 1

    # avanca ate o fim
    cur = p2
    while cur.get("tem_mais"):
        cur = asyncio.run(audio.blindaudio(vid_url, continuar=True))
    fim = asyncio.run(audio.blindaudio(vid_url, continuar=True))
    assert fim.get("fim_da_transcricao") is True

    # parte especifica funciona direto do cache
    p_um = asyncio.run(audio.blindaudio(vid_url, parte=1))
    assert p_um["parte_atual"] == 1 and "Primeira parte" in p_um["texto"]


def test_blindaudio_youtube_fallback_whisper_e_travas(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))

    state = {"duration": 120, "is_live": False, "has_sub": True}

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if download:
                outtmpl = self.opts["outtmpl"]
                target = Path(outtmpl.replace("%(ext)s", "wav"))
                _write_wav(target, seconds=2.0)
                return {"id": "abc12345678"}
            return {
                "id": "abc12345678",
                "webpage_url": url,
                "title": "Video Sem Legenda",
                "uploader": "Canal Sem Legenda",
                "duration": state["duration"],
                "language": "pt",
                "is_live": state["is_live"],
                "subtitles": {"pt": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=pt"}]}
                if state["has_sub"]
                else {},
                "automatic_captions": {},
            }

    class FakeModel:
        def transcribe(self, target, **kwargs):
            return [types.SimpleNamespace(start=0.0, end=1.5, text="Transcrito pelo Whisper.")], types.SimpleNamespace(
                language="pt", language_probability=0.95, duration=2.0
            )

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(blindaudio, "get_whisper_model", lambda model_name=None: FakeModel())

    # Video sem legenda cai para Whisper local
    state["has_sub"] = False
    res = asyncio.run(audio.blindaudio("https://www.youtube.com/watch?v=abc12345678"))
    assert res["ok"] and res["metodo"] == "whisper_local"
    assert res["texto"] == "Transcrito pelo Whisper."

    # Video com legenda mas forcar_whisper=True tambem usa Whisper local
    state["has_sub"] = True
    res_forced = asyncio.run(audio.blindaudio("https://www.youtube.com/watch?v=abc12345678", forcar_whisper=True))
    assert res_forced["ok"] and res_forced["metodo"] == "whisper_local"

    # Video com mais de 1h sem legenda e sem recorte e recusado com orientacao clara
    state["has_sub"] = False
    state["duration"] = 5400
    with pytest.raises(ValueError, match="inicio_segundos"):
        asyncio.run(audio.blindaudio("https://www.youtube.com/watch?v=long1234567"))

    # Video ao vivo sem legenda e recusado
    state["is_live"] = True
    state["duration"] = 0
    with pytest.raises(ValueError, match="ao vivo"):
        asyncio.run(audio.blindaudio("https://www.youtube.com/watch?v=live1234567"))


# ----------------------------------------------------------------------------- modelo small, fallback offline e blindaudio_listen
def test_blindaudio_usa_small_por_padrao_e_canais_mantem_base(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BLIND_AUDIO_MODEL", raising=False)
    monkeypatch.delenv("BLIND_STT_MODEL", raising=False)
    assert blindaudio.DEFAULT_AUDIO_MODEL == "small"

    modelos_pedidos = []

    class FakeModel:
        def transcribe(self, target, **kwargs):
            return [types.SimpleNamespace(start=0.0, end=1.0, text="Teste de modelo.")], types.SimpleNamespace(
                language="pt", language_probability=0.97, duration=1.5
            )

    def fake_get_model(model_name=None):
        modelos_pedidos.append(model_name)
        return FakeModel()

    monkeypatch.setattr(blindaudio, "get_whisper_model", fake_get_model)
    wav = _write_wav(tmp_path / "modelo.wav", seconds=2.0)

    r = asyncio.run(audio.blindaudio(str(wav)))
    assert modelos_pedidos[-1] == "small"
    assert r["modelo_whisper"] == "small"
    assert "small" in r["aviso_precisao"]

    ch_res = channels._transcribe_file_sync(str(wav), "pt")
    assert ch_res["ok"] is True
    assert modelos_pedidos[-1] == "base"


def test_blindaudio_fallback_offline_para_base_se_small_falhar(tmp_path, monkeypatch):
    import faster_whisper

    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("BLIND_AUDIO_MODEL", raising=False)
    monkeypatch.setattr(blindaudio, "_WHISPER_MODELS", {})
    monkeypatch.setattr(blindaudio, "_MODEL_ACTUAL_NAMES", {})

    attempts = []

    class FakeWhisperModel:
        def __init__(self, name, **kwargs):
            attempts.append((name, kwargs.get("local_files_only", False)))
            if name == "small":
                raise RuntimeError("sem rede e sem cache local de small")
            self.name = name

        def transcribe(self, target, **kwargs):
            return [types.SimpleNamespace(start=0.0, end=1.0, text="Fallback base.")], types.SimpleNamespace(
                language="pt", language_probability=0.91, duration=1.5
            )

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    wav = _write_wav(tmp_path / "fallback.wav", seconds=2.0)

    r = asyncio.run(audio.blindaudio(str(wav)))
    assert r["ok"] and r["modelo_whisper"] == "base" and "base" in r["aviso_precisao"]
    assert r["texto"] == "Fallback base."
    assert ("small", True) in attempts and ("small", False) in attempts and ("base", True) in attempts


def test_blindaudio_listen_local_direto_mcp_e_recorte_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))
    wav = _write_wav(tmp_path / "direto.wav", seconds=3.0)

    # 1. Padrao (entregar_no_mcp=False): 1 bloco TextContent apontando direto para o WAV original
    blocks = asyncio.run(audio.blindaudio_listen(str(wav), pergunta="O que foi dito?"))
    assert len(blocks) == 1 and blocks[0].type == "text"
    meta = json.loads(blocks[0].text)
    assert meta["ok"] is True
    assert meta["tipo"] == "local"
    assert meta["arquivo_audio"] == str(wav.resolve())
    assert meta["convertido"] is False
    assert meta["pergunta"] == "O que foi dito?"

    # 2. Com entregar_no_mcp=True: devolve [TextContent, AudioContent]
    blocks_mcp = asyncio.run(audio.blindaudio_listen(str(wav), entregar_no_mcp=True))
    assert len(blocks_mcp) == 2
    assert blocks_mcp[0].type == "text"
    assert blocks_mcp[1].type == "audio" and blocks_mcp[1].mimeType == "audio/wav"

    # 3. Com recorte de tempo: converte para mp3 compacto em BLIND_DATA_DIR/blindaudio/audios/ via ffmpeg
    blocks_slice = asyncio.run(audio.blindaudio_listen(str(wav), inicio_segundos=0.5, fim_segundos=2.0))
    assert len(blocks_slice) == 1
    meta_slice = json.loads(blocks_slice[0].text)
    assert meta_slice["ok"] is True
    assert meta_slice["convertido"] is True
    assert meta_slice["formato"] == "mp3"
    out_mp3 = Path(meta_slice["arquivo_audio"])
    assert out_mp3.is_file() and out_mp3.stat().st_size > 0
    assert (tmp_path / "blindaudio" / "audios") in out_mp3.parents


def test_blindaudio_listen_youtube_cache_e_travas(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))
    state = {"duration": 90, "is_live": False}
    downloads = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if download:
                downloads.append(url)
                outtmpl = self.opts["outtmpl"]
                target = Path(outtmpl.replace("%(ext)s", "wav"))
                _write_wav(target, seconds=2.0)
                return {"id": "ytOuvir1234"}
            return {
                "id": "ytOuvir1234",
                "webpage_url": url,
                "title": "Podcast Acessivel",
                "uploader": "Canal Audio",
                "duration": state["duration"],
                "language": "pt",
                "is_live": state["is_live"],
                "chapters": [{"title": "Parte 1", "start_time": 0, "end_time": 45}],
                "subtitles": {},
                "automatic_captions": {},
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

    yt_url = "https://www.youtube.com/watch?v=ytOuvir1234"
    b1 = asyncio.run(audio.blindaudio_listen(yt_url))
    assert len(b1) == 1
    m1 = json.loads(b1[0].text)
    assert m1["ok"] is True and m1["tipo"] == "youtube" and m1["titulo"] == "Podcast Acessivel"
    assert m1["formato"] == "mp3" and Path(m1["arquivo_audio"]).is_file()
    assert len(downloads) == 1

    # Segunda chamada reutiliza o arquivo em cache sem baixar novamente
    b2 = asyncio.run(audio.blindaudio_listen(yt_url))
    m2 = json.loads(b2[0].text)
    assert m2["arquivo_audio"] == m1["arquivo_audio"]
    assert len(downloads) == 1

    # Video >1h sem recorte e recusado
    state["duration"] = 5000
    with pytest.raises(ValueError, match="inicio_segundos"):
        asyncio.run(audio.blindaudio_listen("https://www.youtube.com/watch?v=ytLong12345"))

    # Transmissao ao vivo e recusada
    state["is_live"] = True
    state["duration"] = 0
    with pytest.raises(ValueError, match="ao vivo"):
        asyncio.run(audio.blindaudio_listen("https://www.youtube.com/watch?v=ytLive12345"))


# ----------------------------------------------------------------------------- integracao Blind (comecar, audit, channels)
def test_comecar_habilidade_audit_e_delegacao_channels(tmp_path, monkeypatch):
    r = asyncio.run(blind.comecar("blindaudio transcreva esse video do youtube"))
    nomes_hab = {h["nome"] for h in r["habilidades_do_pedido"]}
    assert "transcrevendo-audios" in nomes_hab
    assert "mcp__blind-audio__blindaudio" in r["nomes_exatos"]["blind-audio"]
    assert "mcp__blind-audio__blindaudio_listen" in r["nomes_exatos"]["blind-audio"]
    assert "mcp__blind-audio__blindaudio_info" in r["nomes_exatos"]["blind-audio"]
    assert "blindaudio" in r["guia"]

    h = _audit.humanize({"server": "blind-audio", "tool": "blindaudio", "ok": True, "terminou": True, "ms": 42})
    assert h["origem"] == "BlindAudio" and "Transcrevi um audio ou video: deu certo" in h["frase"]

    h_ouvir = _audit.humanize({"server": "blind-audio", "tool": "blindaudio_listen", "ok": True, "terminou": True, "ms": 15})
    assert h_ouvir["origem"] == "BlindAudio" and "Preparei um audio para o modelo ouvir: deu certo" in h_ouvir["frase"]

    h_info = _audit.humanize({"server": "blind-audio", "tool": "blindaudio_info", "ok": True, "terminou": True, "ms": 10})
    assert "Consultei informacoes de um audio ou video: deu certo" in h_info["frase"]

    # Verifica que channels._transcribe_file_sync continua retornando as mesmas chaves via motor compartilhado
    class FakeModel:
        def transcribe(self, target, **kwargs):
            return [types.SimpleNamespace(start=0.0, end=1.0, text="Oi canal")], types.SimpleNamespace(
                language="pt", language_probability=0.99, duration=1.2
            )

    monkeypatch.setattr(blindaudio, "get_whisper_model", lambda model_name=None: FakeModel())
    wav = _write_wav(tmp_path / "voz.wav")
    ch_res = channels._transcribe_file_sync(str(wav), "pt")
    assert ch_res == {"ok": True, "text": "Oi canal", "language": "pt", "duration_seconds": 1.2}
