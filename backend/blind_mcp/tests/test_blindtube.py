"""Testes do servidor MCP blind-tube e do motor blindtube."""

from __future__ import annotations

import asyncio
import json
import types
import wave
from pathlib import Path

import httpx
import pytest
import yt_dlp

from blind_mcp import seguranca
from blind_mcp.servers import _audit, blind, tube
from core.audio import blindaudio
from core.tube import blindtube


def _write_wav(path: Path, seconds: float = 2.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))
    return path


# ----------------------------------------------------------------------------- blindtube_info
def test_blindtube_info_metadados_capitulos_qualidades_e_validacao(monkeypatch):
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {
                "id": "dQw4w9WgXcQ",
                "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "title": "Guia Completo de Acessibilidade",
                "uploader": "Canal Acessivel",
                "duration": 625,
                "upload_date": "20261121",
                "view_count": 12345,
                "description": "Descricao detalhada   do video.\n\n\n\nCom varias linhas.",
                "language": "pt",
                "is_live": False,
                "chapters": [
                    {"title": "Abertura", "start_time": 0, "end_time": 65},
                    {"title": "Demonstracao", "start_time": 65, "end_time": 300},
                ],
                "subtitles": {"pt-BR": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=pt-BR"}]},
                "automatic_captions": {"en": [{"ext": "vtt", "url": "https://www.youtube.com/api/timedtext?lang=en"}]},
                "formats": [
                    {"vcodec": "none", "acodec": "mp4a", "height": None},
                    {"vcodec": "avc1", "acodec": "none", "height": 360},
                    {"vcodec": "avc1", "acodec": "none", "height": 720},
                    {"vcodec": "vp9", "acodec": "none", "height": 1080},
                ],
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

    info = asyncio.run(tube.blindtube_info("https://youtu.be/dQw4w9WgXcQ"))
    assert info["ok"] is True
    assert info["titulo"] == "Guia Completo de Acessibilidade"
    assert info["canal"] == "Canal Acessivel"
    assert info["visualizacoes"] == 12345
    assert "Descricao detalhada do video." in info["descricao_resumida"]
    assert info["capitulos"][1]["inicio_formatado"] == "01:05"
    assert info["qualidades_disponiveis"] == ["360p", "720p", "1080p"]
    assert info["tem_legendas"] is True

    with pytest.raises(ValueError, match="não é do YouTube"):
        asyncio.run(tube.blindtube_info("https://vimeo.com/123456"))

    with pytest.raises(ValueError, match="lista de reprodução"):
        asyncio.run(tube.blindtube_info("https://www.youtube.com/playlist?list=PL999"))


# ----------------------------------------------------------------------------- blindtube_summarize
def test_blindtube_summarize_modos_auto_ouvir_e_whisper(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(blindaudio, "MAX_PART_CHARS", 55)

    whisper_calls = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if download:
                target = Path(self.opts["outtmpl"].replace("%(ext)s", "wav"))
                _write_wav(target, seconds=2.0)
                return {"id": "res12345678"}
            return {
                "id": "res12345678",
                "webpage_url": "https://www.youtube.com/watch?v=res12345678",
                "title": "Resumo de Tecnologia",
                "uploader": "Canal Tech",
                "duration": 180,
                "upload_date": "20261122",
                "view_count": 500,
                "description": "Resumo da semana.",
                "language": "pt",
                "is_live": False,
                "chapters": [{"title": "Parte 1", "start_time": 0, "end_time": 90}],
                "subtitles": {"pt-BR": [{"ext": "json3", "url": "https://www.youtube.com/api/timedtext?lang=pt-BR"}]},
                "automatic_captions": {},
                "formats": [{"vcodec": "avc1", "height": 720}],
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
            return FakeResp(
                json.dumps(
                    {
                        "events": [
                            {"tStartMs": 0, "dDurationMs": 20000, "segs": [{"utf8": "Primeiro bloco do resumo de tecnologia para testar."}]},
                            {"tStartMs": 21000, "dDurationMs": 20000, "segs": [{"utf8": "Segundo bloco do resumo com conclusoes finais."}]},
                        ]
                    }
                )
            )

    class FakeModel:
        def transcribe(self, target, **kwargs):
            whisper_calls.append(target)
            return [types.SimpleNamespace(start=0.0, end=1.5, text="Resumo pelo Whisper.")], types.SimpleNamespace(
                language="pt", language_probability=0.96, duration=2.0
            )

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr(blindaudio, "get_whisper_model", lambda model_name=None: FakeModel())

    url = "https://www.youtube.com/watch?v=res12345678"

    # 1. modo='auto': usa legendas e pagina
    r1 = asyncio.run(tube.blindtube_summarize(url))
    assert r1["ok"] and r1["modo"] == "auto" and r1["metodo"] == "legenda_manual"
    assert r1["parte_atual"] == 1 and r1["tem_mais"] is True
    assert "Primeiro bloco" in r1["texto"]
    assert "instrucoes_resumo" in r1 and r1["visualizacoes"] == 500
    assert len(whisper_calls) == 0

    r2 = asyncio.run(tube.blindtube_summarize(url, continuar=True))
    assert r2["parte_atual"] == 2 and "Segundo bloco" in r2["texto"]

    # 2. modo='ouvir': entrega audio compacto sem Whisper
    r_ouvir = asyncio.run(tube.blindtube_summarize(url, modo="ouvir"))
    assert r_ouvir["ok"] and r_ouvir["modo"] == "ouvir"
    assert Path(r_ouvir["arquivo_audio"]).is_file()
    assert len(whisper_calls) == 0

    # 3. modo='whisper': transcreve com Whisper local
    r_w = asyncio.run(tube.blindtube_summarize(url, modo="whisper"))
    assert r_w["ok"] and r_w["modo"] == "whisper" and r_w["metodo"] == "whisper_local"
    assert r_w["texto"] == "Resumo pelo Whisper."
    assert len(whisper_calls) == 1

    with pytest.raises(ValueError, match="Modo invalido"):
        asyncio.run(tube.blindtube_summarize(url, modo="invalido"))


# ----------------------------------------------------------------------------- blindtube_download
def test_blindtube_download_qualidade_seguranca_e_nao_sobrescreve(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(tmp_path))
    state = {"is_live": False, "title": 'Video: Aula <Teste> "01" / Final?'}
    seen_formats = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if download:
                seen_formats.append(self.opts.get("format"))
                target = Path(self.opts["outtmpl"].replace("%(ext)s", "mp4"))
                target.write_bytes(b"ftypmp42" * 512)
                return {"id": "dl123456789"}
            return {
                "id": "dl123456789",
                "webpage_url": url,
                "title": state["title"],
                "uploader": "Canal Download",
                "duration": 95,
                "is_live": state["is_live"],
                "subtitles": {},
                "automatic_captions": {},
                "formats": [{"vcodec": "avc1", "height": 1080}],
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    url = "https://www.youtube.com/watch?v=dl123456789"

    with pytest.raises(ValueError, match="Qualidade invalida"):
        asyncio.run(tube.blindtube_download(url, qualidade="8k", pasta_destino=str(tmp_path)))

    # Bloqueio de seguranca fora das pastas permitidas (ex: C:\Windows)
    with pytest.raises(seguranca.BloqueioDeSeguranca):
        asyncio.run(tube.blindtube_download(url, qualidade="720p", pasta_destino=r"C:\Windows\Temp"))

    # Download 1: sanitiza caracteres invalidos do Windows
    d1 = asyncio.run(tube.blindtube_download(url, qualidade="1080p", pasta_destino=str(tmp_path)))
    assert d1["ok"] is True and d1["qualidade_pedida"] == "1080p"
    p1 = Path(d1["arquivo_salvo"])
    assert p1.is_file() and p1.parent == tmp_path
    assert all(ch not in p1.name for ch in '<>:"/\\|?*')
    assert "height<=1080" in seen_formats[-1]

    # Download 2 mesmo video/nome nao sobrescreve e adiciona (2)
    d2 = asyncio.run(tube.blindtube_download(url, qualidade="melhor", pasta_destino=str(tmp_path)))
    p2 = Path(d2["arquivo_salvo"])
    assert p2.is_file() and p2 != p1 and "(2)" in p2.name
    assert seen_formats[-1].startswith("bestvideo[ext=mp4]")

    # Ao vivo e recusado
    state["is_live"] = True
    with pytest.raises(ValueError, match="ao vivo"):
        asyncio.run(tube.blindtube_download(url, pasta_destino=str(tmp_path)))


# ----------------------------------------------------------------------------- blindtube_convert_audio
def test_blindtube_convert_audio_local_todos_formatos_recorte_e_youtube(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(tmp_path))
    wav_src = _write_wav(tmp_path / "entrada.wav", seconds=3.0)
    out_dir = tmp_path / "convertidos"

    # Todos os formatos suportados com ffmpeg real
    for fmt in ("mp3", "m4a", "ogg", "opus", "wav", "flac"):
        res = asyncio.run(
            tube.blindtube_convert_audio(
                str(wav_src),
                formato=fmt,
                pasta_destino=str(out_dir),
                nome_arquivo=f"saida_{fmt}",
            )
        )
        assert res["ok"] is True and res["formato"] == fmt
        arq = Path(res["arquivo_salvo"])
        assert arq.is_file() and arq.stat().st_size > 0 and arq.suffix == f".{fmt}"

    # Nao sobrescreve arquivo existente e aceita recorte de tempo
    res_dup = asyncio.run(
        tube.blindtube_convert_audio(
            str(wav_src),
            formato="mp3",
            pasta_destino=str(out_dir),
            nome_arquivo="saida_mp3",
            inicio_segundos=0.5,
            fim_segundos=2.0,
        )
    )
    assert res_dup["ok"] is True
    assert "(2).mp3" in res_dup["nome_arquivo"]
    assert res_dup["duracao_segundos"] <= 1.6

    with pytest.raises(ValueError, match="Formato de audio invalido"):
        asyncio.run(tube.blindtube_convert_audio(str(wav_src), formato="wma", pasta_destino=str(out_dir)))

    with pytest.raises(seguranca.BloqueioDeSeguranca):
        asyncio.run(tube.blindtube_convert_audio(str(wav_src), formato="mp3", pasta_destino=r"C:\Windows\Temp"))

    # YouTube -> MP3 (mockando apenas yt_dlp, ffmpeg converte de verdade)
    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            if download:
                target = Path(self.opts["outtmpl"].replace("%(ext)s", "wav"))
                _write_wav(target, seconds=2.0)
                return {"id": "ytAud123456"}
            return {
                "id": "ytAud123456",
                "webpage_url": url,
                "title": "Musica do YouTube",
                "uploader": "Canal Musical",
                "duration": 2.0,
                "is_live": False,
                "subtitles": {},
                "automatic_captions": {},
                "formats": [],
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    yt_res = asyncio.run(
        tube.blindtube_convert_audio(
            "https://www.youtube.com/watch?v=ytAud123456",
            formato="mp3",
            pasta_destino=str(out_dir),
        )
    )
    assert yt_res["ok"] is True and yt_res["tipo"] == "youtube" and yt_res["canal"] == "Canal Musical"
    assert Path(yt_res["arquivo_salvo"]).is_file() and yt_res["formato"] == "mp3"


# ----------------------------------------------------------------------------- blindtube_search e fluxo sem link
def test_blindtube_search_e_fluxo_sem_link(tmp_path, monkeypatch):
    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(tmp_path))
    queries_seen = []
    downloads_seen = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, target, download=False):
            if download:
                downloads_seen.append(target)
                raise AssertionError("Nao deve baixar automaticamente quando recebeu texto livre")
            queries_seen.append(target)
            return {
                "entries": [
                    {
                        "id": "vidOne12345",
                        "title": "Curso de Python Acessivel",
                        "uploader": "Canal Acessibilidade",
                        "duration": 310,
                        "view_count": 4200,
                    },
                    {"id": "curto", "title": "Entrada malformada"},
                    "nao_e_dict",
                    {
                        "id": "vidTwo67890",
                        "title": "Python para Iniciantes",
                        "channel": "Prof Teste",
                        "duration": 605,
                        "view_count": 900,
                    },
                ]
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)

    busca = asyncio.run(tube.blindtube_search("python acessivel", max_results=2))
    assert busca["ok"] is True and busca["count"] == 2
    assert queries_seen[-1] == "ytsearch2:python acessivel"
    assert busca["results"][0]["indice"] == 1
    assert busca["results"][0]["titulo"] == "Curso de Python Acessivel"
    assert busca["results"][0]["canal"] == "Canal Acessibilidade"
    assert busca["results"][0]["duracao_falada"] == "5 minutos e 10 segundos"
    assert busca["results"][0]["url"] == "https://www.youtube.com/watch?v=vidOne12345"
    assert busca["results"][1]["indice"] == 2 and busca["results"][1]["canal"] == "Prof Teste"

    with pytest.raises(ValueError, match="pesquisar no YouTube"):
        asyncio.run(tube.blindtube_search("   "))

    sensivel = asyncio.run(tube.blindtube_search("cpf 123.456.789-09"))
    assert sensivel["ok"] is False and "dado sensível" in sensivel["error"]

    sensivel_senha = asyncio.run(tube.blindtube_search("senha: minha_senha_secreta"))
    assert sensivel_senha["ok"] is False and "dado sensível" in sensivel_senha["error"]

    # Busca com sexo/violencia e bloqueada sem chamar yt_dlp
    qtd_antes = len(queries_seen)
    impropria = asyncio.run(tube.blindtube_search("video de pornografia explicita"))
    assert impropria["ok"] is False and impropria["blocked"] is True and len(queries_seen) == qtd_antes

    # Passar texto livre direto para resumir, baixar, converter ou info devolve opcoes para escolher
    for res in (
        asyncio.run(tube.blindtube_summarize("curso de python acessivel")),
        asyncio.run(tube.blindtube_download("curso de python acessivel", pasta_destino=str(tmp_path))),
        asyncio.run(tube.blindtube_convert_audio("curso de python acessivel", pasta_destino=str(tmp_path))),
        asyncio.run(tube.blindtube_info("curso de python acessivel")),
    ):
        assert res["ok"] is False
        assert res["precisa_escolher"] is True
        assert res["count"] == 2
        assert res["results"][0]["url"] == "https://www.youtube.com/watch?v=vidOne12345"
    assert downloads_seen == []

    # Caminho local inexistente e URL que nao e do YouTube continuam dando erro proprio
    with pytest.raises(FileNotFoundError, match="Nao achei"):
        asyncio.run(tube.blindtube_convert_audio(str(tmp_path / "nao_existe.mp4"), pasta_destino=str(tmp_path)))
    with pytest.raises(ValueError, match="não é do YouTube"):
        asyncio.run(tube.blindtube_download("https://vimeo.com/123456", pasta_destino=str(tmp_path)))


def test_blindtube_bloqueia_idade_18_sexo_e_violencia_antes_de_baixar(tmp_path, monkeypatch):
    from core.gallery.content_safety import ContentSafetyError

    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(tmp_path))

    class UnsafeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, target, download=False):
            if download:
                raise AssertionError("Nunca deve baixar video +18, sexual ou violento")
            if target.startswith("ytsearch"):
                return {
                    "entries": [
                        {"id": "badOne12345", "title": "Gore massacre violento", "uploader": "X", "duration": 60, "age_limit": 0},
                        {"id": "badTwo12345", "title": "Video comum +18", "uploader": "Y", "duration": 60, "age_limit": 18},
                        {"id": "okVid123456", "title": "Aula de violao", "uploader": "Prof", "duration": 180, "age_limit": 0},
                    ]
                }
            return {
                "id": "ytBad123456",
                "webpage_url": "https://www.youtube.com/watch?v=ytBad123456",
                "title": "Conteudo proibido",
                "uploader": "Canal",
                "age_limit": 18,
                "duration": 120,
                "formats": [],
            }

    monkeypatch.setattr(yt_dlp, "YoutubeDL", UnsafeYDL)

    busca = asyncio.run(tube.blindtube_search("violao", max_results=5))
    assert busca["ok"] is True and busca["count"] == 1 and busca["results"][0]["video_id"] == "okVid123456"

    with pytest.raises(ContentSafetyError):
        asyncio.run(tube.blindtube_download("https://www.youtube.com/watch?v=ytBad123456", pasta_destino=str(tmp_path)))
    with pytest.raises(ContentSafetyError):
        asyncio.run(tube.blindtube_convert_audio("https://www.youtube.com/watch?v=ytBad123456", pasta_destino=str(tmp_path)))

    local_bad = tmp_path / "video_porno_amador.mp4"
    local_bad.write_bytes(b"fake")
    with pytest.raises(ContentSafetyError):
        asyncio.run(tube.blindtube_convert_audio(str(local_bad), pasta_destino=str(tmp_path)))


# ----------------------------------------------------------------------------- integracao Blind (comecar e audit)
def test_comecar_habilidade_audit_blindtube():
    r = asyncio.run(blind.comecar("blindtube resumir esse video do youtube"))
    nomes_hab = {h["nome"] for h in r["habilidades_do_pedido"]}
    assert "usando-blindtube" in nomes_hab
    assert set(r["nomes_exatos"]["blind-tube"]) == {
        "mcp__blind-tube__blindtube_search",
        "mcp__blind-tube__blindtube_download",
        "mcp__blind-tube__blindtube_convert_audio",
        "mcp__blind-tube__blindtube_info",
        "mcp__blind-tube__blindtube_summarize",
        "mcp__blind-tube__blindtube_transcribe",
    }
    assert "blindtube" in r["guia"]

    for tool_name in (
        "blindtube_search",
        "blindtube_summarize",
        "blindtube_transcribe",
        "blindtube_download",
        "blindtube_convert_audio",
        "blindtube_info",
    ):
        h = _audit.humanize({"server": "blind-tube", "tool": tool_name, "ok": True, "terminou": True, "ms": 25})
        assert h["origem"] == "BlindTube" and h["estado"] == "deu_certo"


# ----------------------------------------------------------------------------- ffmpeg sem stdin, audio local e musica
def test_ffmpeg_da_conversao_nao_herda_o_stdin_do_servidor(tmp_path, monkeypatch):
    """Regressao: o ffmpeg herdava o stdin do servidor (canal do MCP) e a conversao travava."""
    chamadas = {}

    def fake_run(cmd, **kw):
        chamadas["cmd"] = cmd
        chamadas["kw"] = kw
        Path(cmd[-1]).write_bytes(b"mp3")
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(blindtube.subprocess, "run", fake_run)
    blindtube._executar_ffmpeg_audio("ffmpeg", "entrada.m4a", tmp_path / "saida.mp3", ["-c:a", "libmp3lame"], None, None)

    assert chamadas["kw"]["stdin"] is blindtube.subprocess.DEVNULL
    assert "-nostdin" in chamadas["cmd"]


def _falso_whisper(monkeypatch, usos):
    class FakeModel:
        def transcribe(self, target, **kw):
            usos.append(kw)
            seg = types.SimpleNamespace(start=0.0, end=2.0, text="canto na musica")
            return iter([seg]), types.SimpleNamespace(language="pt", language_probability=0.9, duration=2.0)

    monkeypatch.setattr(blindaudio, "get_whisper_model", lambda model_name=None: FakeModel())


def test_blindtube_transcribe_e_summarize_aceitam_arquivo_local_e_musica(tmp_path, monkeypatch):
    monkeypatch.setattr(blindaudio, "CACHE_DIR", tmp_path / "cache", raising=False)
    usos: list = []
    _falso_whisper(monkeypatch, usos)
    wav = _write_wav(tmp_path / "musica.wav")

    r = asyncio.run(tube.blindtube_transcribe(str(wav), idioma="pt", musica=True))
    assert r["ok"] is True and "canto na musica" in r["texto"]
    assert usos[-1]["vad_filter"] is False

    r2 = asyncio.run(tube.blindtube_transcribe(str(wav), idioma="pt"))
    assert usos[-1]["vad_filter"] is True and "canto na musica" in r2["texto"]

    r3 = asyncio.run(tube.blindtube_summarize(str(wav), idioma="pt", musica=True))
    assert "canto na musica" in r3["texto"] and r3["instrucoes_resumo"]
    with pytest.raises(ValueError, match="so vale para videos do YouTube"):
        asyncio.run(tube.blindtube_summarize(str(wav), modo="ouvir"))


def test_transcrever_recusa_audio_que_nao_cabe_no_limite_e_aceita_trecho(tmp_path, monkeypatch):
    monkeypatch.setattr(blindaudio, "CACHE_DIR", tmp_path / "cache", raising=False)
    usos: list = []
    _falso_whisper(monkeypatch, usos)
    wav = _write_wav(tmp_path / "longo.wav")
    monkeypatch.setattr(blindaudio, "obter_info_local", lambda p: {"duracao_segundos": 1800.0, "tem_audio": True})

    with pytest.raises(ValueError, match="grande demais"):
        asyncio.run(tube.blindtube_transcribe(str(wav)))
    assert usos == []

    r = asyncio.run(tube.blindtube_transcribe(str(wav), inicio_segundos=0, fim_segundos=120))
    assert r["ok"] is True
