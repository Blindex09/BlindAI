"""Servidor MCP: blindaudio, transcricao acessivel de audios locais e videos do YouTube.

Ferramentas: blindaudio, blindaudio_listen e blindaudio_info.
No YouTube, `blindaudio` prioriza legendas existentes (manuais e depois automaticas) e usa
Whisper `small` local (faster-whisper) quando nao ha legenda ou quando `forcar_whisper=true`
(ideal para Claude e modelos que nao ouvem audio).
Para modelos que ouvem audio diretamente (como Gemini), `blindaudio_listen` prepara ou baixa so
o audio compacto (sem Whisper) para leitura direta ou entrega via MCP.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ._common import run_blocking

mcp = FastMCP("blind-audio")


@mcp.tool()
async def blindaudio(
    fonte: str,
    idioma: str = "auto",
    com_tempos: bool = False,
    continuar: bool = False,
    parte: int = 0,
    forcar_whisper: bool = False,
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
) -> Dict[str, Any]:
    """Transcreve audio ou video em texto usando o blindaudio.

    `fonte` aceita caminho de arquivo local de audio ou video no computador (.mp3, .wav, .ogg, .opus,
    .m4a, .mp4, .mkv, .webm) OU um link de video do YouTube.
    No YouTube, usa as legendas do video quando houver; se nao houver (ou com `forcar_whisper=true`),
    baixa so o audio para pasta temporaria e transcreve localmente com Whisper `small` (ideal para
    Claude e modelos que nao ouvem audio diretamente). Modelos que ouvem audio diretamente (como
    Gemini) devem preferir `blindaudio_listen` ou abrir arquivos de audio locais direto sem Whisper.
    Em arquivos longos, devolve por partes: chame de novo com `continuar=true` para ouvir a proxima
    parte, ou informe `parte=2` para ir a uma parte especifica.
    `com_tempos=true` inclui marcas [MM:SS] antes de cada trecho.
    `inicio_segundos` e `fim_segundos` permitem transcrever apenas um trecho.
    """
    from core.audio.blindaudio import transcrever_fonte

    return await run_blocking(
        transcrever_fonte,
        fonte,
        idioma,
        com_tempos,
        continuar,
        parte,
        forcar_whisper,
        inicio_segundos,
        fim_segundos,
    )


@mcp.tool(structured_output=False)
async def blindaudio_listen(
    fonte: str,
    pergunta: Optional[str] = None,
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
    entregar_no_mcp: bool = False,
) -> List[Any]:
    """Prepara ou baixa um audio compacto (de arquivo local, video ou link do YouTube) para que um
    modelo capaz de ouvir audio (como o Gemini) escute e transcreva diretamente, sem usar Whisper.

    Se o modelo desta conversa nao ouvir audio (como o Claude), use `blindaudio`.
    Por padrao devolve os metadados e o caminho `arquivo_audio` (para abrir direto com `read_file`).
    Com `entregar_no_mcp=true`, tambem anexa o audio embutido na resposta MCP (limite 12 MB).
    """
    from mcp.server.fastmcp import Audio
    from mcp.types import TextContent

    from core.audio.blindaudio import preparar_audio_para_ouvir

    res = await run_blocking(
        preparar_audio_para_ouvir,
        fonte,
        inicio_segundos,
        fim_segundos,
        entregar_no_mcp,
    )
    raw_bytes = res.pop("_audio_bytes", None)
    fmt = res.pop("_audio_format", "mpeg")
    if pergunta and pergunta.strip():
        res["pergunta"] = pergunta.strip()

    blocks: List[Any] = [TextContent(type="text", text=json.dumps(res, ensure_ascii=False, indent=2))]
    if entregar_no_mcp and raw_bytes:
        blocks.append(Audio(data=raw_bytes, format=fmt).to_audio_content())
    return blocks


@mcp.tool()
async def blindaudio_info(fonte: str) -> Dict[str, Any]:
    """Mostra informacoes rapidas antes de transcrever: titulo, canal, duracao falada, capitulos e
    legendas disponiveis (no YouTube) ou formato, tamanho e duracao (em arquivo local).
    """
    from core.audio.blindaudio import info_fonte

    return await run_blocking(info_fonte, fonte)


def _preimport_native_libs() -> None:
    """Pre-carrega DLLs pesadas na thread principal antes do loop stdio/anyio.
    No Windows + Python 3.14, o primeiro LoadLibrary de numpy/faster-whisper dentro de
    asyncio.to_thread com o leitor de stdin ativo pode travar."""
    for mod in ("numpy", "onnxruntime", "faster_whisper", "yt_dlp"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    from . import _audit

    _preimport_native_libs()
    _audit.install_audit(mcp, "blind-audio")
    mcp.run()
