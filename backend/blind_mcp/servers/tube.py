"""Servidor MCP: blindtube, para pesquisar, resumir, transcrever, baixar e converter videos do YouTube e audios locais.

Ferramentas: blindtube_search, blindtube_summarize, blindtube_transcribe, blindtube_download,
blindtube_convert_audio e blindtube_info.
A pessoa NAO precisa mandar link: quando pedir por nome ou assunto, pesquise com `blindtube_search`,
leia as opcoes para ela escolher e depois resuma, baixe ou converta o escolhido.
Reutiliza as regras de gravacao segura do Blind (Downloads, Documentos, Musicas e Videos) e
o motor de legendas/audio do `blindaudio`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP

from ._common import run_blocking

mcp = FastMCP("blind-tube")


@mcp.tool()
async def blindtube_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Pesquisa videos no YouTube por titulo, assunto ou canal (sem precisar que a pessoa envie link).

    Devolve opcoes numeradas com titulo, canal, duracao falada e url. Leia para a pessoa escolher
    (sem ler os links) e depois use a url escolhida em blindtube_summarize, blindtube_download ou
    blindtube_convert_audio.
    """
    from core.tube.blindtube import search_videos

    return await run_blocking(search_videos, query, max_results)


@mcp.tool()
async def blindtube_summarize(
    url: str,
    idioma: str = "auto",
    continuar: bool = False,
    parte: int = 0,
    modo: str = "auto",
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
    musica: bool = False,
) -> Dict[str, Any]:
    """Busca titulo, canal, duracao, capitulos e o conteudo de um video do YouTube ou de um arquivo de audio/video
    do computador (caminho completo) para resumir.

    A pessoa nao precisa enviar o link: voce pode pesquisar antes com `blindtube_search` ou passar
    uma frase de busca aqui para receber opcoes e perguntar qual ela quer.
    `musica=true` (so arquivo local) desliga o filtro de voz do Whisper: use para musica cantada.
    `modo`:
    - 'auto' (padrao): usa as legendas do YouTube quando houver, ou Whisper small local quando nao houver (ideal para o Claude).
    - 'ouvir': prepara ou baixa so o audio compacto e devolve `arquivo_audio` para modelos que ouvem audio diretamente (como o Gemini) sem usar Whisper.
    - 'whisper': transcreve diretamente pelo audio com Whisper small local.
    Em videos longos, use `continuar=true` ou `parte=2` para ler os proximos trechos.
    """
    from core.tube.blindtube import resumir_video

    return await run_blocking(
        resumir_video,
        url,
        idioma,
        continuar,
        parte,
        modo,
        inicio_segundos,
        fim_segundos,
        musica,
    )


@mcp.tool()
async def blindtube_transcribe(
    fonte: str,
    idioma: str = "auto",
    com_tempos: bool = False,
    continuar: bool = False,
    parte: int = 0,
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
    musica: bool = False,
) -> Dict[str, Any]:
    """Transcreve em texto um arquivo de audio ou video do computador (caminho completo) ou um video do YouTube.

    No YouTube usa as legendas quando houver; senao, e em arquivo local, usa Whisper small local.
    `musica=true` (so arquivo local) desliga o filtro de voz do Whisper: use para musica cantada,
    senao o canto sobre o instrumental pode sair vazio. `com_tempos=true` poe marcas [MM:SS].
    Em arquivos longos, chame de novo com `continuar=true` ou informe `parte=2`.
    """
    from core.tube.blindtube import transcrever_audio

    return await run_blocking(
        transcrever_audio,
        fonte,
        idioma,
        com_tempos,
        continuar,
        parte,
        inicio_segundos,
        fim_segundos,
        musica,
    )


@mcp.tool()
async def blindtube_download(
    url: str,
    qualidade: str = "720p",
    pasta_destino: str = "",
    nome_arquivo: str = "",
) -> Dict[str, Any]:
    """Baixa um video do YouTube para o computador (so quando a pessoa pedir).

    A pessoa nao precisa enviar o link: pesquise antes com `blindtube_search` ou passe o assunto aqui
    para listar opcoes e confirmar qual video baixar antes de iniciar o download.
    `qualidade`: 'melhor', '1080p', '720p' (padrao), '480p' ou '360p'.
    Por padrao salva na pasta Videos da pessoa; aceita apenas Downloads, Documentos, Musicas ou Videos
    e nunca sobrescreve arquivos ja existentes.
    """
    from core.tube.blindtube import baixar_video

    return await run_blocking(baixar_video, url, qualidade, pasta_destino, nome_arquivo)


@mcp.tool()
async def blindtube_convert_audio(
    fonte: str,
    formato: str = "mp3",
    pasta_destino: str = "",
    nome_arquivo: str = "",
    inicio_segundos: Optional[float] = None,
    fim_segundos: Optional[float] = None,
) -> Dict[str, Any]:
    """Baixa ou extrai o audio e converte um video do YouTube ou arquivo local de video/audio para audio.

    A pessoa nao precisa enviar o link: para videos do YouTube, pesquise antes com `blindtube_search`
    ou passe o assunto aqui para listar opcoes e confirmar qual video converter.
    `formato`: 'mp3' (padrao), 'm4a', 'wav', 'ogg', 'opus' ou 'flac'.
    Por padrao salva na pasta Musicas da pessoa; aceita apenas Downloads, Documentos, Musicas ou Videos
    e nunca sobrescreve arquivos ja existentes. Use `inicio_segundos` e `fim_segundos` para extrair so um trecho.
    """
    from core.tube.blindtube import converter_para_audio

    return await run_blocking(
        converter_para_audio,
        fonte,
        formato,
        pasta_destino,
        nome_arquivo,
        inicio_segundos,
        fim_segundos,
    )


@mcp.tool()
async def blindtube_info(url: str) -> Dict[str, Any]:
    """Mostra informacoes rapidas de um video do YouTube antes de resumir, baixar ou converter:
    titulo, canal, duracao falada, visualizacoes, descricao resumida, capitulos, legendas e qualidades disponiveis.
    Se receber texto livre em vez de link, devolve opcoes da busca no YouTube para escolher.
    """
    from core.tube.blindtube import info_video

    return await run_blocking(info_video, url)


def _preimport_native_libs() -> None:
    """Pre-carrega DLLs pesadas na thread principal antes do loop stdio/anyio.
    Evita travamento de LoadLibrary no Windows + Python 3.14 dentro de asyncio.to_thread."""
    for mod in ("numpy", "onnxruntime", "faster_whisper", "yt_dlp"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    from . import _audit

    _preimport_native_libs()
    _audit.install_audit(mcp, "blind-tube")
    mcp.run()
