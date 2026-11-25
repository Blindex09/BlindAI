"""Utilitarios compartilhados pelos servidores MCP proprios do Blind AI."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Mesmo bootstrap de caminhos do server.py, para os modulos de core/ resolverem imports.
for _p in (BACKEND_DIR, BACKEND_DIR / "core", BACKEND_DIR / "core" / "utils"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _call_with_com(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """UI Automation e SAPI usam COM, que precisa ser inicializado em cada thread do pool."""
    try:
        import pythoncom

        pythoncom.CoInitialize()
    except Exception:  # noqa: BLE001 - sem pywin32 (ex.: testes): segue sem COM
        return fn(*args, **kwargs)
    try:
        return fn(*args, **kwargs)
    finally:
        pythoncom.CoUninitialize()


async def run_blocking(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Executa codigo bloqueante (CLI, UIA, COM) fora do loop para o servidor continuar respondendo."""
    return await asyncio.to_thread(_call_with_com, fn, *args, **kwargs)


def cliente_e_claude() -> bool:
    """True quando o servidor roda atras do proxy do Claude (BLIND_CLIENTE=claude). Nesse caso quem enxerga imagens e o
    proprio Claude: nenhuma imagem sai para modelo de visao externo (Ollama Cloud, Gemini, OpenAI)."""
    return os.environ.get("BLIND_CLIENTE", "").strip().lower() == "claude"


# Respostas do roteador de visao que significam "nenhum modelo de visao descreveu de verdade".
_SEM_VISAO = ("Não consegui usar um modelo de visão", "Não consegui descrever esta imagem: nenhum modelo de visão respondeu")


def sem_modelo_de_visao(descricao: Any) -> bool:
    """True se o roteador de visao nao conseguiu descrever (so metadados ou falha). Nesse caso a imagem vai ao Claude."""
    return isinstance(descricao, str) and descricao.strip().startswith(_SEM_VISAO)


def imagem_para_o_modelo(caminho: str, pergunta: Any = None, contexto: str = "Imagem") -> list:
    """Entrega uma imagem ao proprio modelo da conversa (Claude) para ele descrever, sem chave nem IA externa.
    Reduz para no maximo 1568 px e devolve [texto, imagem]."""
    import io

    from mcp.server.fastmcp import Image
    from mcp.types import TextContent
    from PIL import Image as PILImage

    p = Path(caminho).expanduser()
    if not p.is_file():
        raise ValueError("Não achei esse arquivo de imagem.")
    if p.stat().st_size > 30 * 1024 * 1024:
        raise ValueError("A imagem é grande demais (mais de 30 MB).")
    try:
        with PILImage.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((1568, 1568))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            dados = buf.getvalue()
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Não consegui abrir como imagem: {type(e).__name__}.") from e
    nota = (f"{contexto} entregue para você descrever para uma pessoa cega, com a habilidade descrevendo-imagens. "
            + (f"Pergunta da pessoa: {pergunta}" if pergunta else "Sem pergunta específica: faça a descrição na ordem da habilidade.")
            + " Diga que a descrição é sua e pode errar; não identifique ninguém pelo rosto.")
    return [TextContent(type="text", text=nota), Image(data=dados, format="jpeg").to_image_content()]


def unwrap_bridge(result: Dict[str, Any]) -> Any:
    """agent-desktop devolve {ok, data|error}. Erro vira excecao, que o MCP marca como isError."""
    if result.get("ok"):
        return result.get("data", result)
    err = result.get("error") or {}
    raise RuntimeError(f"{err.get('code', 'ERROR')}: {err.get('message', 'falha desconhecida')}")
