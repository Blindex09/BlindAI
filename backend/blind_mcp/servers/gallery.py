"""Servidor MCP: blindgallery e blindphoto, para galeria acessivel e fotos guiadas por voz.

Ferramentas do BlindPhoto (camera acessivel pela webcam):
- blindphoto_cameras, blindphoto_check_position, blindphoto_capture
Ferramentas do BlindGallery (galeria acessivel, trava por conteudo/impressao digital e busca por descricao):
- blindgallery_scan, blindgallery_list, blindgallery_search, blindgallery_view,
  blindgallery_confirm_save, blindgallery_discard_capture, blindgallery_mark_safe,
  blindgallery_save_note, blindgallery_remove, blindgallery_albums
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from ._common import run_blocking

mcp = FastMCP("blind-gallery")


def _resolve_vision_key() -> str:
    try:
        from config.settings import config

        return (
            getattr(config, "gemini_api_key", "")
            or os.getenv("GEMINI_API_KEY", "")
            or getattr(config, "openai_api_key", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
    except Exception:  # noqa: BLE001
        return os.getenv("GEMINI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")


@mcp.tool()
async def blindphoto_cameras() -> Dict[str, Any]:
    """Lista as câmeras conectadas ao computador para uso com blindphoto_capture e blindphoto_check_position."""
    from core.gallery.camera import list_cameras

    cams = await run_blocking(list_cameras)
    return {
        "ok": True,
        "count": len(cams),
        "cameras": cams,
    }


@mcp.tool()
async def blindphoto_check_position(
    camera_name: str = "",
    stability_mode: str = "normal",
    mirror_horizontal: bool = False,
    speak: bool = True,
) -> Dict[str, Any]:
    """Verifica rapidamente pela webcam onde está o rosto (esquerda, direita, cima, baixo, perto/longe, inclinação e luz) e fala a orientação."""
    from core.gallery.blindgallery import check_position

    return await run_blocking(check_position, camera_name, stability_mode, mirror_horizontal, speak)


@mcp.tool(structured_output=False)
async def blindphoto_capture(
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
    describe_with_vision: bool = False,
) -> List[Any]:
    """Tira uma foto acessível pela câmera com orientação por voz e escolha automática do quadro mais nítido.

    `mode`:
    - 'portrait': retrato/selfie com detecção facial em tempo real (esquerda, direita, cima, baixo, distância, cortes, inclinação e parada estável).
    - 'money': notas e moedas de dinheiro (identifica valores e repete por extenso sem recusar).
    - 'text': rótulos, remédios, receitas, embalagens ou documentos.
    - 'object': identificar e descrever objetos na frente da câmera.
    - 'general': foto geral da cena.
    `stability_mode`: 'normal' (padrão), 'gentle' (mais tolerante para tremores ou movimentos involuntários) ou 'strict'.
    Por segurança, a foto vai primeiro para área temporária sem salvar nada definitivo: depois de conferir que não
    há conteúdo impróprio de sexo ou violência, salve com `blindgallery_confirm_save` (ou descarte com `blindgallery_discard_capture`).
    """
    from mcp.server.fastmcp import Image
    from mcp.types import TextContent

    from core.gallery.blindgallery import capture_photo, confirm_capture_save, discard_capture
    from core.gallery.content_safety import BLOCKED_MESSAGE, SAFETY_REVIEW_PROMPT

    payload, preview_bytes = await run_blocking(
        capture_photo,
        mode,
        question,
        guided,
        stability_mode,
        hold_seconds,
        timeout_seconds,
        speak_guidance,
        audio_cues,
        mirror_horizontal,
        camera_name,
        album,
        save_folder,
    )
    if describe_with_vision:
        try:
            from core.vision.smart_vision_router import smart_vision_router

            safety_prompt = (
                f"{SAFETY_REVIEW_PROMPT}\n"
                "Responda começando a PRIMEIRA linha com exatamente SAFE, UNSAFE_SEXUAL ou UNSAFE_VIOLENCE, "
                "e depois escreva a descrição em português.\n\n"
                + (payload.get("vision_instructions") or "")
            )
            desc = await smart_vision_router.describe_multimodal(
                image_base64=base64.b64encode(preview_bytes).decode("ascii"),
                provider="auto",
                api_key=_resolve_vision_key(),
                prompt_override=safety_prompt,
            )
            raw_desc = (desc or "").strip()
            first_line = raw_desc.splitlines()[0].strip().upper() if raw_desc else ""
            if first_line.startswith("UNSAFE"):
                await run_blocking(discard_capture, payload["pending_id"], "unsafe")
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"ok": False, "blocked": True, "saved": False, "error": BLOCKED_MESSAGE},
                            ensure_ascii=False,
                        ),
                    )
                ]
            if first_line.startswith("SAFE"):
                clean_desc = "\n".join(raw_desc.splitlines()[1:]).strip()
                if not clean_desc:
                    clean_desc = raw_desc[4:].lstrip(":-. ").strip()
                saved = await run_blocking(
                    confirm_capture_save,
                    payload["pending_id"],
                    True,
                    "",
                    clean_desc,
                    payload.get("album", ""),
                    "",
                    False,
                    save_folder,
                )
                payload["saved"] = True
                payload["status"] = "saved"
                payload["safety_review_required"] = False
                payload["item_id"] = saved["id"]
                payload["file_path"] = saved["file_path"]
                payload["filename"] = saved["filename"]
                payload["auto_description"] = clean_desc
            elif raw_desc:
                payload["auto_description"] = raw_desc
        except Exception as exc:  # noqa: BLE001
            payload["auto_description_error"] = str(exc)

    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)),
        Image(data=preview_bytes, format="jpeg").to_image_content(),
    ]


@mcp.tool()
async def blindgallery_scan(
    folder: str = "",
    media_type: str = "all",
    max_files: int = 300,
) -> Dict[str, Any]:
    """Cataloga fotos, vídeos e áudios existentes em Imagens, Vídeos, Músicas e Downloads (ou em `folder`) na galeria acessível sem apagar nem mover arquivos."""
    from core.gallery.blindgallery import scan_gallery

    folders = [folder.strip()] if (folder or "").strip() else None
    return await run_blocking(scan_gallery, folders, media_type, max_files)


@mcp.tool()
async def blindgallery_list(
    media_type: str = "",
    album: str = "",
    favorites_only: bool = False,
    undescribed_only: bool = False,
    limit: int = 20,
    offset: int = 0,
) -> Dict[str, Any]:
    """Lista mídias da galeria acessível (do mais recente ao mais antigo) com título, data, álbum e descrição salva."""
    from core.gallery.blindgallery import list_gallery

    return await run_blocking(list_gallery, media_type, album, favorites_only, undescribed_only, limit, offset)


@mcp.tool()
async def blindgallery_search(
    query: str,
    media_type: str = "",
    album: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    """Pesquisa fotos, vídeos e áudios na galeria pelo conteúdo descrito, título, tags, álbum ou nome do arquivo."""
    from core.gallery.blindgallery import search_gallery

    return await run_blocking(search_gallery, query, media_type, album, limit)


@mcp.tool(structured_output=False)
async def blindgallery_view(
    item: str,
    question: str = "",
    review_flagged: bool = False,
) -> List[Any]:
    """Abre uma foto (ou quadro de vídeo) da galeria pelo id, nome ou caminho para descrever.

    Se o arquivo ficou em espera apenas pelo nome suspeito (falso alarme), passe `review_flagged=true`
    para a IA conferir a imagem real e depois aprovar com `blindgallery_mark_safe` ou bloquear de vez
    com `blindgallery_remove(block_unsafe=true)`. Arquivos já bloqueados nunca abrem mesmo com `review_flagged=true`.
    """
    from mcp.server.fastmcp import Image
    from mcp.types import TextContent

    from core.gallery.blindgallery import prepare_view_image

    payload, preview_bytes = await run_blocking(prepare_view_image, item, question, review_flagged)
    return [
        TextContent(type="text", text=json.dumps(payload, ensure_ascii=False)),
        Image(data=preview_bytes, format="jpeg").to_image_content(),
    ]


@mcp.tool()
async def blindgallery_mark_safe(
    file_path: str,
    safe_verified: bool = False,
    title: str = "",
    description: str = "",
    album: str = "",
    tags: str = "",
    favorite: bool = False,
) -> Dict[str, Any]:
    """Libera um arquivo que ficou em espera pelo nome após conferência visual (`safe_verified=true`) confirmando que é seguro, admitindo com título limpo."""
    from core.gallery.blindgallery import mark_gallery_file_safe

    return await run_blocking(
        mark_gallery_file_safe,
        file_path,
        safe_verified,
        title,
        description,
        album,
        tags,
        favorite,
    )


@mcp.tool()
async def blindgallery_save_note(
    item: str,
    title: str = "",
    description: str = "",
    album: str = "",
    tags: str = "",
    favorite: Optional[bool] = None,
) -> Dict[str, Any]:
    """Salva ou atualiza título, descrição acessível, álbum, tags (separadas por vírgula) ou favorito de uma mídia da galeria."""
    from core.gallery.blindgallery import save_gallery_note

    res = await run_blocking(
        save_gallery_note,
        item,
        title if title != "" else None,
        description if description != "" else None,
        album if album != "" else None,
        tags if tags != "" else None,
        favorite,
    )
    return {"ok": True, "item": res}


@mcp.tool()
async def blindgallery_confirm_save(
    pending_id: str,
    safe_verified: bool = False,
    title: str = "",
    description: str = "",
    album: str = "",
    tags: str = "",
    favorite: bool = False,
    save_folder: str = "",
) -> Dict[str, Any]:
    """Salva na pasta Imagens e na galeria uma captura pendente, somente após confirmar (safe_verified=true) que não há sexo nem violência."""
    from core.gallery.blindgallery import confirm_capture_save

    saved = await run_blocking(
        confirm_capture_save,
        pending_id,
        safe_verified,
        title,
        description,
        album,
        tags,
        favorite,
        save_folder,
    )
    return {"ok": True, "saved": True, "item": saved}


@mcp.tool()
async def blindgallery_discard_capture(
    pending_id: str,
    reason: str = "discarded",
) -> Dict[str, Any]:
    """Descarta imediatamente uma captura temporária sem salvá-la na pasta Imagens nem na galeria (use quando houver conteúdo impróprio ou a pessoa não quiser guardar)."""
    from core.gallery.blindgallery import discard_capture

    return await run_blocking(discard_capture, pending_id, reason)


@mcp.tool()
async def blindgallery_remove(
    item: str,
    block_unsafe: bool = False,
    reason: str = "removed",
) -> Dict[str, Any]:
    """Remove uma mídia do catálogo da galeria (sem apagar o arquivo do disco) e opcionalmente bloqueia para sempre (block_unsafe=true) se for conteúdo impróprio."""
    from core.gallery.blindgallery import remove_gallery_item

    return await run_blocking(remove_gallery_item, item, block_unsafe, reason)


@mcp.tool()
async def blindgallery_albums() -> Dict[str, Any]:
    """Mostra os álbuns e totais da galeria acessível (quantas fotos, vídeos, áudios, favoritos e itens ainda sem descrição)."""
    from core.gallery.blindgallery import list_gallery_albums

    return await run_blocking(list_gallery_albums)


def _preimport_native_libs() -> None:
    """Pre-carrega DLLs pesadas na thread principal antes do loop stdio/anyio.
    Evita travamento de LoadLibrary no Windows + Python 3.14 dentro de asyncio.to_thread."""
    for mod in ("numpy", "onnxruntime", "PIL"):
        try:
            __import__(mod)
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    from . import _audit

    _preimport_native_libs()
    _audit.install_audit(mcp, "blind-gallery")
    mcp.run()
