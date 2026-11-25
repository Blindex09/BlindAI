"""Correcoes de 25/11/2026: titulo duplicado, pasta de destino por nome, visao entregue ao Claude."""

from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from blind_mcp.servers import _common
from core.tube import blindtube
from core.writer import office_io


def test_sem_titulo_repetido_tira_so_a_primeira_linha_igual():
    assert office_io.sem_titulo_repetido("Teste", "# Teste\n\nCorpo") == "Corpo"
    assert office_io.sem_titulo_repetido("Teste", "# teste\n\nCorpo") == "Corpo"  # sem diferenca de maiuscula
    assert office_io.sem_titulo_repetido("Teste", "# Outro\n\nCorpo") == "# Outro\n\nCorpo"
    assert office_io.sem_titulo_repetido("", "# Teste\n\nCorpo") == "# Teste\n\nCorpo"


def test_docx_nao_repete_titulo_e_secao_vira_titulo(tmp_path, monkeypatch):
    from docx import Document

    from core.writer.smart_writer import writer

    monkeypatch.setenv("BLIND_PASTAS_EXTRAS", str(tmp_path))
    alvo = tmp_path / "t.docx"
    writer.save_document({"target_path": str(alvo), "filename": "t.docx", "title": "Teste",
                          "content": "# Teste\n\nTexto.\n\n# Secao\n\nOutro."})
    doc = Document(str(alvo))
    textos = [p.text for p in doc.paragraphs]
    assert textos == ["Teste", "Texto.", "Secao", "Outro."]
    assert [p.style.name for p in doc.paragraphs][:1] == ["Heading 1"]
    assert doc.paragraphs[2].style.name == "Heading 2"


def test_pasta_de_destino_por_nome_solto():
    assert blindtube._resolver_pasta_destino("Downloads", "Music") == blindtube._pasta_padrao("Downloads")
    assert blindtube._resolver_pasta_destino("músicas", "Videos") == blindtube._pasta_padrao("Music")
    assert blindtube._resolver_pasta_destino("", "Videos") == blindtube._pasta_padrao("Videos")
    assert blindtube._resolver_pasta_destino(r"C:\x\y", "Music") == Path(r"C:\x\y")
    assert blindtube._resolver_pasta_destino("minha-pasta", "Music") == Path("minha-pasta")


def test_ytdlp_nao_escreve_no_stdout_do_servidor_mcp():
    from core.audio import blindaudio

    opts = blindaudio._ydl_base_opts()
    assert opts["noprogress"] is True and opts["quiet"] is True
    assert opts["logger"].info("x") is None  # o logger mudo engole as mensagens (stdout e o canal do protocolo)


def test_sem_modelo_de_visao_reconhece_as_duas_falhas():
    assert _common.sem_modelo_de_visao("Não consegui usar um modelo de visão. Só sei dizer: PNG")
    assert _common.sem_modelo_de_visao("Não consegui descrever esta imagem: nenhum modelo de visão respondeu.")
    assert not _common.sem_modelo_de_visao("Uma tela com um botão azul.")


def test_imagem_para_o_modelo_devolve_texto_e_imagem(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (40, 20), "red").save(img)
    saida = _common.imagem_para_o_modelo(str(img), "O que ha?")
    assert saida[0].type == "text" and "O que ha?" in saida[0].text
    assert saida[1].type == "image" and saida[1].mimeType == "image/jpeg"


def test_no_claude_nenhuma_imagem_vai_para_ia_externa(tmp_path, monkeypatch):
    from blind_mcp.servers import docs

    img = tmp_path / "c.png"
    Image.new("RGB", (30, 30), "green").save(img)
    monkeypatch.setenv("BLIND_CLIENTE", "claude")

    async def proibido(**kwargs):
        raise AssertionError("o roteador de visao externo nao pode ser chamado no Claude")

    monkeypatch.setattr(docs.smart_vision_router, "describe_multimodal", proibido)
    saida = asyncio.run(docs.describe_image(file_path=str(img)))
    assert isinstance(saida, list) and saida[1].type == "image"


def test_describe_image_sem_visao_entrega_a_imagem_ao_claude(tmp_path, monkeypatch):
    from blind_mcp.servers import docs

    img = tmp_path / "b.png"
    Image.new("RGB", (30, 30), "blue").save(img)

    async def sem_visao(**kwargs):
        return "Não consegui usar um modelo de visão. Só sei dizer, de forma bem limitada: PNG"

    monkeypatch.setattr(docs.smart_vision_router, "describe_multimodal", sem_visao)
    saida = asyncio.run(docs.describe_image(file_path=str(img)))
    assert isinstance(saida, list) and saida[1].type == "image"

    async def com_visao(**kwargs):
        return "Um quadrado azul."

    monkeypatch.setattr(docs.smart_vision_router, "describe_multimodal", com_visao)
    assert asyncio.run(docs.describe_image(file_path=str(img))) == "Um quadrado azul."
