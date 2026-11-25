---
nome: usando-blindgallery
quando: Ao usar a galeria acessível (blindgallery) ou tirar fotos guiadas pela câmera (blindphoto: retrato, objeto, dinheiro ou texto).
palavras: blindgallery, blindgalery, blindphoto, galeria, câmera, camera, selfie, retrato, foto, enquadramento, rosto, dinheiro, nota, moeda, objeto, album, álbum
---
# Usando blindgallery e blindphoto

As ferramentas ficam em **blind-gallery**: `blindphoto_cameras`, `blindphoto_check_position`, `blindphoto_capture`, `blindgallery_confirm_save`, `blindgallery_discard_capture`, `blindgallery_mark_safe`, `blindgallery_scan`, `blindgallery_list`, `blindgallery_search`, `blindgallery_view`, `blindgallery_save_note`, `blindgallery_remove` e `blindgallery_albums`.

## Regra obrigatória contra sexo ou violência e impressão digital
- **Nunca salve fotos ou mídias impróprias** (nudez, sexo explícito, pornografia ou violência gráfica extrema). Fotos comuns (retratos, dinheiro, remédios, documentos) são permitidas.
- **Toda captura vai primeiro para área temporária**: após `blindphoto_capture`, confira a imagem primeiro. Se houver conteúdo impróprio, chame `blindgallery_discard_capture` (ele também grava a impressão digital do conteúdo para bloquear cópias); se estiver segura, salve com `blindgallery_confirm_save` (`safe_verified=True`).
- **Bloqueio resistente a troca de nome**: `blindgallery_remove` com `block_unsafe=True` grava SHA-256 e dHash da mídia; mesmo se o arquivo mudar de nome ou pasta ou for regravado, continuará bloqueado.
- **Falso alarme pelo nome do arquivo**: se `blindgallery_scan` deixar um arquivo em espera apenas pelo nome suspeito e a pessoa pedir para conferir, abra com `blindgallery_view(..., review_flagged=True)`. Se for inocente, libere com `blindgallery_mark_safe(safe_verified=True, title=...)`; se for impróprio, bloqueie com `blindgallery_remove(block_unsafe=True)`.

## Fotos guiadas e galeria acessível
1. **Use o modo certo**: `portrait` (rosto/selfie com guia esquerda, direita, cima, baixo, perto/longe, cortes, inclinação e parada estável; `stability_mode='gentle'` para tremores), `money`, `text` ou `object`.
2. **Qualidade e verdade**: se a foto sair escura, borrada ou cortada, avise e ofereça tirar outra sem culpar.
3. **Dinheiro, remédio e documento**: leia sempre sem recusar (`descrevendo-imagens` e `dados-sensiveis`); repita valores/doses por extenso e em números separados; nunca identifique rostos.
4. **Anotações limpas**: nunca salve senhas, CPF ou cartões nas notas da galeria.
