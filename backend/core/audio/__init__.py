"""Motor de transcricao de audio local e do YouTube (blindaudio)."""

from .blindaudio import (
    DEFAULT_AUDIO_MODEL,
    classificar_fonte,
    formatar_duracao,
    formatar_tempo,
    get_whisper_model,
    info_fonte,
    obter_info_local,
    obter_info_youtube,
    preparar_audio_para_ouvir,
    transcribe_local_whisper,
    transcrever_fonte,
)

__all__ = [
    "DEFAULT_AUDIO_MODEL",
    "classificar_fonte",
    "formatar_duracao",
    "formatar_tempo",
    "get_whisper_model",
    "info_fonte",
    "obter_info_local",
    "obter_info_youtube",
    "preparar_audio_para_ouvir",
    "transcribe_local_whisper",
    "transcrever_fonte",
]
