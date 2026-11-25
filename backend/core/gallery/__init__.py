"""Pacote central do BlindGallery e BlindPhoto (galeria acessivel, seguranca de conteudo e camera guiada por voz)."""

from .blindgallery import (
    capture_photo,
    check_position,
    confirm_capture_save,
    discard_capture,
    list_cameras,
    list_gallery,
    list_gallery_albums,
    mark_gallery_file_safe,
    prepare_view_image,
    remove_gallery_item,
    resolve_capture_mode,
    save_gallery_note,
    scan_gallery,
    search_gallery,
)
from .content_safety import (
    ContentSafetyError,
    FilenameFlaggedError,
    assert_safe_text,
    check_filename_safety,
    check_text_safety,
    check_youtube_metadata,
)
from .gallery_store import get_item

view_gallery_item = prepare_view_image

__all__ = [
    "ContentSafetyError",
    "FilenameFlaggedError",
    "assert_safe_text",
    "capture_photo",
    "check_filename_safety",
    "check_position",
    "check_text_safety",
    "check_youtube_metadata",
    "confirm_capture_save",
    "discard_capture",
    "get_item",
    "list_cameras",
    "list_gallery",
    "list_gallery_albums",
    "mark_gallery_file_safe",
    "prepare_view_image",
    "remove_gallery_item",
    "resolve_capture_mode",
    "save_gallery_note",
    "scan_gallery",
    "search_gallery",
    "view_gallery_item",
]
