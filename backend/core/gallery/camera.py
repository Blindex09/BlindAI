"""Descoberta de webcams DirectShow via ffmpeg, captura em stream sem atraso e orientador por voz SAPI."""

from __future__ import annotations

import os
import queue
import re
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DSHOW_DEV_RE = re.compile(r'\[(?:dshow|in#\d+)[^\]]*\]\s*"([^"]+)"\s+\((video|audio|none)\)', re.I)
_DSHOW_ALT_RE = re.compile(r'\[(?:dshow|in#\d+)[^\]]*\]\s*Alternative name\s+"([^"]+)"', re.I)


def list_cameras(timeout: float = 8.0) -> List[Dict[str, Any]]:
    """Lista webcams disponíveis no Windows usando ffmpeg DirectShow."""
    cmd = ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_NO_WINDOW,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except FileNotFoundError as exc:
        raise RuntimeError("Não encontrei o ffmpeg no sistema para listar ou abrir a câmera.") from exc
    except subprocess.TimeoutExpired:
        return []

    cameras: List[Dict[str, Any]] = []
    last_was_video = False
    for line in output.splitlines():
        m_dev = _DSHOW_DEV_RE.search(line)
        if m_dev:
            name, kind = m_dev.group(1), m_dev.group(2).lower()
            if kind == "video":
                cameras.append(
                    {
                        "index": len(cameras),
                        "name": name,
                        "alternative_name": "",
                        "default": len(cameras) == 0,
                    }
                )
                last_was_video = True
            else:
                last_was_video = False
            continue
        m_alt = _DSHOW_ALT_RE.search(line)
        if m_alt and last_was_video and cameras:
            cameras[-1]["alternative_name"] = m_alt.group(1)
            last_was_video = False
    return cameras


def resolve_camera_name(camera_name: str = "") -> str:
    """Resolve o nome exato da câmera (padrão: primeira webcam detectada)."""
    cams = list_cameras()
    if not cams:
        raise RuntimeError(
            "Nenhuma câmera foi encontrada no computador. Confira se a webcam está ligada ou sem bloqueio de privacidade."
        )
    wanted = (camera_name or "").strip().lower()
    if not wanted:
        return cams[0]["name"]
    for cam in cams:
        if cam["name"].lower() == wanted:
            return cam["name"]
    for cam in cams:
        if wanted in cam["name"].lower():
            return cam["name"]
    if wanted.isdigit():
        idx = int(wanted)
        if 0 <= idx < len(cams):
            return cams[idx]["name"]
    disponiveis = ", ".join(c["name"] for c in cams)
    raise ValueError(f"Não encontrei a câmera '{camera_name}'. Câmeras disponíveis: {disponiveis}.")


class CameraStream:
    """Lê quadros RGB em segundo plano via ffmpeg DirectShow mantendo sempre apenas o quadro mais recente."""

    def __init__(self, camera_name: str = "") -> None:
        self.requested_name = camera_name
        self.camera_name: str = ""
        self.width: int = 0
        self.height: int = 0
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: Optional[Tuple[int, float, np.ndarray]] = None
        self._frame_id: int = 0
        self._stderr_tail: str = ""

    def _profiles(self) -> List[Tuple[int, int, List[str]]]:
        return [
            (1280, 720, ["-framerate", "30", "-vcodec", "mjpeg", "-video_size", "1280x720"]),
            (640, 480, ["-framerate", "30", "-vcodec", "mjpeg", "-video_size", "640x480"]),
            (640, 480, ["-video_size", "640x480"]),
            (640, 480, []),
        ]

    def _spawn(self, width: int, height: int, input_opts: List[str]) -> bool:
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "dshow",
            *input_opts,
            "-i",
            f"video={self.camera_name}",
            "-vf",
            f"fps=15,scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        self._stop.clear()
        self._latest = None
        self._frame_id = 0
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=width * height * 3 * 2,
            creationflags=_NO_WINDOW,
        )
        self._proc = proc
        self.width = width
        self.height = height
        self._thread = threading.Thread(target=self._reader_loop, args=(proc, width, height), daemon=True)
        self._thread.start()

        deadline = time.monotonic() + 5.0
        with self._cond:
            while self._frame_id == 0 and not self._stop.is_set() and proc.poll() is None:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                self._cond.wait(timeout=rem)
            ok = self._frame_id > 0
        if ok:
            return True
        try:
            self._stderr_tail = (proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else "")[:300]
        except Exception:  # noqa: BLE001
            pass
        self.close()
        return False

    def _reader_loop(self, proc: subprocess.Popen[bytes], width: int, height: int) -> None:
        frame_bytes = width * height * 3
        stdout = proc.stdout
        if stdout is None:
            return
        try:
            while not self._stop.is_set():
                chunks: List[bytes] = []
                got = 0
                while got < frame_bytes and not self._stop.is_set():
                    part = stdout.read(frame_bytes - got)
                    if not part:
                        return
                    chunks.append(part)
                    got += len(part)
                if got < frame_bytes or self._stop.is_set():
                    return
                buf = b"".join(chunks) if len(chunks) > 1 else chunks[0]
                arr = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy()
                now = time.monotonic()
                with self._cond:
                    self._frame_id += 1
                    self._latest = (self._frame_id, now, arr)
                    self._cond.notify_all()
        except Exception:  # noqa: BLE001
            pass
        finally:
            with self._cond:
                self._cond.notify_all()

    def open(self) -> "CameraStream":
        self.camera_name = resolve_camera_name(self.requested_name)
        for w, h, opts in self._profiles():
            if self._spawn(w, h, opts):
                return self
        detalhe = f" ({self._stderr_tail.strip()})" if self._stderr_tail.strip() else ""
        raise RuntimeError(
            f"Não consegui abrir a câmera {self.camera_name}. Confira se ela não está ocupada por outro programa{detalhe}."
        )

    def wait_ready(self, warmup_frames: int = 5, timeout: float = 5.0) -> Optional[Tuple[int, float, np.ndarray]]:
        """Descarta os primeiros quadros para estabilizar foco e exposição automática."""
        deadline = time.monotonic() + max(0.5, float(timeout))
        with self._cond:
            while self._frame_id < warmup_frames and not self._stop.is_set():
                if self._proc is not None and self._proc.poll() is not None:
                    break
                rem = deadline - time.monotonic()
                if rem <= 0:
                    break
                self._cond.wait(timeout=rem)
            return self._latest

    def get_latest(self, last_id: int = -1, timeout: float = 1.5) -> Optional[Tuple[int, float, np.ndarray]]:
        """Devolve o quadro mais novo (id, timestamp, rgb) ou o atual após timeout."""
        deadline = time.monotonic() + max(0.05, float(timeout))
        with self._cond:
            while not self._stop.is_set():
                if self._latest is not None and self._latest[0] > last_id:
                    return self._latest
                if self._proc is not None and self._proc.poll() is not None:
                    return self._latest
                rem = deadline - time.monotonic()
                if rem <= 0:
                    return self._latest
                self._cond.wait(timeout=rem)
            return self._latest

    def close(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        proc = self._proc
        self._proc = None
        if proc is not None:
            for stream in (proc.stdout, proc.stderr, proc.stdin):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

    def __enter__(self) -> "CameraStream":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


class VoiceCoach:
    """Orientador de voz SAPI não-bloqueante com fila dedicada e bipes curtos opcionais."""

    def __init__(self, enabled: bool = True, audio_cues: bool = True, rate: int = 1) -> None:
        self.enabled = bool(enabled)
        self.audio_cues = bool(audio_cues)
        self.rate = max(-10, min(10, int(rate)))
        self._queue: queue.Queue[Optional[Tuple[str, str]]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._last_text: str = ""
        self._last_spoken_at: float = 0.0
        self._ready = threading.Event()
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._ready.wait(timeout=2.0)

    def _loop(self) -> None:
        voice = None
        pythoncom = None
        try:
            import pythoncom as _pythoncom
            import win32com.client

            pythoncom = _pythoncom
            pythoncom.CoInitialize()
            voice = win32com.client.Dispatch("SAPI.SpVoice")
            voice.Rate = self.rate
        except Exception:  # noqa: BLE001
            voice = None
        finally:
            self._ready.set()

        try:
            while True:
                item = self._queue.get()
                if item is None:
                    if voice is not None:
                        try:
                            voice.Speak("", 3)
                        except Exception:  # noqa: BLE001
                            pass
                    break
                text, priority = item
                if priority == "__DRAIN__":
                    break
                if voice is None or not text:
                    continue
                try:
                    # 1 = Async, 3 = Async + PurgeBeforeSpeak
                    flags = 3 if priority == "high" else 1
                    voice.Speak(text, flags)
                    while not voice.WaitUntilDone(80):
                        if pythoncom is not None:
                            pythoncom.PumpWaitingMessages()
                        with self._queue.mutex:
                            next_item = self._queue.queue[0] if self._queue.queue else False
                        if next_item is None or (isinstance(next_item, tuple) and next_item[1] == "high"):
                            voice.Speak("", 3)
                            break
                except Exception:  # noqa: BLE001
                    pass
        finally:
            if pythoncom is not None:
                try:
                    pythoncom.CoUninitialize()
                except Exception:  # noqa: BLE001
                    pass

    def say(self, text: str, priority: str = "normal", min_interval: float = 1.6) -> None:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        if not self.enabled or not clean:
            return
        now = time.monotonic()
        if priority != "high":
            if clean == self._last_text and (now - self._last_spoken_at) < max(min_interval * 1.8, 3.0):
                return
            if (now - self._last_spoken_at) < min_interval:
                return
        else:
            # Limpa fila pendente ao receber aviso urgente (centralizado / contagem / clique).
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._last_text = clean
        self._last_spoken_at = now
        self._queue.put((clean, priority))

    def play_cue(self, kind: str) -> None:
        if not self.audio_cues:
            return
        tones = {
            "centered": (880, 110),
            "tick": (660, 75),
            "shutter": (1046, 130),
            "warn": (440, 120),
        }
        freq, dur = tones.get(kind, (0, 0))
        if not freq:
            return

        def _beep() -> None:
            try:
                import winsound

                winsound.Beep(freq, dur)
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_beep, daemon=True).start()

    def close(self, wait: bool = True, drain: bool = False, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._queue.put(("", "__DRAIN__") if drain else None)
        if wait and self._thread.is_alive():
            self._thread.join(timeout=max(0.2, float(timeout)))
            if self._thread.is_alive():
                self._queue.put(None)
                self._thread.join(timeout=0.5)
        self._thread = None

    def __enter__(self) -> "VoiceCoach":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
