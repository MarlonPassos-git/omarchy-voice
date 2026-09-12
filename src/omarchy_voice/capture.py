"""The ears: reuse Voxtype's daemon or record a bounded clip for an STT API."""

from __future__ import annotations

import json
import secrets
import signal
import subprocess
import time
from pathlib import Path

from .providers import ProviderError, checked_url, key_headers, request_bytes
from .speech import stop_process

TRANSCRIPT_LIMIT = 16384
CLIP_LIMIT = 4 * 1024 * 1024


def voxtype_command(config, *args: str) -> str:
    """Invoke Voxtype without a shell; stdout contains only its control result."""
    try:
        result = subprocess.run([config.stt_binary, *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProviderError("Cannot reach Voxtype; check its binary and user service") from exc
    if result.returncode:
        raise ProviderError("Voxtype command failed; check voxtype status and its daemon log")
    return result.stdout


def voxtype_state(config) -> str:
    """Read Voxtype's documented Waybar JSON class, not localized tooltip text."""
    try:
        payload = json.loads(voxtype_command(config, "status", "--format", "json"))
        state = payload["class"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ProviderError("Voxtype must expose status --format json with a class field") from exc
    if not isinstance(state, str):
        raise ProviderError("Unrecognized Voxtype state")
    return state


class VoxtypeCapture:
    """One per-recording file override; never read or overwrite the clipboard."""

    def __init__(self, config, directory: Path):
        self.config = config
        self.path = directory / "transcript.txt"
        self.recording = False

    def start(self) -> None:
        """Refuse to hijack an existing F9 recording or transcription."""
        if voxtype_state(self.config) != "idle":
            raise ProviderError("Voxtype is not idle; finish the current dictation first")
        voxtype_command(self.config, "record", "start", "--output-file", str(self.path))
        self.recording = True

    def finish(self, cancelled) -> str:
        """Stop and wait for this recording's output; old transcripts cannot match."""
        voxtype_command(self.config, "record", "stop")
        deadline = time.monotonic() + self.config.stt_timeout
        while time.monotonic() < deadline and not cancelled():
            # Idle is published after output; it also prevents a partial-file read.
            if self.path.exists() and voxtype_state(self.config) == "idle":
                with self.path.open("rb") as file:
                    raw = file.read(TRANSCRIPT_LIMIT + 1)
                if len(raw) > TRANSCRIPT_LIMIT:
                    raise ProviderError("Voxtype transcript is too large")
                self.recording = False
                return raw.decode("utf-8").strip()
            time.sleep(0.1)
        if cancelled():
            return ""
        raise ProviderError("Voxtype transcription timed out; no command was executed")

    def cancel(self) -> None:
        """Stop only a recording started by us; discarded text never reaches the brain."""
        if self.recording:
            voxtype_command(self.config, "record", "cancel")
            deadline = time.monotonic() + 5
            while voxtype_state(self.config) != "idle":
                if time.monotonic() >= deadline:
                    raise ProviderError("Voxtype did not acknowledge cancellation; check its status")
                time.sleep(0.1)
            self.recording = False


def multipart_audio(config, audio: bytes) -> tuple[bytes, str]:
    """Encode the fixed STT form without requests or a provider SDK."""
    boundary = "omarchy-voice-" + secrets.token_hex(16)
    fields = {"model": config.stt_model, "response_format": "json"}
    if config.stt_language:
        fields["language"] = config.stt_language
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
             for key, value in fields.items()]
    parts.extend([f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="speech.wav"\r\n'
                  'Content-Type: audio/wav\r\n\r\n'.encode(), audio,
                  f'\r\n--{boundary}--\r\n'.encode()])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe_api(config, path: Path) -> str:
    """Send a short WAV only to the explicitly selected STT endpoint."""
    with path.open("rb") as file:
        audio = file.read(CLIP_LIMIT + 1)
    if len(audio) > CLIP_LIMIT:
        raise ProviderError("Recording exceeds the 4 MiB STT limit")
    body, content_type = multipart_audio(config, audio)
    base = checked_url(config.stt_base_url or ("https://api.openai.com/v1" if config.stt_provider == "openai" else ""))
    env = config.stt_api_key_env or ("OPENAI_API_KEY" if config.stt_provider == "openai"
                                      and base == "https://api.openai.com/v1" else "")
    headers = key_headers(env, required=bool(env))
    raw = request_bytes(base + "/audio/transcriptions", body,
                        {"Content-Type": content_type, **headers}, config.stt_timeout)
    try:
        text = json.loads(raw)["text"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ProviderError("STT API did not return a text transcription") from exc
    if not isinstance(text, str) or len(text) > TRANSCRIPT_LIMIT:
        raise ProviderError("STT API returned an invalid or oversized transcript")
    return text.strip()


class ApiCapture:
    """PipeWire capture starts on demand and is stopped before any upload."""

    def __init__(self, config, directory: Path):
        self.config = config
        self.path = directory / "speech.wav"
        self.process = None

    def start(self) -> None:
        command = ["pw-record", "--rate", "16000", "--channels", "1", "--format", "s16"]
        if self.config.device:
            command += ["--target", self.config.device]
        self.process = subprocess.Popen([*command, str(self.path)], stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def finish(self, cancelled) -> str:
        if self.process is None or self.process.poll() is not None:
            raise ProviderError("PipeWire recorder stopped unexpectedly")
        # SIGINT lets pw-record finalize the WAV header before we read the file.
        self.process.send_signal(signal.SIGINT)
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired as exc:
            raise ProviderError("PipeWire recorder did not stop") from exc
        finally:
            self.cancel()
        return "" if cancelled() else transcribe_api(self.config, self.path)

    def cancel(self) -> None:
        stop_process(self.process)
        self.process = None


def make_capture(config, directory: Path):
    """Choose ears independently of the text brain and speech mouth."""
    if config.stt_provider == "voxtype":
        return VoxtypeCapture(config, directory)
    if config.stt_provider in {"openai", "compatible"}:
        return ApiCapture(config, directory)
    raise ProviderError(f"unsupported ears provider: {config.stt_provider!r}")
