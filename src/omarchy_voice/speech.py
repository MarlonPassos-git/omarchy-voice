"""The mouth: persistent local Piper or explicitly selected speech APIs."""

from __future__ import annotations

import io
import json
import os
import re
import select
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from urllib.parse import quote

from .config import RUNTIME_DIR
from .providers import AUDIO_LIMIT, ProviderError, checked_url, key_headers, request_bytes


def model_path(config) -> Path:
    """Locate a Piper voice, e.g. <models_dir>/piper/pt_BR-faber-medium.onnx."""
    if config.tts_model:
        return Path(config.tts_model).expanduser()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.tts_voice):
        raise ProviderError("Piper voice must be a model identifier, not a path")
    return Path(config.models_dir).expanduser() / "piper" / f"{config.tts_voice}.onnx"


def stop_process(process) -> None:
    """Reap a child even if it ignores termination; safe to call twice."""
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def wait_process(process, timeout: float, cancelled) -> None:
    """Wait for playback while keeping local cancellation responsive."""
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if cancelled() or time.monotonic() >= deadline:
            stop_process(process)
            raise ProviderError("Audio operation cancelled or timed out")
        time.sleep(0.05)
    if process.returncode:
        raise ProviderError(f"Audio process exited with status {process.returncode}")


class PiperWorker:
    """Keep one voice resident until idle timeout or explicit unload."""

    def __init__(self, config):
        self.config = config
        self.process = None
        self.last_used = 0.0

    def _response(self, cancelled) -> dict:
        deadline = time.monotonic() + self.config.tts_timeout
        data = bytearray()
        while time.monotonic() < deadline and not cancelled():
            if self.process is None or self.process.poll() is not None:
                raise ProviderError("Piper worker stopped; check its Python runtime and model")
            if not select.select([self.process.stdout], [], [], 0.1)[0]:
                continue
            chunk = os.read(self.process.stdout.fileno(), 4096)
            if not chunk or len(data) + len(chunk) > 8192:
                raise ProviderError("Invalid response from Piper worker")
            data.extend(chunk)
            if b"\n" in data:
                result = json.loads(data)
                if result.get("error"):
                    raise ProviderError(result["error"])
                return result
        raise ProviderError("Piper cancelled or timed out")

    def load(self, cancelled=lambda: False) -> None:
        """Load once; calling load again does not allocate another model."""
        if self.process is not None and self.process.poll() is None:
            self.last_used = time.monotonic()
            return
        self.close()
        path = model_path(self.config)
        if not path.is_file() or not Path(str(path) + ".json").is_file():
            raise ProviderError("Piper voice files are missing; run omarchy-voice models download mouth")
        command = [str(Path(self.config.tts_python).expanduser()), "-u",
                   str(Path(__file__).with_name("piper_worker.py")), str(path), self.config.tts_device]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
        try:
            if not self._response(cancelled).get("ready"):
                raise ProviderError("Piper did not become ready")
        except Exception:
            self.close()
            raise
        self.last_used = time.monotonic()

    def synthesize(self, text: str, output: Path, cancelled) -> None:
        """Write a WAV using the resident worker, never a shell command."""
        self.load(cancelled)
        try:
            payload = json.dumps({"text": text, "output": str(output)}) + "\n"
            self.process.stdin.write(payload.encode())
            self.process.stdin.flush()
            if not self._response(cancelled).get("ok"):
                raise ProviderError("Piper returned no audio")
            self.last_used = time.monotonic()
        except Exception:
            self.close()
            raise

    def expire(self) -> None:
        """Release idle RAM/VRAM; zero keeps the voice resident indefinitely."""
        idle = self.config.tts_idle_seconds
        if self.process is not None and idle > 0 and time.monotonic() - self.last_used >= idle:
            self.close()

    def close(self) -> None:
        """Unload the optional model without stopping the assistant or Voxtype."""
        stop_process(self.process)
        self.process = None


def pcm_wav(pcm: bytes, rate: int) -> bytes:
    """Wrap provider PCM s16le mono in a WAV container for PipeWire."""
    if not pcm or len(pcm) % 2:
        raise ProviderError("Speech API returned invalid PCM audio")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


def api_speech(config, text: str) -> bytes:
    """Synthesize using OpenAI-compatible speech or ElevenLabs' native API."""
    if config.tts_provider == "elevenlabs":
        base = checked_url(config.tts_base_url or "https://api.elevenlabs.io/v1")
        url = base + "/text-to-speech/" + quote(config.tts_voice, safe="") + "?output_format=pcm_24000"
        env = config.tts_api_key_env or ("ELEVENLABS_API_KEY" if base == "https://api.elevenlabs.io/v1" else "")
        headers = key_headers(env, header="xi-api-key")
        payload = {"text": text, "model_id": config.tts_model or "eleven_multilingual_v2"}
    else:
        base = checked_url(config.tts_base_url or ("https://api.openai.com/v1" if config.tts_provider == "openai" else ""))
        url = base + "/audio/speech"
        env = config.tts_api_key_env
        if not env and config.tts_provider == "openai" and base == "https://api.openai.com/v1":
            env = "OPENAI_API_KEY"
        headers = key_headers(env, required=bool(env))
        payload = {"input": text, "model": config.tts_model or "gpt-4o-mini-tts",
                   "voice": config.tts_voice, "response_format": "wav"}
    data = request_bytes(url, json.dumps(payload).encode(),
                         {"Content-Type": "application/json", **headers}, config.tts_timeout, AUDIO_LIMIT)
    return pcm_wav(data, 24000) if config.tts_provider == "elevenlabs" else data


class Mouth:
    """Own model lifetime and playback; muted audio never starts a recorder."""

    def __init__(self, config):
        self.config = config
        self.piper = PiperWorker(config)

    def speak(self, text: str, cancelled=lambda: False) -> None:
        """Speak one short reply, e.g. mouth.speak('Navegador aberto.')."""
        if self.config.tts_provider == "none" or not text.strip() or cancelled():
            return
        if len(text) > 4096:
            raise ProviderError("Reply exceeds the 4096-character speech limit; read the notification")
        RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="speech-", dir=RUNTIME_DIR) as directory:
            path = Path(directory) / "reply.wav"
            if self.config.tts_provider == "piper":
                self.piper.synthesize(text, path, cancelled)
            elif self.config.tts_provider in {"openai", "compatible", "elevenlabs"}:
                path.write_bytes(api_speech(self.config, text))
            else:
                raise ProviderError(f"unsupported mouth provider: {self.config.tts_provider!r}")
            if not cancelled():
                self._play(path, cancelled)

    def _play(self, path: Path, cancelled) -> None:
        # Validate the container before asking an external decoder to consume it.
        try:
            with wave.open(str(path), "rb") as wav:
                duration = wav.getnframes() / max(1, wav.getframerate())
        except (wave.Error, EOFError) as exc:
            raise ProviderError("Speech provider did not return a valid WAV") from exc
        process = subprocess.Popen(["pw-play", str(path)], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_process(process, min(300, duration + 10), cancelled)
        finally:
            stop_process(process)

    def close(self) -> None:
        """Release the resident voice; playback is reaped by speak's finally block."""
        self.piper.close()
