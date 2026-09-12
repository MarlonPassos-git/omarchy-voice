"""Turn-based ears -> brain -> mouth, sharing the original desktop policy gate.

The control socket stays responsive while a provider is working. Cancellation
prevents subsequent tool calls; it cannot undo a tool already executing.
"""

from __future__ import annotations

import fcntl
import queue
import shutil
import subprocess
import signal
import tempfile
import threading
import time
from pathlib import Path

from .capture import make_capture, voxtype_command, voxtype_state
from .config import RUNTIME_DIR, dir_is_private
from .feedback import Feedback
from .planner import Planner
from .providers import ProviderError, brain_endpoint, checked_url, key_headers
from .session import ControlServer, daemon_running
from .speech import Mouth, model_path
from .tools import Executor


def configuration_errors(config) -> list[str]:
    """Validate pipeline choices without starting a model or making API calls."""
    errors = []
    choices = {
        "voice_mode": {"pipeline", "realtime"},
        "stt_provider": {"voxtype", "openai", "compatible"},
        "tts_provider": {"none", "piper", "openai", "compatible", "elevenlabs"},
        "tts_device": {"cpu", "cuda"},
    }
    for name, allowed in choices.items():
        if not isinstance(getattr(config, name), str) or getattr(config, name) not in allowed:
            errors.append(f"{name} must be one of: {', '.join(sorted(allowed))}")
    for name, low, high in [("voice_max_record_seconds", 1, 120), ("brain_timeout", 1, 300),
                            ("stt_timeout", 1, 300), ("tts_timeout", 1, 300),
                            ("tts_idle_seconds", 0, 86400), ("max_turns", 1, 32)]:
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
            errors.append(f"{name} must be between {low} and {high}")
    if not isinstance(config.max_turns, int) or isinstance(config.max_turns, bool):
        errors.append("max_turns must be an integer")
    if not isinstance(config.brain_context, int) or isinstance(config.brain_context, bool) or not 1024 <= config.brain_context <= 131072:
        errors.append("brain.context must be an integer between 1024 and 131072")
    if not isinstance(config.brain_keep_alive, str):
        errors.append("brain.keep_alive must be a duration string")
    if not isinstance(config.brain_extra, dict):
        errors.append("brain.extra must be a table")
    try:
        endpoint = brain_endpoint(config)
        key_headers(endpoint.key_env, required=bool(endpoint.key_env))
    except (ProviderError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def check_ready(config) -> list[str]:
    """Check only selected providers; local mode needs no OpenAI key/websocket."""
    errors = configuration_errors(config)
    if errors:
        return errors
    if config.stt_provider == "voxtype":
        try:
            help_text = voxtype_command(config, "record", "start", "--help")
            if "--output-file" not in help_text:
                errors.append("Voxtype needs record start --output-file support; update Voxtype")
            voxtype_command(config, "record", "cancel", "--help")
            voxtype_state(config)
        except ProviderError as exc:
            errors.append(str(exc))
    else:
        if not shutil.which("pw-record"):
            errors.append("pw-record is required for API transcription")
        try:
            base = checked_url(config.stt_base_url or ("https://api.openai.com/v1" if config.stt_provider == "openai" else ""))
            env = config.stt_api_key_env or ("OPENAI_API_KEY" if config.stt_provider == "openai"
                                             and base == "https://api.openai.com/v1" else "")
            key_headers(env, required=bool(env))
        except (ProviderError, OSError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
    if config.tts_provider in {"openai", "compatible", "elevenlabs"} and not config.tts_voice.strip():
        errors.append("mouth.voice must identify a voice for the selected provider")
    if config.tts_provider != "none" and not shutil.which("pw-play"):
        errors.append("pw-play is required for speech playback")
    if config.tts_provider == "piper":
        try:
            path = model_path(config)
            if not path.is_file() or not Path(str(path) + ".json").is_file():
                errors.append("Piper voice missing; run omarchy-voice models download mouth")
            if not shutil.which(str(Path(config.tts_python).expanduser())):
                errors.append("Piper Python runtime missing; run omarchy-voice setup")
            else:
                result = subprocess.run([str(Path(config.tts_python).expanduser()), "-c",
                    "import importlib.util; raise SystemExit(importlib.util.find_spec('piper') is None)"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
                if result.returncode:
                    errors.append("Selected Python has no piper-tts; run omarchy-voice setup")
        except (ProviderError, OSError, subprocess.SubprocessError) as exc:
            errors.append(str(exc))
    if config.tts_provider in {"openai", "elevenlabs", "compatible"}:
        try:
            default = {"elevenlabs": "https://api.elevenlabs.io/v1", "openai": "https://api.openai.com/v1"}.get(config.tts_provider, "")
            base = checked_url(config.tts_base_url or default)
            env = config.tts_api_key_env
            if not env and base == default:
                env = {"openai": "OPENAI_API_KEY", "elevenlabs": "ELEVENLABS_API_KEY"}.get(config.tts_provider, "")
            key_headers(env, required=bool(env))
        except ProviderError as exc:
            errors.append(str(exc))
    return errors


class PipelineSession:
    """A single turn worker; the socket thread only schedules or cancels work."""

    def __init__(self, config):
        self.config = config
        self.feedback = Feedback(config)
        self.executor = Executor(config, on_action=self._on_action)
        self.planner = Planner(config, self.executor)
        self.mouth = Mouth(config)
        self.commands = queue.Queue(maxsize=1)
        self.cancelled = threading.Event()
        self.quitting = threading.Event()
        self.lock = threading.Lock()
        self.status = "idle"
        self.capture = None
        self.directory = None
        self.recorded_at = 0.0
        self.watch_at = 0.0

    def _state(self, status: str, text: str = "") -> None:
        with self.lock:
            # A cancellation arriving during a state update must keep new work
            # out until cleanup has run on the turn worker.
            self.status = "cancelling" if self.cancelled.is_set() else status
        self.feedback.state(status, text)

    def _on_action(self, name: str, description: str) -> None:
        self._state("acting", description)
        if self.config.voice_log_transcripts:
            self.feedback.log(f"action  {description}")

    def control(self, command: str) -> str:
        """Accept local controls immediately, even during a slow HTTP request."""
        with self.lock:
            if command == "toggle" and self.status not in {"idle", "listening"}:
                command = "cancel"
            if command in {"cancel", "stop", "quit"}:
                self.cancelled.set()
                self.status = "cancelling"
                if command == "quit":
                    self.quitting.set()
                return "cancellation requested; already-running actions cannot be undone"
            action, _, text = command.partition(" ")
            if action == "toggle":
                action = "finish" if self.status == "listening" else "start"
            allowed = {"start": "idle", "finish": "listening", "say": "idle", "confirm": "confirm",
                       "models": "idle"}
            if action not in allowed:
                return "error: unsupported pipeline control"
            if self.status != allowed[action] or self.commands.full():
                return "busy; use listen confirm for a held action, or listen cancel"
            if action == "say" and (not text.strip() or len(text) > 16384):
                return "error: say needs 1..16384 characters"
            if action == "models" and text not in {"load mouth", "unload mouth"}:
                return "error: unsupported model control"
            self.cancelled.clear()
            self.status = "queued"
            self.commands.put_nowait((action, text))
            return "accepted"

    def _start(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="utterance-", dir=RUNTIME_DIR)
        self.capture = make_capture(self.config, Path(self.directory.name))
        self.capture.start()
        self.recorded_at = time.monotonic()
        self._state("listening", "Press the voice shortcut again to submit")

    def _release_capture(self) -> None:
        try:
            if self.capture is not None:
                self.capture.cancel()
        finally:
            self.capture = None
            if self.directory is not None:
                self.directory.cleanup()
                self.directory = None

    def _finish(self) -> None:
        self._state("transcribing")
        try:
            text = self.capture.finish(self.cancelled.is_set)
        finally:
            self._release_capture()
        if text and not self.cancelled.is_set():
            self._think(text)
        else:
            self._state("idle", "No command submitted")

    def _think(self, text: str) -> None:
        if self.config.voice_log_transcripts:
            self.feedback.log(f"heard   {text}")
        self._state("thinking")
        turn = self.planner.think(text, self.cancelled.is_set)
        self.feedback.log(f"pipeline elapsed={turn.elapsed:.3f}s actions={len(turn.actions)} "
                          f"tokens={turn.tokens} error={bool(turn.error)}")
        if self.cancelled.is_set():
            return
        reply = turn.reply
        if self.executor.pending:
            reply = "Confirmation required. Use omarchy-voice listen confirm, or listen cancel."
        if turn.error:
            self.feedback.notify("Voice provider error", turn.error, "normal")
        self._reply(reply)
        # Do not accumulate a whole day's desktop contents in memory.
        self.executor.transcript[:] = self.executor.transcript[-100:]

    def _reply(self, text: str) -> None:
        if self.cancelled.is_set():
            return
        self.feedback.notify("OMA", text)
        if self.config.voice_log_transcripts:
            self.feedback.log(f"reply   {text}")
        try:
            self._state("speaking", text)
            self.mouth.speak(text, self.cancelled.is_set)
        except (ProviderError, OSError) as exc:
            self.feedback.notify("Speech unavailable", str(exc), "normal")
        finally:
            held = self.executor.describe(*self.executor.pending) if self.executor.pending else ""
            self._state("confirm" if held else "idle", held or text)

    def _confirm(self) -> None:
        # Only control() can enqueue this action. Transcripts never release the
        # policy gate, and the original Executor still applies dry-run behavior.
        if not self.cancelled.is_set():
            result = self.executor.run_pending()
            self._reply(result.output or ("Done." if result.ok else "Action failed."))

    def _model_control(self, text: str) -> None:
        if self.config.tts_provider != "piper":
            raise ProviderError("Resident mouth model controls require Piper")
        if text == "load mouth":
            self.mouth.piper.load(self.cancelled.is_set)
        else:
            self.mouth.close()
        self._state("idle", text)

    def _abort(self) -> None:
        try:
            self._release_capture()
        except ProviderError as exc:
            self.feedback.notify("Recorder cleanup failed", str(exc), "normal")
        self.executor.drop_pending()
        with self.lock:
            while not self.commands.empty():
                self.commands.get_nowait()
            self.cancelled.clear()
            self.status = "idle"
        self.feedback.state("idle", "Cancelled")

    def _maintenance(self) -> None:
        if self.capture is not None and time.monotonic() - self.recorded_at >= self.config.voice_max_record_seconds:
            self.cancelled.set()
            self.feedback.notify("Recording limit reached", "Recording discarded; no command executed")
        self.mouth.piper.expire()
        if self.status == "idle" and time.monotonic() - self.watch_at > 3:
            self.watch_at = time.monotonic()
            for job in self.executor.poll_watches():
                reason = "closed" if job["vanished"] else "timed out" if job["timed_out"] else "finished"
                self.feedback.notify("Terminal task", f"{job['label']}: {reason}")

    def step(self, timeout: float = 0.1) -> None:
        """Process one control or housekeeping tick; deterministic in unit tests."""
        if self.cancelled.is_set():
            self._abort()
            return
        try:
            action, text = self.commands.get(timeout=timeout)
        except queue.Empty:
            self._maintenance()
            return
        try:
            if self.cancelled.is_set():
                return
            handlers = {"start": self._start, "finish": self._finish, "confirm": self._confirm}
            if action == "say":
                self._think(text)
            elif action == "models":
                self._model_control(text)
            else:
                handlers[action]()
        except Exception as exc:
            self.feedback.log(f"pipeline failure={type(exc).__name__}")
            self.feedback.notify("Voice pipeline error", str(exc), "normal")
            self.cancelled.set()
        finally:
            if self.cancelled.is_set():
                self._abort()

    def close(self) -> None:
        """Release our capture and voice, but never stop the user's Voxtype daemon."""
        self.cancelled.set()
        self._abort()
        self.mouth.close()


def run(config) -> int:
    """Start the same control socket/UI as Realtime, with turn-based capture."""
    problems = check_ready(config)
    if problems:
        for problem in problems:
            print(f"error: {problem}")
        return 1
    RUNTIME_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not dir_is_private(RUNTIME_DIR):
        raise ProviderError("Voice runtime directory must be owned by you with mode 700")
    with (RUNTIME_DIR / "pipeline.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ProviderError("Another voice pipeline is already running")
        if daemon_running():
            raise ProviderError("Another voice daemon is already running")
        return _serve(config)


def _serve(config) -> int:
    session = PipelineSession(config)
    server = ControlServer(session.control)
    previous = {sig: signal.signal(sig, lambda *_: session.control("quit"))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        server.start()
        session.feedback.state("idle", "Pipeline ready; microphone off")
        session.feedback.level(0.0)  # Voxtype owns capture; do not invent a second VU stream.
        while not session.quitting.is_set():
            session.step()
        return 0
    finally:
        server.stop()
        session.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
