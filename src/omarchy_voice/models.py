"""Explicit model downloads and lifecycle, delegated to the owning runtime."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .providers import ProviderError, brain_endpoint, key_headers, ollama_root, request_json
from .session import send_control
from .speech import model_path


def native_command(command: list[str], *, cwd: Path | None = None) -> None:
    """Run a user-requested setup operation, never an LLM-supplied command."""
    try:
        result = subprocess.run(command, cwd=cwd, check=False)
    except OSError as exc:
        raise ProviderError(f"Cannot start {Path(command[0]).name}; install its runtime first") from exc
    if result.returncode:
        raise ProviderError(f"Model command failed with status {result.returncode}")


def ollama_operation(config, action: str) -> dict:
    """List, download, load or unload the configured model on its Ollama server."""
    if config.brain_provider != "ollama":
        raise ProviderError("Brain model management requires brain.provider = 'ollama'")
    endpoint = brain_endpoint(config)
    payload = {"model": endpoint.model, "stream": False}
    if action == "list":
        route, payload = "tags", None
    elif action == "download":
        route = "pull"
    elif action in {"load", "unload"}:
        route = "generate"
        payload["keep_alive"] = config.brain_keep_alive if action == "load" else 0
        payload["options"] = {"num_ctx": config.brain_context}
    else:
        raise ProviderError(f"Unsupported brain model action: {action}")
    headers = key_headers(endpoint.key_env, required=bool(endpoint.key_env))
    data = request_json(ollama_root(config) + "/api/" + route, payload, headers, 1800)
    if data.get("error"):
        raise ProviderError("Ollama model operation failed; inspect the server log")
    return data


def manage(config, action: str, target: str) -> str:
    """Manage ears/brain/mouth without silently selecting or downloading a model."""
    if target == "ears":
        if config.stt_provider != "voxtype":
            raise ProviderError("API ears have no local model to manage")
        commands = {"list": ["status", "--extended", "--format", "json"],
                    "download": ["setup", "--download"]}
        if action not in commands:
            raise ProviderError("Voxtype owns its model lifetime; use its configuration for preload/unload")
        native_command([config.stt_binary, *commands[action]])
        return "Voxtype retains its existing model directory and configuration."
    if target == "brain":
        return json.dumps(ollama_operation(config, action), indent=2)
    if target != "mouth" or config.tts_provider != "piper":
        raise ProviderError("Mouth model management requires mouth.provider = 'piper'")
    if action == "list":
        folder = Path(config.models_dir).expanduser() / "piper"
        voices = sorted(str(path) for path in folder.glob("*.onnx") if Path(str(path) + ".json").is_file())
        return json.dumps({"selected": str(model_path(config)), "downloaded": voices}, indent=2)
    if action == "download":
        if config.tts_model:
            raise ProviderError("mouth.model points to an existing file; clear it to download a catalog voice")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.tts_voice):
            raise ProviderError("Invalid Piper catalog voice identifier")
        folder = model_path(config).parent
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        native_command([str(Path(config.tts_python).expanduser()), "-m", "piper.download_voices",
                        config.tts_voice, "--data-dir", str(folder)])
        return f"Piper model directory: {folder}"
    if action in {"load", "unload"}:
        return send_control(f"models {action} mouth")
    raise ProviderError(f"Unsupported mouth model action: {action}")
