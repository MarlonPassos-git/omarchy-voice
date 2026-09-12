"""Optional Piper subprocess: one loaded voice, JSON requests, WAV responses.

Kept in a separate interpreter so the main CLI needs neither Piper nor ONNX.
No model code is downloaded or executed by this worker.
"""

from __future__ import annotations

import json
import sys
import wave


def reply(payload: dict) -> None:
    """Write one complete protocol response, e.g. reply({'ready': True})."""
    print(json.dumps(payload), flush=True)


def serve(voice, source, send=reply) -> None:
    """Synthesize requests with an already-loaded voice; useful for fake-voice tests."""
    for line in source:
        try:
            request = json.loads(line)
            text = request["text"]
            if not isinstance(text, str) or not text.strip() or len(text) > 4096:
                raise ValueError("text must contain 1..4096 characters")
            with wave.open(request["output"], "wb") as wav:
                voice.synthesize_wav(text, wav)
            send({"ok": True})
        except Exception as exc:
            send({"error": f"Piper synthesis failed ({type(exc).__name__})"})


def main() -> int:
    """Load the selected local ONNX file once, then serve until stdin closes."""
    try:
        from piper import PiperVoice
        voice = PiperVoice.load(sys.argv[1], use_cuda=sys.argv[2] == "cuda")
    except Exception as exc:
        reply({"error": f"Piper could not load ({type(exc).__name__}); check runtime and model"})
        return 1
    reply({"ready": True})
    serve(voice, sys.stdin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
