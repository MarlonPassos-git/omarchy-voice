# Ears, brain and mouth

`omarchy-voice setup` selects one of two modes. Existing installations remain in
`realtime` until you select `pipeline`; no API key or model is migrated implicitly.
The `omarchy voice setup` alias is available when the optional Omarchy routes are
installed. No extra framework or Python SDK is required for the pipeline itself.

| Mode | Input / reasoning / output | Interaction |
| --- | --- | --- |
| `realtime` | Existing OpenAI Realtime speech-to-speech transport | Existing streaming conversation |
| `pipeline` | Independent ears, text brain and mouth | Toggle once to record; again to submit a command |

Pipeline is **not** a drop-in full-duplex conversation transport. Each utterance
starts a fresh planner conversation; tool rounds within that utterance retain
history. It records no audio while its brain or mouth is working. There is no
wake-word or automatic end-of-speech detector in this mode. `barge_in` remains a
Realtime setting; use `listen cancel` to stop pipeline speech.

## Quick start

```sh
./install.sh                 # offers setup; does not require OpenAI for local mode
omarchy-voice setup          # can also be rerun later
omarchy-voice doctor         # checks selected dependencies, not paid API inference
systemctl --user restart omarchy-voice

# No desktop changes; live read-only queries are still allowed:
omarchy-voice --dry-run say "Abra o navegador no workspace dois"
# Test the mouth only, never desktop tools:
omarchy-voice speak "Olá, estou funcionando."
```

The existing **SUPER + SHIFT + V** shortcut controls the selected mode. In
pipeline, the first press starts recording and the second submits it. `listen
start` starts capture; `listen stop` / `listen cancel` **discard**, not submit.
Toggling while thinking/transcribing/speaking requests cancellation. The bar shows
these phases; no second microphone is opened to animate an audio-level meter.
The running daemon also accepts `listen say "..."` for typed pipeline commands.
`omarchy-voice say "..."` is a separate one-shot CLI planner with terminal-based
confirmation, not spoken output; use `listen say` for the complete daemon flow.

F9 continues to belong to Voxtype. Do not use F9 and the agent shortcut at the
same time: they share Voxtype's recorder, not two independent microphone sessions.
The adapter refuses to start when Voxtype is busy, but it cannot prevent another
application/keybinding from controlling that shared daemon afterwards.

## Supported providers

| Stage | Provider | Implemented transport |
| --- | --- | --- |
| Ears | `voxtype` | Existing daemon, per-recording `--output-file`, no clipboard |
| Ears | `openai`, `compatible` | Bounded WAV -> `/audio/transcriptions` multipart |
| Brain | `openai`, `gemini`, `openrouter`, `lmstudio`, `compatible` | Chat Completions + tools; opaque assistant metadata retained |
| Brain | `ollama` | Native `/api/chat`, context size and keep-alive; tool argument normalization |
| Mouth | `piper` | Persistent local Piper worker in a selected Python interpreter |
| Mouth | `openai`, `compatible` | `/audio/speech`, WAV output |
| Mouth | `elevenlabs` | Native text-to-speech endpoint, PCM 24 kHz wrapped as WAV |
| Mouth | `none` | Text notifications only |

A compatible text endpoint must actually support tool calls; the label
"OpenAI-compatible" alone is not sufficient. A compatible speech server must
support the relevant audio endpoint, not just Chat Completions. A separately
installed Kokoro server exposing `/audio/speech` can be configured as
`compatible`; this PR does **not** install or manage Kokoro natively.

Google is supported as the **text brain** through Gemini's documented OpenAI
compatibility API. Native Gemini Live, Google Cloud Speech/TTS and native
Anthropic adapters are not implemented here. Claude and other text models can
be used through a compatible gateway such as OpenRouter, subject to that
model/provider's tool support. The integrated Realtime mode remains OpenAI-only.

## Fully local example

Start from [`share/pipeline.example.toml`](../share/pipeline.example.toml), or use
the wizard. `qwen3:4b` is an example, not a claim of measured performance or tool
accuracy on a particular machine.

```toml
[voice]
mode = "pipeline"
max_record_seconds = 30
log_transcripts = false

[ears]
provider = "voxtype"
binary = "voxtype"
timeout = 60

[brain]
provider = "ollama"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen3:4b"
context = 16384
keep_alive = "5m"
timeout = 120

[mouth]
provider = "piper"
voice = "pt_BR-faber-medium"
python = "/home/YOU/.local/share/omarchy-voice/runtime/piper/bin/python"
device = "cpu"
idle_seconds = 300
notify = true
```

`[ears]` / `[mouth]` are friendly aliases for `[stt]` / `[tts]`. Existing
`ears.device`, `ears.barge_in`, `mouth.notify`, `mouth.speak` and `mouth.tts_command`
retain their legacy meanings. For pipeline, `mouth.provider` selects speech;
legacy `mouth.speak = false` does not disable a selected Piper/API mouth.
For Voxtype, its **own** mic/engine/model settings apply; `ears.model` is for API
transcription, not a way to replace Voxtype's configured Whisper model.

A large desktop manifest requires a substantial context window. Local LLM
latency and memory use depend on prompt size, context, quantization and GPU
contention. Do not expect a 4B model, STT and TTS all to fit comfortably into 6 GB
of VRAM without measurement. CPU Piper + the existing Voxtype + a remote text
brain is an alternative; none of the hardware latency figures are benchmarked
by this PR. Unsupported tool calls are rejected, not interpreted as shell text.

## Hybrid examples

Keep local ears and Piper, replacing only the brain:

```toml
[brain]
provider = "gemini"
model = "YOUR_TOOL_CAPABLE_GEMINI_MODEL"
api_key_env = "GEMINI_API_KEY"
```

Or configure `provider = "openrouter"`, an explicit model ID and
`api_key_env = "OPENROUTER_API_KEY"`. OpenAI uses the legacy `planner_model`
unless `brain.model` or CLI `--model` is specified. Other providers require an
explicit model. `--model` overrides either configuration.

For ElevenLabs output:

```toml
[mouth]
provider = "elevenlabs"
model = "eleven_multilingual_v2"
voice = "YOUR_ELEVENLABS_VOICE_ID"
api_key_env = "ELEVENLABS_API_KEY"
```

The wizard stores tokens only in `~/.config/omarchy-voice/env` (mode 600), with
hidden input. It backs up existing configuration and secret files using private
timestamped backups; keep these backups private too. Config values and security
rules are preserved, while comments remain in the original backup rather than
in the rewritten TOML. Restart the daemon after changing configuration.

Provider presets keep credentials separate. Changing to a custom base URL does
not automatically forward the original provider's key. Select its environment
variable explicitly. HTTP is accepted only for literal loopback/localhost;
remote/LAN services require HTTPS. Credential-bearing redirects are refused.
Do not put credentials in URL query strings. Loopback requests bypass system
HTTP proxies. There are no automatic billable retries or cloud fallbacks.

## Model storage and lifecycle

The pipeline uses existing model managers rather than duplicating their caches:

| Stage | Storage and lifecycle |
| --- | --- |
| Voxtype | Its existing model directory and preload/unload configuration. No second Whisper download. |
| Ollama | The selected Ollama server owns the cache and resident models. `OLLAMA_MODELS`, if needed, belongs to that server's environment. |
| Piper | `$XDG_DATA_HOME/omarchy-voice/models/piper` (normally `~/.local/share/...`), or your selected `models_dir`. One persistent worker, unloaded after idle timeout. |

```sh
omarchy-voice models list ears
omarchy-voice models download ears       # delegates to voxtype setup --download
voxtype setup model                     # choose/change its model using Voxtype

omarchy-voice models list brain          # Ollama only
omarchy-voice models download brain      # explicit download, may be several GB
omarchy-voice models load brain
omarchy-voice models unload brain        # affects that model on the shared server

omarchy-voice models list mouth          # Piper only
omarchy-voice models download mouth
omarchy-voice models load mouth          # asks the running pipeline to keep it resident
omarchy-voice models unload mouth
```

Daemon model controls acknowledge scheduling; check `status`/notifications for
completion. They are refused while a command/recording is active. These controls
are not exposed as LLM tools. API models have no local download/unload operation.
The application does not stop shared Ollama or Voxtype services.

Setup optionally creates an isolated Piper virtual environment and installs
`piper-tts` from PyPI **only after confirmation**. The main CLI itself remains
stdlib-based, with `websockets` optional for Realtime. Piper downloads use its
maintained `piper.download_voices` command; both `.onnx` and `.onnx.json` must be
present. A custom `mouth.model` points to an existing ONNX file instead of the
catalog. Review model licenses before redistribution; Piper's runtime has its
own license and is not vendored into this MIT project.

The persistent worker loads on first speech or `models load mouth`. Default
idle unload is 300 seconds; zero disables idle unload. `mouth.device = "cuda"`
requires a working `onnxruntime-gpu` installation in that selected interpreter;
CPU is the setup default. Piper here generates a short WAV before playback;
this is not token/audio streaming. Cold model loads and long text can add delay.

## Safety and privacy boundaries

All desktop actions still pass through the original `Executor` and policy gate.
In pipeline, a held action can only be released by the local confirmation button / `listen confirm` or
cancelled by `listen cancel`; saying "confirm" does not release it. The planner
stops a tool batch immediately at a held action. `--dry-run` still allows live
read-only queries but does not execute mutating desktop actions.

Cancellation discards queued work, requests that the owned recorder stop, kills
playback, and prevents subsequent tool calls after a provider returns. It cannot
undo an action already executing, or retract a request already received by an
API. An in-flight HTTP call may take until its timeout to return; controls remain
responsive and no new command is accepted during cleanup. Requests are bounded,
recordings time out and are discarded, and temporary clips/transcripts are kept
under the owner-only runtime directory rather than the clipboard.

Voxtype is not automatically synonymous with offline inference: it can be
configured with remote engines or post-processors. Verify its own configuration.
A remote brain receives transcripts, desktop context and tool results; a remote
mouth receives response text; API ears receive audio. Web/browser tools can still
access the internet even when all inference models are local. This is not a
network sandbox. The default pipeline log records timing/actions count/token
counts, not utterance text; enable `voice.log_transcripts` only intentionally.
Notifications and the private status file still contain the current response.
Voxtype and external servers may keep their own independent logs.

## Verification and extension points

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src
bash -n install.sh
```

Tests exercise tool schemas, credential isolation, provider payloads, cancellation,
recording cleanup, persistent-worker protocol, configuration backups, model
management and legacy Realtime behavior using fake models/services. They do not
call paid APIs or benchmark a microphone/GPU. `doctor` is a readiness check, not
an end-to-end audio quality or remote connectivity test.

Adapters live in `capture.py` (ears), `providers.py` (brain), `speech.py` (mouth).
`pipeline.py` owns the turn lifecycle and `models.py` delegates model management.
A new provider should get contract tests and explicit privacy/configuration
handling, without replacing the policy gate or silently falling back elsewhere.

Primary API contracts used for this implementation:
- [Voxtype CLI and configuration](https://github.com/peteonrails/voxtype)
- [Piper Python API](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md)
- [Piper downloader](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/CLI.md)
- [Ollama tool calling](https://docs.ollama.com/capabilities/tool-calling)
- [Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- [OpenAI text to speech](https://developers.openai.com/api/docs/guides/text-to-speech)
- [ElevenLabs text to speech](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
