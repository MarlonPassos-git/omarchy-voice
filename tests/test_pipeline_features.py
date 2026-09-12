"""Provider contract and lifecycle tests; no API keys, models or desktop required."""

import copy
import io
import json
import os
import subprocess
import tempfile
import threading
import time
import tomllib
import unittest
import urllib.error
import wave
from dataclasses import replace
from pathlib import Path
from unittest import mock

from omarchy_voice import capture, cli, config, models, pipeline, providers, setup, speech
from omarchy_voice.config import Config
from omarchy_voice.piper_worker import serve
from omarchy_voice.planner import Planner, Turn
from omarchy_voice.tools import Executor, Result


def local_config(**kwargs):
    return replace(Config(voice_mode="pipeline", brain_provider="ollama", brain_model="qwen3:4b"), **kwargs)


def completion(content="Done", calls=None, **metadata):
    return {"choices": [{"message": {"content": content, "tool_calls": calls or [], **metadata}}]}


def tool(name="hypr_query", arguments='{"kind":"clients"}', id="call_1", **metadata):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}, **metadata}


class ProviderContracts(unittest.TestCase):
    def test_local_provider_does_not_inherit_cloud_credentials(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "private"}, clear=True):
            endpoint = providers.brain_endpoint(local_config())
            self.assertEqual(endpoint.key_env, "")
            self.assertEqual(providers.key_headers(endpoint.key_env, required=False), {})

    def test_each_remote_preset_has_its_own_key(self):
        for provider, env in (("openai", "OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY"),
                              ("openrouter", "OPENROUTER_API_KEY")):
            self.assertEqual(providers.brain_endpoint(Config(brain_provider=provider, brain_model="m")).key_env, env)

    def test_custom_endpoint_cannot_inherit_openai_key(self):
        endpoint = providers.brain_endpoint(Config(brain_base_url="https://example.org/v1"))
        self.assertEqual(endpoint.key_env, "")

    def test_non_openai_needs_explicit_model(self):
        with self.assertRaises(providers.ProviderError):
            providers.brain_endpoint(Config(brain_provider="gemini"))

    def test_endpoint_validation(self):
        for url in ("https://example.org/v1", "http://127.0.0.1:11434/v1", "http://[::1]:1234/v1"):
            self.assertEqual(providers.checked_url(url), url)
        for url in ("http://example.org", "https://user:pass@example.org", "https://a/b?api_key=secret",
                    "file:///etc/passwd", "https://a/#fragment", "http://127.0.0.1.evil.test"):
            with self.subTest(url=url), self.assertRaises(providers.ProviderError):
                providers.checked_url(url)

    def test_redirects_are_not_followed(self):
        with self.assertRaises(providers.ProviderError):
            providers.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.test")

    def test_missing_credentials_and_newlines_rejected(self):
        for value in ("", "abc\nInjected: header"):
            with mock.patch.dict(os.environ, {"KEY": value}, clear=True), self.assertRaises(providers.ProviderError):
                providers.key_headers("KEY")

    def test_http_errors_do_not_leak_response_bodies(self):
        error = urllib.error.HTTPError("https://a", 401, "unauthorized", {}, io.BytesIO(b"secret prompt"))
        with mock.patch.object(providers.urllib.request, "build_opener") as opener:
            opener.return_value.open.side_effect = error
            with self.assertRaises(providers.ProviderError) as raised:
                providers.request_bytes("https://a", b"request", {}, 1)
        self.assertNotIn("secret", str(raised.exception))

    def test_response_bytes_are_bounded(self):
        with mock.patch.object(providers.urllib.request, "build_opener") as opener:
            opener.return_value.open.return_value.__enter__.return_value.read.return_value = b"12345"
            with self.assertRaises(providers.ProviderError):
                providers.request_bytes("https://a", None, {}, 1, limit=4)

    def test_json_contract_rejects_non_objects(self):
        for result in (b"[]", b"bad json"):
            with mock.patch.object(providers, "request_bytes", return_value=result), self.assertRaises(providers.ProviderError):
                providers.request_json("https://a", None, {}, 1)

    def test_compatible_wire_shape_and_explicit_key(self):
        cfg = local_config(brain_provider="gemini", brain_model="selected-model")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "google-key"}, clear=True), \
             mock.patch.object(providers, "request_json", return_value=completion()) as request:
            providers.chat_completion([], [], cfg)
        url, body, headers, timeout = request.call_args.args
        self.assertEqual(url, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
        self.assertEqual(body["model"], "selected-model")
        self.assertEqual(headers["Authorization"], "Bearer google-key")
        self.assertEqual(body["tool_choice"], "auto")

    def test_extra_cannot_replace_safety_relevant_request_fields(self):
        for provider in ("compatible", "ollama"):
            cfg = local_config(brain_provider=provider, brain_base_url="http://localhost:1234/v1",
                               brain_extra={"tools": []})
            with self.assertRaises(providers.ProviderError):
                providers.chat_completion([], [], cfg)

    def test_ollama_roundtrip_keeps_thinking_and_converts_arguments(self):
        messages = [{"role": "assistant", "content": "", "thinking": "opaque",
                     "tool_calls": [tool()]}, {"role": "tool", "tool_call_id": "call_1", "content": "[]"}]
        original = copy.deepcopy(messages)
        output = providers.ollama_messages(messages)
        self.assertEqual(messages, original)
        self.assertEqual(output[0]["thinking"], "opaque")
        self.assertEqual(output[0]["tool_calls"][0]["function"]["arguments"], {"kind": "clients"})
        self.assertEqual(output[1]["tool_name"], "hypr_query")
        response = {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "hypr_query", "arguments": {"kind": "clients"}}}]},
                    "prompt_eval_count": 20, "eval_count": 5}
        with mock.patch.object(providers, "request_json", return_value=response) as request:
            normalized = providers.chat_completion(messages, [], local_config())
        self.assertTrue(request.call_args.args[0].endswith("/api/chat"))
        self.assertEqual(request.call_args.args[1]["options"]["num_ctx"], 16384)
        call = normalized["choices"][0]["message"]["tool_calls"][0]
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertTrue(call["id"])
        self.assertEqual(normalized["usage"]["prompt_tokens"], 20)


class PlannerContracts(unittest.TestCase):
    def setUp(self):
        self.prompt = mock.patch("omarchy_voice.planner._system_prompt", return_value="desktop tools")
        self.prompt.start()
        self.addCleanup(self.prompt.stop)
        self.executor = Executor(local_config())
        self.planner = Planner(local_config(), self.executor)

    def test_opaque_gemini_metadata_survives_tool_round(self):
        calls = [tool(extra_content={"google": {"thought_signature": "opaque"}})]
        first = completion("", calls, extra_content={"signature": "keep"})
        first["usage"] = {"prompt_tokens": 10, "completion_tokens": 2}
        second = completion("Finished")
        second["usage"] = {"prompt_tokens": 20, "completion_tokens": 3}
        with mock.patch("omarchy_voice.planner._chat", side_effect=[first, second]) as request, \
             mock.patch.object(self.executor, "call", return_value=Result(True, "[]")):
            turn = self.planner.think("query")
        self.assertEqual(turn.tokens, {"in": 30, "out": 5})
        self.assertEqual(request.call_args.args[0][2]["extra_content"], {"signature": "keep"})
        self.assertEqual(request.call_args.args[0][2]["tool_calls"], calls)

    def test_cancel_before_request_never_calls_provider(self):
        with mock.patch("omarchy_voice.planner._chat") as request:
            turn = self.planner.think("query", lambda: True)
        self.assertEqual(turn.reply, "Cancelled.")
        request.assert_not_called()

    def test_cancel_after_http_prevents_all_tool_execution(self):
        stop = threading.Event()
        def response(*_):
            stop.set()
            return completion("", [tool()])
        with mock.patch("omarchy_voice.planner._chat", side_effect=response), \
             mock.patch.object(self.executor, "call") as execute:
            self.planner.think("query", stop.is_set)
        execute.assert_not_called()

    def test_pending_action_stops_rest_of_tool_batch(self):
        calls = [tool("omarchy_cli", '{"command":"system reboot"}'), tool(id="second")]
        with mock.patch("omarchy_voice.planner._chat", return_value=completion("", calls)), \
             mock.patch.object(self.executor, "_shell") as shell:
            turn = self.planner.think("reboot then query")
        self.assertIsNotNone(self.executor.pending)
        self.assertEqual(len(turn.actions), 1)
        shell.assert_not_called()

    def test_invalid_arguments_and_unoffered_tools_are_not_executed(self):
        for call in (tool(arguments="[]"), tool(arguments="not json"), tool(name="run_shell")):
            with self.subTest(call=call), mock.patch("omarchy_voice.planner._chat", side_effect=[completion("", [call]), completion()]), \
                 mock.patch.object(self.executor, "call") as execute:
                self.planner.think("query")
                execute.assert_not_called()

    def test_original_policy_still_denies_shell_escalation(self):
        cfg = local_config(allow_shell=True)
        executor = Executor(cfg)
        with mock.patch("omarchy_voice.planner._chat", side_effect=[
            completion("", [tool("run_shell", '{"command":"sudo id"}')]), completion("Refused")]), \
             mock.patch.object(executor, "_shell") as shell:
            Planner(cfg, executor).think("do something")
        self.assertTrue(any("DENIED" in line for line in executor.transcript))
        shell.assert_not_called()

    def test_malformed_response_is_an_error_not_success(self):
        for data in ({}, {"choices": []}, completion(["not text"]), completion("", "not a list")):
            with self.subTest(data=data), mock.patch("omarchy_voice.planner._chat", return_value=data):
                self.assertTrue(self.planner.think("query").error)

    def test_step_limit_does_not_claim_success(self):
        self.planner.config.max_turns = 1
        with mock.patch("omarchy_voice.planner._chat", return_value=completion("will do", [tool()])), \
             mock.patch.object(self.executor, "call", return_value=Result(True, "[]")):
            self.assertIn("Maximum", self.planner.think("query").error)


class CaptureContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.cfg = local_config()

    def test_voxtype_uses_per_recording_file_not_clipboard(self):
        recording = capture.VoxtypeCapture(self.cfg, self.path)
        with mock.patch.object(capture, "voxtype_state", return_value="idle"), \
             mock.patch.object(capture, "voxtype_command") as command:
            recording.start()
        self.assertEqual(command.call_args.args[1:4], ("record", "start", "--output-file"))
        self.assertEqual(Path(command.call_args.args[-1]).parent, self.path)

    def test_voxtype_refuses_busy_and_never_cancels_unowned_recording(self):
        recording = capture.VoxtypeCapture(self.cfg, self.path)
        with mock.patch.object(capture, "voxtype_state", return_value="recording"), \
             mock.patch.object(capture, "voxtype_command") as command:
            with self.assertRaises(providers.ProviderError):
                recording.start()
            recording.cancel()
        command.assert_not_called()

    def test_finished_transcript_is_read_only_after_idle(self):
        recording = capture.VoxtypeCapture(self.cfg, self.path)
        recording.recording = True
        recording.path.write_text("  abra o navegador  ")
        with mock.patch.object(capture, "voxtype_command"), \
             mock.patch.object(capture, "voxtype_state", side_effect=["transcribing", "idle"]), \
             mock.patch.object(capture.time, "sleep"):
            self.assertEqual(recording.finish(lambda: False), "abra o navegador")

    def test_oversized_transcript_is_refused(self):
        recording = capture.VoxtypeCapture(self.cfg, self.path)
        recording.path.write_text("x" * (capture.TRANSCRIPT_LIMIT + 1))
        with mock.patch.object(capture, "voxtype_command"), \
             mock.patch.object(capture, "voxtype_state", return_value="idle"), self.assertRaises(providers.ProviderError):
            recording.finish(lambda: False)

    def test_voxtype_status_validates_schema(self):
        for value in ('{"class":"idle"}', '{}', '{"class":[]}'):
            with mock.patch.object(capture, "voxtype_command", return_value=value):
                if value == '{"class":"idle"}':
                    self.assertEqual(capture.voxtype_state(self.cfg), "idle")
                else:
                    with self.assertRaises(providers.ProviderError):
                        capture.voxtype_state(self.cfg)

    def test_multipart_has_audio_and_language(self):
        body, content_type = capture.multipart_audio(replace(self.cfg, stt_language="pt"), b"WAV")
        self.assertIn(b'name="file"; filename="speech.wav"', body)
        self.assertIn(b"\r\npt\r\n", body)
        self.assertIn(b"WAV", body)
        self.assertIn("boundary=", content_type)

    def test_transcription_api_is_explicit_and_bounded(self):
        path = self.path / "audio.wav"
        path.write_bytes(b"WAV")
        cfg = replace(self.cfg, stt_provider="compatible", stt_base_url="http://localhost:8000/v1")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "secret"}, clear=True), \
             mock.patch.object(capture, "request_bytes", return_value=b'{"text":" Ola "}') as request:
            self.assertEqual(capture.transcribe_api(cfg, path), "Ola")
        self.assertNotIn("Authorization", request.call_args.args[2])
        path.write_bytes(b"x" * (capture.CLIP_LIMIT + 1))
        with self.assertRaises(providers.ProviderError):
            capture.transcribe_api(cfg, path)

    def test_cancelled_recording_is_not_uploaded(self):
        recording = capture.ApiCapture(self.cfg, self.path)
        recording.process = mock.Mock()
        recording.process.poll.return_value = None
        with mock.patch.object(capture, "stop_process"), mock.patch.object(capture, "transcribe_api") as upload:
            self.assertEqual(recording.finish(lambda: True), "")
        upload.assert_not_called()

    def test_no_unknown_provider_fallback(self):
        with self.assertRaises(providers.ProviderError):
            capture.make_capture(replace(self.cfg, stt_provider="typo"), self.path)


class SpeechContracts(unittest.TestCase):
    def test_piper_model_path_and_no_traversal(self):
        cfg = local_config(models_dir="/tmp/models")
        self.assertEqual(speech.model_path(cfg), Path("/tmp/models/piper/pt_BR-faber-medium.onnx"))
        with self.assertRaises(providers.ProviderError):
            speech.model_path(replace(cfg, tts_voice="../../secret"))

    def test_pcm_sample_rate_and_format_are_preserved(self):
        with wave.open(io.BytesIO(speech.pcm_wav(b"\0\0" * 240, 24000)), "rb") as wav:
            self.assertEqual((wav.getframerate(), wav.getsampwidth(), wav.getnchannels()), (24000, 2, 1))
        with self.assertRaises(providers.ProviderError):
            speech.pcm_wav(b"a", 24000)

    def test_openai_and_elevenlabs_wire_contracts(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "o", "ELEVENLABS_API_KEY": "e"}, clear=True), \
             mock.patch.object(speech, "request_bytes", return_value=b"\0\0") as request:
            speech.api_speech(local_config(tts_provider="openai", tts_voice="alloy"), "Ola")
            self.assertEqual(json.loads(request.call_args.args[1])["response_format"], "wav")
            self.assertEqual(request.call_args.args[2]["Authorization"], "Bearer o")
            wav = speech.api_speech(local_config(tts_provider="elevenlabs", tts_voice="voice-id"), "Ola")
            self.assertTrue(request.call_args.args[0].endswith("voice-id?output_format=pcm_24000"))
            self.assertEqual(request.call_args.args[2]["xi-api-key"], "e")
            self.assertTrue(wav.startswith(b"RIFF"))

    def test_compatible_speech_requires_explicit_endpoint(self):
        with self.assertRaises(providers.ProviderError):
            speech.api_speech(local_config(tts_provider="compatible"), "Ola")

    def test_none_mouth_never_loads_or_connects(self):
        mouth = speech.Mouth(local_config())
        with mock.patch.object(mouth.piper, "load") as load, mock.patch.object(speech, "api_speech") as api:
            mouth.speak("Ola")
        load.assert_not_called()
        api.assert_not_called()

    def test_worker_uses_one_loaded_voice_for_multiple_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            messages = [json.dumps({"text": text, "output": str(Path(directory) / f"{i}.wav")})
                        for i, text in enumerate(("Primeiro", "Segundo"))]
            voice = mock.Mock()
            def synthesize(text, wav):
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(22050); wav.writeframes(b"\0\0")
            voice.synthesize_wav.side_effect = synthesize
            replies = []
            serve(voice, messages, replies.append)
            self.assertEqual(replies, [{"ok": True}, {"ok": True}])
            self.assertEqual(voice.synthesize_wav.call_count, 2)

    def test_worker_rejects_malformed_requests(self):
        voice, replies = mock.Mock(), []
        serve(voice, ['not json', '{"text":""}'], replies.append)
        self.assertEqual(len(replies), 2)
        self.assertTrue(all("error" in item for item in replies))
        voice.synthesize_wav.assert_not_called()

    def test_piper_reuses_live_worker_and_expires(self):
        worker = speech.PiperWorker(local_config(tts_idle_seconds=1))
        worker.process = mock.Mock()
        worker.process.poll.return_value = None
        with mock.patch.object(speech.subprocess, "Popen") as spawn:
            worker.load()
        spawn.assert_not_called()
        worker.last_used = time.monotonic() - 5
        with mock.patch.object(worker, "close") as close:
            worker.expire()
        close.assert_called_once()

    def test_worker_process_lifecycle_with_fake_local_voice(self):
        # Real pipes/process lifecycle, but no model download or third-party runtime.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "fake.onnx").touch()
            (folder / "fake.onnx.json").write_text("{}")
            (folder / "piper.py").write_text(
                "class PiperVoice:\n"
                " @classmethod\n"
                " def load(cls, path, use_cuda=False): return cls()\n"
                " def synthesize_wav(self, text, wav):\n"
                "  wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(22050)\n"
                "  wav.writeframes(b'\\0\\0' * 32)\n")
            worker = speech.PiperWorker(local_config(tts_model=str(folder / "fake.onnx")))
            with mock.patch.dict(os.environ, {"PYTHONPATH": directory}):
                try:
                    worker.load()
                    pid = worker.process.pid
                    worker.synthesize("Ola", folder / "first.wav", lambda: False)
                    worker.synthesize("Outra frase", folder / "second.wav", lambda: False)
                    self.assertEqual(worker.process.pid, pid)
                    with wave.open(str(folder / "second.wav")) as wav:
                        self.assertEqual(wav.getframerate(), 22050)
                    child = worker.process
                finally:
                    worker.close()
                self.assertIsNotNone(child.poll())
                self.assertIsNone(worker.process)

    def test_missing_model_does_not_trigger_download(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = speech.PiperWorker(local_config(models_dir=directory))
            with mock.patch.object(speech.subprocess, "Popen") as spawn, self.assertRaises(providers.ProviderError):
                worker.load()
            spawn.assert_not_called()

    def test_stop_process_reaps_stubborn_child(self):
        child = mock.Mock()
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("piper", 2), 0]
        speech.stop_process(child)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        child.stdout.close.assert_called_once()


class PipelineLifecycle(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for target, replacement in (("RUNTIME_DIR", Path(self.directory.name)), ("Feedback", mock.Mock())):
            patch = mock.patch.object(pipeline, target, replacement)
            patch.start(); self.addCleanup(patch.stop)
        self.session = pipeline.PipelineSession(local_config())
        self.session.mouth = mock.Mock()
        self.session.planner = mock.Mock()
        self.session.planner.think.return_value = Turn("query", reply="Done")
        self.addCleanup(self.session.close)

    def test_toggle_is_capture_then_transcribe_then_act_then_speak(self):
        order = []
        recording = mock.Mock()
        recording.start.side_effect = lambda: order.append("record")
        recording.finish.side_effect = lambda _: order.append("transcribe") or "abra"
        recording.cancel.side_effect = lambda: order.append("mic-off")
        self.session.planner.think.side_effect = lambda *_: order.append("brain") or Turn("abra", reply="Abri")
        self.session.mouth.speak.side_effect = lambda *_: order.append("mouth")
        with mock.patch.object(pipeline, "make_capture", return_value=recording):
            self.assertEqual(self.session.control("toggle"), "accepted")
            self.session.step(0)
            self.assertEqual(self.session.status, "listening")
            self.session.control("toggle")
            self.session.step(0)
        self.assertEqual(order, ["record", "transcribe", "mic-off", "brain", "mouth"])
        self.assertEqual(self.session.status, "idle")
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_cancel_removes_queued_request(self):
        self.session.control("say execute")
        self.session.control("cancel")
        self.session.step(0)
        self.session.planner.think.assert_not_called()
        self.assertEqual(self.session.status, "idle")

    def test_busy_rejects_duplicate_request(self):
        self.assertEqual(self.session.control("say first"), "accepted")
        self.assertIn("busy", self.session.control("say second"))

    def test_toggle_while_busy_requests_cancellation(self):
        self.session.control("say query")
        self.assertIn("cancellation requested", self.session.control("toggle"))
        self.session.step(0)
        self.session.planner.think.assert_not_called()

    def test_recording_deadline_discards_not_executes(self):
        self.session.capture = mock.Mock()
        self.session.recorded_at = time.monotonic() - 100
        self.session.status = "listening"
        self.session.step(0)
        self.session.step(0)
        self.session.planner.think.assert_not_called()
        self.assertIsNone(self.session.capture)

    def test_pending_action_can_only_be_confirmed_through_local_control(self):
        self.session.executor.pending = ("omarchy_cli", {"command": "system reboot"})
        self.session.status = "confirm"
        self.assertIn("busy", self.session.control("say confirm"))
        with mock.patch.object(self.session.executor, "run_pending", return_value=Result(True, "done")) as execute:
            self.session.control("confirm")
            self.session.step(0)
        execute.assert_called_once()
        self.session.planner.think.assert_not_called()

    def test_cancel_in_provider_prevents_reply(self):
        def think(*_):
            self.session.control("cancel")
            return Turn("query", reply="must not speak")
        self.session.planner.think.side_effect = think
        self.session.control("say query")
        self.session.step(0)
        self.session.mouth.speak.assert_not_called()

    def test_error_cleans_up_recording_and_allows_next_turn(self):
        recording = mock.Mock()
        recording.start.side_effect = providers.ProviderError("unavailable")
        with mock.patch.object(pipeline, "make_capture", return_value=recording):
            self.session.control("start")
            self.session.step(0)
        self.assertEqual(self.session.status, "idle")
        self.assertIsNone(self.session.capture)
        self.assertEqual(self.session.control("say next"), "accepted")

    def test_model_load_is_only_for_piper(self):
        self.session.control("models load mouth")
        self.session.step(0)
        self.session.mouth.piper.load.assert_not_called()

    def test_no_transcript_in_log_by_default(self):
        self.session.control("say very private text")
        self.session.step(0)
        self.assertNotIn("very private text", str(self.session.feedback.log.call_args_list))


class SetupAndModels(unittest.TestCase):
    def test_legacy_defaults_and_friendly_aliases(self):
        self.assertEqual(Config().voice_mode, "realtime")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('[voice]\nmode="pipeline"\n[ears]\nprovider="voxtype"\ndevice="mic"\n'
                            '[brain]\nprovider="ollama"\nmodel="m"\n[mouth]\nprovider="piper"\nnotify=false\n')
            cfg = config.load(path)
        self.assertEqual((cfg.stt_provider, cfg.device, cfg.tts_provider, cfg.notify), ("voxtype", "mic", "piper", False))
        self.assertEqual(cfg.unknown_keys, [])

    def test_setup_preserves_safety_unknowns_and_replaces_aliases(self):
        original = {"stt_provider": "old", "stt": {"provider": "old"}, "ears": {"provider": "old", "device": "mic"},
                    "hands": {"allow_shell": False, "deny_patterns": ["private"]}, "future": {"setting": [1, 2]}}
        updated = setup.merged_settings(original, {"ears": {"provider": "voxtype"}})
        self.assertEqual(updated["hands"], original["hands"])
        self.assertEqual(updated["future"], original["future"])
        self.assertNotIn("stt_provider", updated)
        self.assertNotIn("provider", updated["stt"])
        self.assertEqual(updated["ears"]["device"], "mic")
        self.assertEqual(tomllib.loads(setup.dump_toml(updated)), updated)

    def test_save_is_private_and_backup_retains_original_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            original = '# keep\n[hands]\nallow_shell=false\n'
            path.write_text(original)
            setup.save_settings(path, {"voice": {"mode": "pipeline"}})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            backup = next(path.parent.glob("*.bak-*"))
            self.assertEqual(backup.read_text(), original)
            self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
            self.assertFalse(config.load(path).allow_shell)

    def test_secret_file_preserves_unrelated_keys_and_rejects_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "env"
            path.write_text("OLD=keep\nNEW=replace\n")
            setup.save_secrets(path, {"NEW": "opaque-token"})
            self.assertEqual(path.read_text(), "OLD=keep\nNEW=opaque-token\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            for secret in ("key\nEVIL=x", "$(command)"):
                with self.assertRaises(providers.ProviderError):
                    setup.save_secrets(path, {"NEW": secret})

    def test_symlink_configuration_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("original")
            link = Path(directory) / "link"
            link.symlink_to(target)
            with self.assertRaises(providers.ProviderError):
                setup.atomic_private(link, "replacement")
            self.assertEqual(target.read_text(), "original")

    def test_invalid_pipeline_settings_fail_before_network(self):
        for changes in ({"voice_mode": "wrong"}, {"brain_timeout": 0}, {"max_turns": 1.5},
                        {"stt_provider": []}, {"brain_context": 0}, {"brain_extra": []}):
            with self.subTest(changes=changes):
                self.assertTrue(pipeline.configuration_errors(local_config(**changes)))

    def test_local_readiness_requires_no_openai_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(pipeline, "voxtype_command", return_value="--output-file"), \
             mock.patch.object(pipeline, "voxtype_state", return_value="idle"):
            self.assertEqual(pipeline.check_ready(local_config()), [])

    def test_ollama_model_operations(self):
        with mock.patch.object(models, "request_json", return_value={"ok": True}) as request:
            models.manage(local_config(), "download", "brain")
            self.assertTrue(request.call_args.args[0].endswith("/api/pull"))
            self.assertEqual(request.call_args.args[1]["model"], "qwen3:4b")
            models.manage(local_config(), "unload", "brain")
            self.assertEqual(request.call_args.args[1]["keep_alive"], 0)
            models.manage(local_config(), "list", "brain")
            self.assertIsNone(request.call_args.args[1])

    def test_piper_download_uses_managed_directory_and_no_shell(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(models, "native_command") as command:
            cfg = local_config(tts_provider="piper", models_dir=directory)
            models.manage(cfg, "download", "mouth")
            self.assertEqual(command.call_args.args[0][-2:], ["--data-dir", str(Path(directory) / "piper")])
            self.assertIn("piper.download_voices", command.call_args.args[0])

    def test_native_voxtype_download_does_not_duplicate_cache(self):
        with mock.patch.object(models, "native_command") as command:
            models.manage(local_config(), "download", "ears")
        self.assertEqual(command.call_args.args[0], ["voxtype", "setup", "--download"])

    def test_piper_load_and_unload_go_to_resident_daemon(self):
        with mock.patch.object(models, "send_control", return_value="accepted") as send:
            models.manage(local_config(tts_provider="piper"), "load", "mouth")
        send.assert_called_once_with("models load mouth")

    def test_cli_selects_pipeline_without_realtime(self):
        with mock.patch.object(pipeline, "run", return_value=0) as run, \
             mock.patch.object(cli.realtime_mod, "run") as realtime:
            self.assertEqual(cli.cmd_run(None, local_config()), 0)
        run.assert_called_once()
        realtime.assert_not_called()

    def test_cli_retains_default_realtime(self):
        with mock.patch.object(cli.realtime_mod, "run", return_value=0) as run:
            self.assertEqual(cli.cmd_run(None, Config()), 0)
        run.assert_called_once()

    def test_setup_default_local_path_never_downloads_or_installs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            cfg = local_config(models_dir=str(Path(directory) / "models"))
            with mock.patch("builtins.input", side_effect=[""] * 14), \
                 mock.patch("builtins.print"), mock.patch.object(setup.cfg, "ENV_FILE", Path(directory) / "env"), \
                 mock.patch.object(setup, "native_command") as install, mock.patch.object(setup, "manage") as download:
                self.assertEqual(setup.run(path, cfg), 0)
            saved = config.load(path)
            self.assertEqual((saved.voice_mode, saved.brain_provider, saved.brain_api_key_env), ("pipeline", "ollama", ""))
            install.assert_not_called()
            download.assert_not_called()

    def test_cancellation_stops_owned_transcription_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            recording = capture.VoxtypeCapture(local_config(), Path(directory))
            recording.recording = True
            with mock.patch.object(capture, "voxtype_command") as command, \
                 mock.patch.object(capture, "voxtype_state", return_value="idle"):
                self.assertEqual(recording.finish(lambda: True), "")
                self.assertTrue(recording.recording)
                recording.cancel()
                self.assertFalse(recording.recording)
            self.assertEqual([call.args[1:] for call in command.call_args_list], [("record", "stop"), ("record", "cancel")])

    def test_cli_commands_parse(self):
        parser = cli.build_parser()
        for argv in (["setup"], ["models", "download", "mouth"], ["speak", "Ola"]):
            self.assertTrue(callable(parser.parse_args(argv).func))


if __name__ == "__main__":
    unittest.main()
