"""Gradio UI for Jarvis. Runs on the Brev box, exposed via brev port-forward.

Audio is continuously streamed from the browser microphone. The stream handler
buffers short numpy chunks server-side and enqueues WAV segments for a
background worker, so Whisper + agent planning do not stall the mic stream.

Manual overrides remain available:
  - Inject text chunk (no mic)
  - Reset memory
  - Reload seed facts
"""
from __future__ import annotations

import io
import os
import threading
import time
import wave
from math import gcd
from queue import Empty, Queue
from typing import Optional

import gradio as gr
import numpy as np

try:
    from scipy.signal import resample_poly
except Exception:  # scipy is optional; numpy interpolation fallback is below.
    resample_poly = None

import agent
import memory
import transcription
import tts
from tools import calendar as cal_tool


# --- streaming audio tuning ---
TARGET_SAMPLE_RATE = 16_000
FLUSH_AFTER_S = 3.0
SILENCE_AFTER_S = 1.0
MIN_UTTERANCE_S = 0.45
SILENCE_RMS = float(os.getenv("JARVIS_SILENCE_RMS", "0.012"))
DROP_RMS = float(os.getenv("JARVIS_DROP_RMS", "0.006"))
MAX_BUFFER_S = 12.0
OVERLAP_S = 0.20


# --- live state for the UI ---
_LIVE_LOG: list[str] = []
_LAST_QUESTION = ""
_LAST_RESULT = ""


# --- background worker state ---
_AUDIO_JOBS: Queue = Queue()
_AUDIO_RESULTS: Queue = Queue()
_WORKER_STARTED = False
_WORKER_BUSY = False
_WORKER_LOCK = threading.Lock()


def log(line: str) -> None:
    _LIVE_LOG.append(f"{time.strftime('%H:%M:%S')}  {line}")
    if len(_LIVE_LOG) > 200:
        del _LIVE_LOG[:50]


def render_log() -> str:
    return "\n".join(_LIVE_LOG[-60:]) or "(no events yet)"


def render_episodic() -> str:
    items = memory.episodic_recent(20)
    if not items:
        return "(empty)"
    return "\n".join(f"[{c.get('speaker','?')}] {c['text']}" for c in items)


def render_semantic() -> str:
    items = memory.semantic_all()
    if not items:
        return "(empty - pre-warm with Reload Seeds, or have a conversation)"
    lines = []
    for it in items:
        fact = it["text"].split(" - ", 1)[-1].split(" — ", 1)[-1]
        lines.append(
            f"  • [{it.get('type','?')}] {it.get('subject','?')}: "
            f"{fact}  (conf {it.get('confidence', 0):.2f})"
        )
    return "\n".join(lines)


def render_calendar() -> str:
    booked = cal_tool.list_booked()
    if not booked:
        return "(no events booked yet)"
    return "\n".join(
        f"  • {b['title']}  {b['start']} -> {b['end']}  attendees={b['attendees']}"
        for b in booked
    )


def _new_audio_state() -> dict:
    now = time.time()
    return {
        "buffer": np.empty(0, dtype=np.float32),
        "last_voice_at": now,
        "last_flush_at": now,
        "had_voice": False,
        "chunk_index": 0,
        "paused": False,
    }


def _enum_value(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _queue_depth() -> int:
    return _AUDIO_JOBS.qsize() + (1 if _WORKER_BUSY else 0)


def _render_listening_status(state: Optional[dict]) -> str:
    state = state or _new_audio_state()
    if state.get("paused"):
        return "### 🤫 Paused"

    buffered_s = len(state.get("buffer", [])) / TARGET_SAMPLE_RATE
    pending = _queue_depth()
    if pending:
        return f"### 👂 Listening  |  processing {pending} audio chunk(s)  |  buffer {buffered_s:.1f}s"
    return f"### 👂 Listening  |  buffer {buffered_s:.1f}s"


def _main_outputs(refresh_memory: bool = True):
    episodic = render_episodic() if refresh_memory else gr.update()
    semantic = render_semantic() if refresh_memory else gr.update()
    return render_log(), episodic, semantic, _LAST_QUESTION, _LAST_RESULT


# ============================================================================
# Audio conversion
# ============================================================================

def _to_mono_float32(audio_data) -> np.ndarray:
    samples = np.asarray(audio_data)
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)

    if samples.ndim == 2:
        # Gradio usually returns (samples, channels), but handle channels-first.
        if samples.shape[0] <= 2 and samples.shape[1] > samples.shape[0]:
            samples = samples.T
        samples = samples.mean(axis=1)

    if np.issubdtype(samples.dtype, np.integer):
        max_abs = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / float(max_abs)
    else:
        samples = samples.astype(np.float32)

    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(samples, -1.0, 1.0)


def _resample_to_target(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    if samples.size == 0 or sample_rate == TARGET_SAMPLE_RATE:
        return samples.astype(np.float32, copy=False)

    if resample_poly is not None:
        factor = gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // factor
        down = int(sample_rate) // factor
        return resample_poly(samples, up, down).astype(np.float32)

    duration_s = samples.size / float(sample_rate)
    target_len = max(1, int(round(duration_s * TARGET_SAMPLE_RATE)))
    src_x = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _float32_to_wav_bytes(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(TARGET_SAMPLE_RATE)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))


def _parse_audio_input(audio) -> tuple[int, np.ndarray] | None:
    if audio is None:
        return None
    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, data = audio
        return int(sample_rate), _to_mono_float32(data)
    raise ValueError(f"unexpected audio payload from Gradio: {type(audio)!r}")


# ============================================================================
# Background audio processing
# ============================================================================

def _ensure_audio_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        thread = threading.Thread(target=_audio_worker_loop, daemon=True)
        thread.start()
        _WORKER_STARTED = True


def _audio_worker_loop() -> None:
    global _WORKER_BUSY
    while True:
        job = _AUDIO_JOBS.get()
        _WORKER_BUSY = True
        try:
            text = transcription.transcribe_bytes(job["wav_bytes"], filename=job["filename"])
            text = text.strip()
            if not text:
                _AUDIO_RESULTS.put({
                    "kind": "log",
                    "message": f"[audio] whisper returned empty for {job['duration_s']:.1f}s chunk",
                })
                continue

            _AUDIO_RESULTS.put({
                "kind": "transcript",
                "speaker": job["speaker"],
                "text": text,
            })
            out = agent.handle_chunk(text, speaker=job["speaker"])
            _AUDIO_RESULTS.put({"kind": "agent_result", "out": out})
        except Exception as e:
            _AUDIO_RESULTS.put({"kind": "log", "message": f"[audio worker error] {e}"})
        finally:
            _WORKER_BUSY = False
            _AUDIO_JOBS.task_done()


def _apply_agent_output(out: dict) -> None:
    global _LAST_QUESTION, _LAST_RESULT

    decision = out.get("decision")
    if decision:
        route = _enum_value(decision.get("route", out.get("action", "?"))).upper()
        confidence = float(decision.get("confidence", 0.0))
        reason = decision.get("reason", "")
        log(f"TRIAGE: {route} ({confidence:.2f}) - {reason}")

    if out.get("action") == "act" and "question" in out:
        _LAST_QUESTION = out["question"]
        log(f"INTERJECT: {out['question']}")
    elif out.get("action") == "executed":
        result = out.get("result", {})
        _LAST_RESULT = (
            f"{result.get('status')}: {result.get('message')} "
            f"(id={result.get('artifact_id')})"
        )
        log(f"EXECUTED: {_LAST_RESULT}")
    elif out.get("action") == "rejected":
        log("REJECTED: user rejected last proposal")
    elif out.get("action") == "unclear":
        log("CONFIRMATION: unclear response; waiting for yes/no/modify")


def _drain_audio_results() -> bool:
    drained = False
    while True:
        try:
            event = _AUDIO_RESULTS.get_nowait()
        except Empty:
            return drained

        drained = True
        if event["kind"] == "log":
            log(event["message"])
        elif event["kind"] == "transcript":
            log(f"[{event['speaker']}] {event['text']}")
        elif event["kind"] == "agent_result":
            _apply_agent_output(event["out"])
        _AUDIO_RESULTS.task_done()


def _flush_audio_buffer(state: dict, speaker: str, reason: str) -> bool:
    samples = state.get("buffer")
    if samples is None or samples.size == 0:
        return False

    duration_s = samples.size / TARGET_SAMPLE_RATE
    rms = _rms(samples)
    keep_overlap = reason == "duration" and samples.size > int(OVERLAP_S * TARGET_SAMPLE_RATE)
    overlap = samples[-int(OVERLAP_S * TARGET_SAMPLE_RATE):].copy() if keep_overlap else np.empty(0, dtype=np.float32)

    state["buffer"] = overlap
    state["last_flush_at"] = time.time()
    state["last_voice_at"] = time.time()
    state["had_voice"] = bool(overlap.size and _rms(overlap) >= SILENCE_RMS)

    if duration_s < MIN_UTTERANCE_S or rms < DROP_RMS:
        return False

    chunk_id = state["chunk_index"]
    state["chunk_index"] += 1
    wav_bytes = _float32_to_wav_bytes(samples)
    _AUDIO_JOBS.put({
        "wav_bytes": wav_bytes,
        "speaker": speaker,
        "filename": f"stream_{chunk_id}.wav",
        "duration_s": duration_s,
        "reason": reason,
    })
    log(f"[audio] queued {duration_s:.1f}s chunk ({reason}, rms={rms:.3f})")
    return True


# ============================================================================
# Handlers
# ============================================================================

def on_audio_stream(audio, state: Optional[dict], speaker: str):
    _ensure_audio_worker()
    state = state or _new_audio_state()
    drained = _drain_audio_results()

    if state.get("paused"):
        live, ep, sem, question, result = _main_outputs(refresh_memory=drained)
        return state, live, ep, sem, question, result, _render_listening_status(state)

    try:
        parsed = _parse_audio_input(audio)
    except Exception as e:
        log(f"[audio stream error] {e}")
        live, ep, sem, question, result = _main_outputs(refresh_memory=True)
        return state, live, ep, sem, question, result, _render_listening_status(state)

    queued = False
    if parsed is not None:
        sample_rate, samples = parsed
        samples = _resample_to_target(samples, sample_rate)
        if samples.size:
            state["buffer"] = np.concatenate([state.get("buffer", np.empty(0, dtype=np.float32)), samples])
            max_samples = int(MAX_BUFFER_S * TARGET_SAMPLE_RATE)
            if state["buffer"].size > max_samples:
                state["buffer"] = state["buffer"][-max_samples:]

            chunk_rms = _rms(samples)
            now = time.time()
            if chunk_rms >= SILENCE_RMS:
                state["last_voice_at"] = now
                state["had_voice"] = True

            buffered_s = state["buffer"].size / TARGET_SAMPLE_RATE
            silence_s = now - state["last_voice_at"]
            if buffered_s >= FLUSH_AFTER_S:
                queued = _flush_audio_buffer(state, speaker, "duration")
            elif state.get("had_voice") and buffered_s >= MIN_UTTERANCE_S and silence_s >= SILENCE_AFTER_S:
                queued = _flush_audio_buffer(state, speaker, "silence")

    live, ep, sem, question, result = _main_outputs(refresh_memory=drained or queued)
    return state, live, ep, sem, question, result, _render_listening_status(state)


def on_toggle_listening(state: Optional[dict], speaker: str):
    _ensure_audio_worker()
    state = state or _new_audio_state()
    if state.get("paused"):
        state["paused"] = False
        state["last_voice_at"] = time.time()
        log("[audio] resumed continuous listening")
        button_update = gr.update(value="Pause")
    else:
        _flush_audio_buffer(state, speaker, "pause")
        state["paused"] = True
        log("[audio] paused continuous listening")
        button_update = gr.update(value="Resume")

    _drain_audio_results()
    live, ep, sem, question, result = _main_outputs(refresh_memory=True)
    return state, button_update, _render_listening_status(state), live, ep, sem, question, result


def on_poll_updates(state: Optional[dict]):
    state = state or _new_audio_state()
    drained = _drain_audio_results()
    live, ep, sem, question, result = _main_outputs(refresh_memory=drained)
    return state, live, ep, sem, question, result, _render_listening_status(state), render_calendar()


def on_text_inject(text: str, speaker: str):
    _drain_audio_results()
    if not text.strip():
        live, ep, sem, question, result = _main_outputs(refresh_memory=False)
        return live, ep, sem, question, result
    log(f"[{speaker}] {text}")
    out = agent.handle_chunk(text, speaker=speaker)
    _apply_agent_output(out)
    return _main_outputs(refresh_memory=True)


def on_speak_question() -> Optional[str]:
    """TTS the current pending question to a WAV the browser can play."""
    if not _LAST_QUESTION:
        return None
    try:
        path = "/tmp/last_q.wav"
        tts.synthesize_to_file(_LAST_QUESTION, path)
        return path
    except Exception as e:
        log(f"[tts error] {e}")
        return None


def on_reset_memory(state: Optional[dict]):
    global _LAST_QUESTION, _LAST_RESULT
    memory.reset_all()
    cal_tool.reset_session()
    _LIVE_LOG.clear()
    _LAST_QUESTION = ""
    _LAST_RESULT = ""
    state = _new_audio_state()
    log("[reset] memory + calendar cleared")
    live, ep, sem, question, result = _main_outputs(refresh_memory=True)
    return state, live, ep, sem, question, result, render_calendar(), _render_listening_status(state), gr.update(value="Pause")


def on_seed():
    n = memory.seed_from_file(os.path.join(os.path.dirname(__file__), "data/seed_facts.json"))
    log(f"[seed] +{n} semantic facts")
    return render_log(), render_episodic(), render_semantic()


# ============================================================================
# UI
# ============================================================================

def build_ui():
    with gr.Blocks(title="Jarvis - Track 5 Agent") as app:
        initial_audio_state = _new_audio_state()
        audio_state = gr.State(initial_audio_state)
        gr.Markdown("# Jarvis - Proactive Scheduling Agent\n*TOA Track 5 · NemoClaw + vLLM + Red Hat AI*")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Continuous Audio")
                listen_status = gr.Markdown(_render_listening_status(initial_audio_state))
                audio_in = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Live microphone",
                )
                with gr.Row():
                    speaker_audio = gr.Dropdown(["user", "alex", "sam", "julie"], value="user", label="Speaker")
                    btn_pause = gr.Button("Pause")

                gr.Markdown("### Text inject (no mic)")
                text_in = gr.Textbox(placeholder="Yo, want to grab drinks at 7:30?", lines=1)
                with gr.Row():
                    speaker_text = gr.Dropdown(["user", "alex", "sam", "julie"], value="user", label="Speaker")
                    btn_text = gr.Button("Inject", variant="primary")

                gr.Markdown("### Pending interjection")
                question_box = gr.Textbox(label="Agent's question (TTS source)", interactive=False)
                with gr.Row():
                    btn_speak = gr.Button("🔊 Speak it")
                    audio_out = gr.Audio(label="TTS", interactive=False)
                result_box = gr.Textbox(label="Last action result", interactive=False)

                gr.Markdown("### Controls")
                with gr.Row():
                    btn_reset = gr.Button("🗑 Reset memory")
                    btn_seed = gr.Button("✨ Reload seed facts")

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Live log"):
                        live_box = gr.Textbox(value=render_log(), label="Events", lines=20, interactive=False, max_lines=50)
                    with gr.Tab("Episodic memory (rolling 10 min)"):
                        ep_box = gr.Textbox(value=render_episodic(), label="Recent transcripts", lines=20, interactive=False, max_lines=50)
                    with gr.Tab("Semantic memory (what I've learned)"):
                        sem_box = gr.Textbox(value=render_semantic(), label="Distilled facts", lines=20, interactive=False, max_lines=50)
                    with gr.Tab("Calendar"):
                        cal_box = gr.Textbox(value=render_calendar(), label="Booked events", lines=12, interactive=False)

        audio_in.stream(
            on_audio_stream,
            inputs=[audio_in, audio_state, speaker_audio],
            outputs=[audio_state, live_box, ep_box, sem_box, question_box, result_box, listen_status],
        )
        btn_pause.click(
            on_toggle_listening,
            inputs=[audio_state, speaker_audio],
            outputs=[audio_state, btn_pause, listen_status, live_box, ep_box, sem_box, question_box, result_box],
        )
        btn_text.click(
            on_text_inject,
            inputs=[text_in, speaker_text],
            outputs=[live_box, ep_box, sem_box, question_box, result_box],
        )
        btn_speak.click(on_speak_question, outputs=audio_out)
        btn_reset.click(
            on_reset_memory,
            inputs=[audio_state],
            outputs=[audio_state, live_box, ep_box, sem_box, question_box, result_box, cal_box, listen_status, btn_pause],
        )
        btn_seed.click(on_seed, outputs=[live_box, ep_box, sem_box])

        poll_timer = gr.Timer(0.75)
        poll_timer.tick(
            on_poll_updates,
            inputs=[audio_state],
            outputs=[audio_state, live_box, ep_box, sem_box, question_box, result_box, listen_status, cal_box],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )
