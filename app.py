"""Gradio UI for Jarvis. Runs on the Brev box, exposed via brev port-forward.

Three tabs:
  - Live: mic input, live transcript, agent state, last interjection.
  - Memory: episodic recents + full semantic dump (the 'what I've learned' panel).
  - Calendar: who's busy + what's been booked this session.

Manual overrides (for demo control):
  - Inject text chunk (no mic)
  - Reset memory
  - Reload seed facts
"""
from __future__ import annotations

import os
import time
import threading
from typing import Optional

import gradio as gr

import memory
import transcription
import tts
import agent
from tools import calendar as cal_tool


# --- live state for the UI ---
_LIVE_LOG: list[str] = []
_LAST_QUESTION = ""
_LAST_RESULT = ""


def log(line: str):
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
        return "(empty — pre-warm with Reload Seeds, or have a conversation)"
    lines = []
    for it in items:
        lines.append(f"  • [{it.get('type','?')}] {it.get('subject','?')}: {it['text'].split(' — ',1)[-1]}  (conf {it.get('confidence', 0):.2f})")
    return "\n".join(lines)


def render_calendar() -> str:
    booked = cal_tool.list_booked()
    if not booked:
        return "(no events booked yet)"
    return "\n".join(f"  • {b['title']}  {b['start']} → {b['end']}  attendees={b['attendees']}" for b in booked)


# ============================================================================
# Handlers
# ============================================================================

def on_audio(file_path: str | None, speaker: str):
    global _LAST_QUESTION, _LAST_RESULT
    if not file_path:
        return render_log(), render_episodic(), render_semantic(), _LAST_QUESTION, _LAST_RESULT
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        text = transcription.transcribe_bytes(audio_bytes, filename=os.path.basename(file_path))
    except Exception as e:
        log(f"[transcribe error] {e}")
        return render_log(), render_episodic(), render_semantic(), _LAST_QUESTION, _LAST_RESULT
    if not text.strip():
        log(f"(silence)")
        return render_log(), render_episodic(), render_semantic(), _LAST_QUESTION, _LAST_RESULT
    log(f"[{speaker}] {text}")
    out = agent.handle_chunk(text, speaker=speaker)
    if out.get("action") == "act" and "question" in out:
        _LAST_QUESTION = out["question"]
        log(f"❗ INTERJECT: {out['question']}")
    elif out.get("action") == "executed":
        r = out.get("result", {})
        _LAST_RESULT = f"✅ {r.get('status')}: {r.get('message')} (id={r.get('artifact_id')})"
        log(_LAST_RESULT)
    elif out.get("action") == "rejected":
        log("✗ user rejected last proposal")
    return render_log(), render_episodic(), render_semantic(), _LAST_QUESTION, _LAST_RESULT


def on_text_inject(text: str, speaker: str):
    global _LAST_QUESTION, _LAST_RESULT
    if not text.strip():
        return render_log(), render_episodic(), render_semantic(), _LAST_QUESTION, _LAST_RESULT
    log(f"[{speaker}] {text}")
    out = agent.handle_chunk(text, speaker=speaker)
    if out.get("action") == "act" and "question" in out:
        _LAST_QUESTION = out["question"]
        log(f"❗ INTERJECT: {out['question']}")
    elif out.get("action") == "executed":
        r = out.get("result", {})
        _LAST_RESULT = f"✅ {r.get('status')}: {r.get('message')} (id={r.get('artifact_id')})"
        log(_LAST_RESULT)
    elif out.get("action") == "rejected":
        log("✗ user rejected last proposal")
    return render_log(), render_episodic(), render_semantic(), _LAST_QUESTION, _LAST_RESULT


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


def on_reset_memory():
    memory.reset_all()
    cal_tool.reset_session()
    _LIVE_LOG.clear()
    log("[reset] memory + calendar cleared")
    return render_log(), render_episodic(), render_semantic(), "", ""


def on_seed():
    n = memory.seed_from_file(os.path.join(os.path.dirname(__file__), "data/seed_facts.json"))
    log(f"[seed] +{n} semantic facts")
    return render_log(), render_episodic(), render_semantic()


# ============================================================================
# UI
# ============================================================================

def build_ui():
    with gr.Blocks(title="Jarvis — Track 5 Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Jarvis — Proactive Scheduling Agent\n*TOA Track 5 · NemoClaw + vLLM + Red Hat AI*")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Audio in")
                audio_in = gr.Audio(sources=["microphone"], type="filepath",
                                    label="Speak a chunk, then release")
                with gr.Row():
                    speaker_audio = gr.Dropdown(["alex", "sam", "julie"], value="alex", label="Speaker")
                    btn_audio = gr.Button("Process audio", variant="primary")

                gr.Markdown("### Text inject (no mic)")
                text_in = gr.Textbox(placeholder="Yo, want to grab drinks at 7:30?", lines=1)
                with gr.Row():
                    speaker_text = gr.Dropdown(["alex", "sam", "julie"], value="alex", label="Speaker")
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
                        live_box = gr.Textbox(value=render_log(), label="Events",
                                              lines=20, interactive=False, max_lines=50)
                    with gr.Tab("Episodic memory (rolling 10 min)"):
                        ep_box = gr.Textbox(value=render_episodic(), label="Recent transcripts",
                                            lines=20, interactive=False, max_lines=50)
                    with gr.Tab("Semantic memory (what I've learned)"):
                        sem_box = gr.Textbox(value=render_semantic(), label="Distilled facts",
                                             lines=20, interactive=False, max_lines=50)
                    with gr.Tab("Calendar"):
                        cal_box = gr.Textbox(value=render_calendar(), label="Booked events",
                                             lines=12, interactive=False)

        # Wiring
        btn_audio.click(on_audio, inputs=[audio_in, speaker_audio],
                        outputs=[live_box, ep_box, sem_box, question_box, result_box])
        btn_text.click(on_text_inject, inputs=[text_in, speaker_text],
                       outputs=[live_box, ep_box, sem_box, question_box, result_box])
        btn_speak.click(on_speak_question, outputs=audio_out)
        btn_reset.click(on_reset_memory,
                        outputs=[live_box, ep_box, sem_box, question_box, result_box])
        btn_seed.click(on_seed, outputs=[live_box, ep_box, sem_box])

        # Auto-refresh calendar tab every 5s
        timer = gr.Timer(5)
        timer.tick(lambda: render_calendar(), outputs=cal_box)

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
