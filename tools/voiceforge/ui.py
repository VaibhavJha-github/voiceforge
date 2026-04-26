"""
VoiceForge Gradio UI — Multi-character batch TTS for Fish-Speech.

Tabs:
1. Voice Setup — Clone/manage character voices (Peter, Stewie, etc.)
2. Generate — Paste or upload scripts, batch generate audio per video
3. Settings — Model params, pause durations
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import gradio as gr
import numpy as np
import soundfile as sf
from loguru import logger

from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from tools.voiceforge.batch_processor import (
    VoiceProfile,
    create_zip,
    process_batch,
)
from tools.voiceforge.script_parser import (
    parse_scripts,
    scripts_summary,
    validate_scripts,
)

# Persistent voice storage
VOICES_DIR = Path("references")
VOICES_META = VOICES_DIR / "_voiceforge_meta.json"

EXAMPLE_SCRIPT = """=== VIDEO 1 ===
PETER: Hey Stewie, did you know that teachers can actually see when you paste from ChatGPT?
STEWIE: Wait, what? Are you serious? I've been doing that for literally every assignment.
PETER: Yeah bro, they've got these AI detectors now. It's getting pretty gnarly out there.
STEWIE: So what am I supposed to do? I can't actually write essays myself, that's barbaric.
PETER: Relax. There's this thing called EvadeGPT. It rewrites everything so it sounds like you actually wrote it.
STEWIE: And it actually works? The detectors can't catch it?
PETER: Nope. Zero detection. I've been using it all semester. Check the link in bio.

=== VIDEO 2 ===
PETER: Stewie, you look stressed mate. What's going on?
STEWIE: I just got an email from my professor. She wants to meet about my last paper.
PETER: Oof. Let me guess, you used raw ChatGPT?
STEWIE: How did you know?
PETER: Because that's what everyone does before they find EvadeGPT. Look, it takes your AI text and humanises it completely.
STEWIE: Humanises it? Like it won't show up on Turnitin?
PETER: Zero percent AI detection. Every time. It's actually insane.
STEWIE: Where do I get it?
PETER: Link in bio. You're welcome.
"""


def _load_voices_meta() -> dict:
    """Load saved voice metadata."""
    if VOICES_META.exists():
        try:
            return json.loads(VOICES_META.read_text())
        except Exception:
            return {}
    return {}


def _save_voices_meta(meta: dict):
    """Save voice metadata."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    VOICES_META.write_text(json.dumps(meta, indent=2))


def _get_available_voices() -> list[str]:
    """List all cloned voice names."""
    meta = _load_voices_meta()
    return sorted(meta.keys())


def _get_voice_profiles() -> dict[str, VoiceProfile]:
    """Load all voice profiles for batch processing."""
    meta = _load_voices_meta()
    profiles = {}
    for name, info in meta.items():
        audio_path = info.get("audio_path", "")
        if os.path.exists(audio_path):
            profiles[name.upper()] = VoiceProfile(
                name=name,
                reference_audio_path=audio_path,
                reference_text=info.get("reference_text", ""),
            )
    return profiles


def build_voiceforge_app(
    inference_engine,
    sample_rate: int,
    theme: str = "light",
) -> gr.Blocks:
    """Build the VoiceForge Gradio app."""

    def tts_inference(
        text: str,
        reference_audio_path: str,
        reference_text: str,
    ) -> np.ndarray:
        """Single TTS call wrapping the Fish-Speech inference engine."""
        with open(reference_audio_path, "rb") as f:
            audio_bytes = f.read()

        req = ServeTTSRequest(
            text=text,
            references=[ServeReferenceAudio(audio=audio_bytes, text=reference_text)],
            reference_id=None,
            max_new_tokens=1024,
            chunk_length=300,
            top_p=0.8,
            repetition_penalty=1.1,
            temperature=0.8,
            use_memory_cache="on",
        )

        for result in inference_engine.inference(req):
            if result.code == "final":
                _sr, audio_array = result.audio  # (sample_rate, np.ndarray)
                return audio_array
            elif result.code == "error":
                raise RuntimeError(f"TTS error: {result.error}")

        raise RuntimeError("No audio generated")

    # ── Voice Setup Tab ──────────────────────────────────────────────

    def clone_voice(name, audio_file, ref_text):
        """Clone a new voice from reference audio."""
        if not name or not name.strip():
            return "Error: Please enter a voice name.", _format_voices_list()
        if not audio_file:
            return "Error: Please upload reference audio.", _format_voices_list()
        if not ref_text or not ref_text.strip():
            return "Error: Please enter the reference text (what's being said in the audio).", _format_voices_list()

        name = name.strip().upper()

        # Save reference audio to persistent location
        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        voice_dir = VOICES_DIR / name.lower()
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Copy audio file
        dest_audio = voice_dir / "reference.wav"
        shutil.copy2(audio_file, str(dest_audio))

        # Save reference text
        (voice_dir / "reference.txt").write_text(ref_text.strip())

        # Update metadata
        meta = _load_voices_meta()
        meta[name] = {
            "audio_path": str(dest_audio),
            "reference_text": ref_text.strip(),
        }
        _save_voices_meta(meta)

        logger.info(f"Voice cloned: {name}")
        return f"Voice '{name}' cloned successfully!", _format_voices_list()

    def delete_voice(name):
        """Delete a cloned voice."""
        if not name:
            return "Select a voice to delete.", _format_voices_list()

        meta = _load_voices_meta()
        if name in meta:
            # Remove files
            voice_dir = VOICES_DIR / name.lower()
            if voice_dir.exists():
                shutil.rmtree(voice_dir)
            del meta[name]
            _save_voices_meta(meta)
            return f"Voice '{name}' deleted.", _format_voices_list()

        return f"Voice '{name}' not found.", _format_voices_list()

    def _format_voices_list():
        meta = _load_voices_meta()
        if not meta:
            return "No voices cloned yet. Upload reference audio to get started."
        lines = ["**Cloned Voices:**\n"]
        for name, info in sorted(meta.items()):
            ref_text = info.get("reference_text", "")[:60]
            lines.append(f"- **{name}** — ref: \"{ref_text}...\"")
        return "\n".join(lines)

    def preview_voice(name, preview_text):
        """Generate a short preview of a cloned voice."""
        if not name:
            return None, "Select a voice to preview."
        if not preview_text or not preview_text.strip():
            return None, "Enter some text to preview."

        profiles = _get_voice_profiles()
        key = name.upper()
        if key not in profiles:
            return None, f"Voice '{name}' not found."

        profile = profiles[key]
        try:
            audio = tts_inference(
                text=preview_text.strip(),
                reference_audio_path=profile.reference_audio_path,
                reference_text=profile.reference_text,
            )
            return (sample_rate, audio), None
        except Exception as e:
            return None, f"Preview failed: {e}"

    # ── Generate Tab ─────────────────────────────────────────────────

    def parse_and_preview(text_input, file_input):
        """Parse scripts and show summary."""
        text = ""
        if file_input is not None:
            # Read uploaded file
            if isinstance(file_input, str):
                text = Path(file_input).read_text(encoding="utf-8", errors="replace")
            else:
                text = file_input
        elif text_input and text_input.strip():
            text = text_input

        if not text.strip():
            return "No script provided. Paste text or upload a .txt file."

        scripts = parse_scripts(text)
        summary = scripts_summary(scripts)

        # Validate against available voices
        profiles = _get_voice_profiles()
        errors = validate_scripts(scripts, set(profiles.keys()))
        if errors:
            summary += "\n\n**Warnings:**\n" + "\n".join(f"- {e}" for e in errors)

        return summary

    def generate_batch(text_input, file_input, progress=gr.Progress()):
        """Generate all audio files from scripts."""
        text = ""
        if file_input is not None:
            if isinstance(file_input, str):
                text = Path(file_input).read_text(encoding="utf-8", errors="replace")
            else:
                text = file_input
        elif text_input and text_input.strip():
            text = text_input

        if not text.strip():
            return None, None, "No script provided."

        scripts = parse_scripts(text)
        if not scripts:
            return None, None, "Could not parse any video scripts from the input."

        profiles = _get_voice_profiles()
        errors = validate_scripts(scripts, set(profiles.keys()))
        if errors:
            # Filter to only blocking errors (missing voices)
            blocking = [e for e in errors if "No voice cloned" in e]
            if blocking:
                return None, None, "\n".join(blocking)

        def progress_cb(current, total, message):
            if total > 0:
                progress(current / total, desc=message)

        try:
            output_dir = tempfile.mkdtemp(prefix="voiceforge_")
            results = process_batch(
                scripts=scripts,
                voice_profiles=profiles,
                inference_fn=tts_inference,
                sample_rate=sample_rate,
                output_dir=output_dir,
                progress_callback=progress_cb,
            )

            if not results:
                return None, None, "No audio was generated."

            # Create zip for batch download
            zip_path = create_zip(results)

            # Return first audio for preview + zip for download
            first_file = results[0][1]
            audio_data, sr = sf.read(first_file)

            status_lines = [f"Generated {len(results)} audio file(s):\n"]
            for fname, fpath in results:
                info = sf.info(fpath)
                status_lines.append(f"- **{fname}** — {info.duration:.1f}s")

            return (sr, audio_data), zip_path, "\n".join(status_lines)

        except Exception as e:
            logger.error(f"Batch generation failed: {e}", exc_info=True)
            return None, None, f"Generation failed: {e}"

    # ── Build the UI ─────────────────────────────────────────────────

    with gr.Blocks(
        title="VoiceForge — Multi-Character Batch TTS",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# VoiceForge\n"
            "Multi-character batch TTS powered by Fish-Speech. "
            "Clone voices, write scripts with character markers, generate audio for multiple videos at once."
        )

        with gr.Tabs():
            # ── TAB 1: Voice Setup ───────────────────────────────
            with gr.Tab("Voice Setup"):
                gr.Markdown("### Clone a New Voice\nUpload 10-30 seconds of reference audio and the text being spoken.")

                with gr.Row():
                    with gr.Column():
                        voice_name = gr.Textbox(
                            label="Voice Name",
                            placeholder="e.g. PETER, STEWIE",
                            info="Use the same name in your scripts (PETER:, STEWIE:)",
                        )
                        ref_audio = gr.Audio(
                            label="Reference Audio (10-30s)",
                            type="filepath",
                        )
                        ref_text = gr.Textbox(
                            label="Reference Text",
                            placeholder="Type exactly what is being said in the reference audio",
                            lines=3,
                        )
                        with gr.Row():
                            clone_btn = gr.Button("Clone Voice", variant="primary")
                            delete_btn = gr.Button("Delete Voice", variant="stop")

                    with gr.Column():
                        voices_display = gr.Markdown(
                            value=_format_voices_list(),
                            label="Cloned Voices",
                        )
                        clone_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Preview a Voice")
                with gr.Row():
                    preview_voice_select = gr.Dropdown(
                        label="Select Voice",
                        choices=_get_available_voices(),
                        interactive=True,
                    )
                    preview_text = gr.Textbox(
                        label="Preview Text",
                        value="Hey, this is a quick test of the voice clone.",
                        lines=2,
                    )
                preview_btn = gr.Button("Preview")
                preview_audio = gr.Audio(label="Preview", type="numpy")
                preview_error = gr.Textbox(label="", visible=False)

                # Wire up voice setup events
                clone_btn.click(
                    clone_voice,
                    [voice_name, ref_audio, ref_text],
                    [clone_status, voices_display],
                ).then(
                    lambda: gr.update(choices=_get_available_voices()),
                    None,
                    [preview_voice_select],
                )
                delete_btn.click(
                    delete_voice,
                    [voice_name],
                    [clone_status, voices_display],
                ).then(
                    lambda: gr.update(choices=_get_available_voices()),
                    None,
                    [preview_voice_select],
                )
                preview_btn.click(
                    preview_voice,
                    [preview_voice_select, preview_text],
                    [preview_audio, preview_error],
                )

            # ── TAB 2: Generate ──────────────────────────────────
            with gr.Tab("Generate"):
                gr.Markdown(
                    "### Batch Script Processing\n"
                    "Paste scripts or upload a .txt file. Use `=== VIDEO N ===` to separate videos. "
                    "Prefix each line with the character name (e.g. `PETER:`, `STEWIE:`)."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        script_text = gr.Textbox(
                            label="Scripts (paste here)",
                            placeholder=EXAMPLE_SCRIPT,
                            lines=20,
                        )
                        script_file = gr.File(
                            label="Or upload a .txt file",
                            file_types=[".txt"],
                        )
                        with gr.Row():
                            parse_btn = gr.Button("Parse & Preview")
                            generate_btn = gr.Button("Generate All Audio", variant="primary")

                    with gr.Column(scale=1):
                        parse_output = gr.Markdown(label="Script Summary")
                        gen_status = gr.Markdown(label="Generation Status")
                        gen_preview_audio = gr.Audio(
                            label="Preview (first video)",
                            type="numpy",
                        )
                        gen_download = gr.File(label="Download All (ZIP)")

                # Wire up generate events
                parse_btn.click(
                    parse_and_preview,
                    [script_text, script_file],
                    [parse_output],
                )
                generate_btn.click(
                    generate_batch,
                    [script_text, script_file],
                    [gen_preview_audio, gen_download, gen_status],
                )

            # ── TAB 3: Help ──────────────────────────────────────
            with gr.Tab("Help"):
                gr.Markdown(f"""
### Script Format

```
=== VIDEO 1 ===
PETER: Hey Stewie, did you know that teachers can see when you paste from ChatGPT?
STEWIE: Wait, what? Are you serious?
PETER: Yeah bro. But there's this thing called EvadeGPT...

=== VIDEO 2 ===
PETER: Another video starts here.
STEWIE: Cool, got it.
```

### Rules
- **`=== VIDEO N ===`** separates different videos (each gets its own audio file)
- **`CHARACTER:`** prefix tells the system which voice to use
- Character names must match your cloned voices (case-insensitive)
- You can have any number of characters, not just Peter and Stewie
- Lines without a character prefix are ignored
- The system adds natural pauses between speaker switches

### Voice Cloning Tips
- Use **10-30 seconds** of clean reference audio
- Avoid background music or noise in reference clips
- Provide the **exact text** being spoken in the reference audio
- Character voices, cartoon voices, etc. all work — Fish-Speech handles them well

### Output
- One `.wav` file per video
- Files are named `video_01.wav`, `video_02.wav`, etc.
- Download individually or as a ZIP
""")

    return app
