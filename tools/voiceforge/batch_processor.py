"""
Batch TTS processor for VoiceForge.

Takes parsed scripts + voice references, generates TTS for each line,
adds natural pauses between speakers, and stitches into one audio file per video.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import soundfile as sf
from loguru import logger

from tools.voiceforge.script_parser import Line, VideoScript


@dataclass
class VoiceProfile:
    """A cloned voice profile with reference audio + text."""
    name: str
    reference_audio_path: str
    reference_text: str


@dataclass
class GeneratedAudio:
    """Result of generating audio for one video."""
    video_number: int
    title: str
    audio_data: np.ndarray
    sample_rate: int
    duration_seconds: float
    line_count: int


# Pause durations (in seconds)
SAME_SPEAKER_PAUSE = 0.3    # Brief pause when same speaker continues
SPEAKER_SWITCH_PAUSE = 0.6  # Natural pause when switching speakers
PARAGRAPH_PAUSE = 1.0       # Longer pause for emphasis markers


def generate_silence(duration_seconds: float, sample_rate: int) -> np.ndarray:
    """Generate silence of specified duration."""
    return np.zeros(int(duration_seconds * sample_rate), dtype=np.float32)


def process_single_video(
    script: VideoScript,
    voice_profiles: dict[str, VoiceProfile],
    inference_fn,
    sample_rate: int,
    progress_callback=None,
) -> GeneratedAudio:
    """
    Process one video script: generate TTS for each line, stitch with pauses.

    Args:
        script: Parsed video script with character lines
        voice_profiles: Map of CHARACTER_NAME -> VoiceProfile
        inference_fn: Function(text, reference_audio_bytes, reference_text) -> np.ndarray
        sample_rate: Audio sample rate from the model
        progress_callback: Optional fn(current_line, total_lines, message) for UI updates
    """
    audio_segments = []
    total_lines = len(script.lines)

    for i, line in enumerate(script.lines):
        if progress_callback:
            progress_callback(
                i, total_lines,
                f"{script.title}: Generating line {i+1}/{total_lines} — {line.character}"
            )

        # Get voice profile for this character
        profile_key = line.character.upper()
        if profile_key not in voice_profiles:
            logger.warning(f"No voice for {profile_key}, skipping line: {line.text[:50]}")
            continue

        profile = voice_profiles[profile_key]

        # Generate TTS for this line
        try:
            audio_chunk = inference_fn(
                text=line.text,
                reference_audio_path=profile.reference_audio_path,
                reference_text=profile.reference_text,
            )
        except Exception as e:
            logger.error(f"TTS failed for line {i+1} ({line.character}): {e}")
            # Generate 1 second of silence as placeholder
            audio_chunk = generate_silence(1.0, sample_rate)

        audio_segments.append(audio_chunk)

        # Add pause between lines
        if i < total_lines - 1:
            next_line = script.lines[i + 1]
            if next_line.character == line.character:
                pause = generate_silence(SAME_SPEAKER_PAUSE, sample_rate)
            else:
                pause = generate_silence(SPEAKER_SWITCH_PAUSE, sample_rate)
            audio_segments.append(pause)

    if not audio_segments:
        # Return silence if nothing was generated
        return GeneratedAudio(
            video_number=script.video_number,
            title=script.title,
            audio_data=generate_silence(1.0, sample_rate),
            sample_rate=sample_rate,
            duration_seconds=1.0,
            line_count=0,
        )

    # Stitch all segments
    full_audio = np.concatenate(audio_segments)
    duration = len(full_audio) / sample_rate

    logger.info(
        f"{script.title}: Generated {duration:.1f}s audio from {total_lines} lines"
    )

    return GeneratedAudio(
        video_number=script.video_number,
        title=script.title,
        audio_data=full_audio,
        sample_rate=sample_rate,
        duration_seconds=duration,
        line_count=total_lines,
    )


def process_batch(
    scripts: list[VideoScript],
    voice_profiles: dict[str, VoiceProfile],
    inference_fn,
    sample_rate: int,
    output_dir: str | None = None,
    progress_callback=None,
) -> list[tuple[str, str]]:
    """
    Process all video scripts and save audio files.

    Returns list of (filename, filepath) tuples.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="voiceforge_")

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for idx, script in enumerate(scripts):
        if progress_callback:
            progress_callback(
                0, len(script.lines),
                f"Processing video {idx+1}/{len(scripts)}: {script.title}"
            )

        generated = process_single_video(
            script=script,
            voice_profiles=voice_profiles,
            inference_fn=inference_fn,
            sample_rate=sample_rate,
            progress_callback=progress_callback,
        )

        # Save with clean naming
        filename = f"video_{generated.video_number:02d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, generated.audio_data, generated.sample_rate)

        results.append((filename, filepath))
        logger.info(f"Saved: {filename} ({generated.duration_seconds:.1f}s)")

    return results


def create_zip(file_list: list[tuple[str, str]]) -> str:
    """Create a zip file from generated audio files. Returns zip path."""
    zip_path = tempfile.mktemp(suffix=".zip", prefix="voiceforge_batch_")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, filepath in file_list:
            zf.write(filepath, filename)
    return zip_path
