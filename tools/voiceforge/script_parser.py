"""
Script parser for VoiceForge.

Parses multi-video, multi-character scripts in this format:

    === VIDEO 1 ===
    PETER: Hey Stewie, did you know teachers can see when you paste from ChatGPT?
    STEWIE: Wait, what? Really?
    PETER: Yeah bro, they've got AI detectors now.

    === VIDEO 2 ===
    PETER: Stewie you look stressed mate.
    ...

Returns structured data for batch TTS processing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Line:
    """A single spoken line."""
    character: str  # e.g. "PETER", "STEWIE"
    text: str       # The dialogue text

    def __repr__(self):
        return f"{self.character}: {self.text[:50]}..."


@dataclass
class VideoScript:
    """All lines for one video."""
    video_number: int
    title: str  # e.g. "VIDEO 1" or custom title
    lines: list[Line] = field(default_factory=list)

    @property
    def characters(self) -> set[str]:
        return {line.character for line in self.lines}


# Regex patterns
VIDEO_DELIMITER = re.compile(
    r"^===\s*(?:VIDEO\s*)?(\d+)?\s*(?:[-:]\s*(.+?))?\s*===$",
    re.IGNORECASE | re.MULTILINE,
)
CHARACTER_LINE = re.compile(
    r"^([A-Z][A-Z0-9_ ]{0,30}):\s*(.+)$",
    re.MULTILINE,
)


def parse_scripts(text: str) -> list[VideoScript]:
    """
    Parse a multi-video script file into structured VideoScript objects.

    Supports two modes:
    1. Delimited: Multiple videos separated by === VIDEO N === markers
    2. Single: No delimiters = one video with all lines
    """
    text = text.strip()
    if not text:
        return []

    # Check if we have video delimiters
    delimiters = list(VIDEO_DELIMITER.finditer(text))

    if not delimiters:
        # Single video mode — treat entire text as one video
        lines = _parse_lines(text)
        if lines:
            return [VideoScript(video_number=1, title="VIDEO 1", lines=lines)]
        return []

    scripts = []
    for i, match in enumerate(delimiters):
        # Extract video number and optional title
        num = int(match.group(1)) if match.group(1) else i + 1
        title = match.group(2) or f"VIDEO {num}"

        # Get text between this delimiter and the next (or end of file)
        start = match.end()
        end = delimiters[i + 1].start() if i + 1 < len(delimiters) else len(text)
        section = text[start:end].strip()

        lines = _parse_lines(section)
        if lines:
            scripts.append(VideoScript(video_number=num, title=title, lines=lines))

    return scripts


def _parse_lines(section: str) -> list[Line]:
    """Parse character lines from a section of text."""
    lines = []
    for match in CHARACTER_LINE.finditer(section):
        character = match.group(1).strip().upper()
        dialogue = match.group(2).strip()
        if dialogue:
            lines.append(Line(character=character, text=dialogue))
    return lines


def validate_scripts(
    scripts: list[VideoScript],
    available_voices: set[str],
) -> list[str]:
    """
    Validate scripts against available cloned voices.
    Returns list of error messages (empty = all good).
    """
    errors = []
    all_characters = set()
    for script in scripts:
        all_characters.update(script.characters)

    # Check that all characters have voices
    missing = all_characters - {v.upper() for v in available_voices}
    if missing:
        errors.append(
            f"No voice cloned for: {', '.join(sorted(missing))}. "
            f"Available voices: {', '.join(sorted(available_voices))}"
        )

    # Check for empty scripts
    for script in scripts:
        if not script.lines:
            errors.append(f"{script.title}: No dialogue lines found")

    return errors


def scripts_summary(scripts: list[VideoScript]) -> str:
    """Human-readable summary of parsed scripts."""
    if not scripts:
        return "No scripts parsed."

    parts = [f"Parsed {len(scripts)} video(s):\n"]
    for s in scripts:
        chars = ", ".join(sorted(s.characters))
        parts.append(f"  {s.title}: {len(s.lines)} lines ({chars})")
    return "\n".join(parts)
