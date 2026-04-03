from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from mutagen import File as MutagenFile

from cuemate_analysis.models import ImportedTrack


SUPPORTED_AUDIO_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".wav",
}


def discover_audio_files(paths: Iterable[str | Path]) -> list[Path]:
    discovered: dict[str, Path] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_dir():
            candidates = [candidate for candidate in path.rglob("*") if candidate.is_file()]
        else:
            candidates = [path]

        for candidate in candidates:
            if candidate.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
                continue
            resolved = candidate.resolve()
            discovered[str(resolved).lower()] = resolved

    return sorted(discovered.values(), key=lambda candidate: str(candidate).lower())


def make_track_id(path: Path) -> str:
    normalized = str(path.resolve()).lower().encode("utf-8")
    return f"trk_{hashlib.sha1(normalized).hexdigest()[:16]}"


def make_playlist_id(name: str) -> str:
    normalized = name.strip().lower().encode("utf-8")
    return f"plt_{hashlib.sha1(normalized).hexdigest()[:16]}"


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _first_value(source: object) -> str | None:
    if source is None:
        return None
    if isinstance(source, list):
        for item in source:
            if item is None:
                continue
            value = str(item).strip()
            if value:
                return value
        return None
    value = str(source).strip()
    return value or None


def _first_tag(tags: object, names: Iterable[str]) -> str | None:
    if tags is None:
        return None

    if hasattr(tags, "get"):
        for name in names:
            try:
                value = tags.get(name)
            except Exception:
                value = None
            parsed = _first_value(value)
            if parsed:
                return parsed

    if isinstance(tags, dict):
        lowered = {str(key).lower(): value for key, value in tags.items()}
        for name in names:
            parsed = _first_value(lowered.get(name.lower()))
            if parsed:
                return parsed

    return None


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def read_track_metadata(path: Path) -> ImportedTrack:
    resolved = path.resolve()
    audio_easy = MutagenFile(resolved, easy=True)
    audio_full = audio_easy if audio_easy is not None else MutagenFile(resolved)

    duration_seconds: float | None = None
    if audio_full is not None and getattr(audio_full, "info", None) is not None:
        duration = getattr(audio_full.info, "length", None)
        if duration is not None:
            duration_seconds = float(duration)

    tags = getattr(audio_easy, "tags", None) if audio_easy is not None else None
    if tags is None and audio_full is not None:
        tags = getattr(audio_full, "tags", None)

    title = _first_tag(tags, ["title"]) or resolved.stem
    artist = _first_tag(tags, ["artist", "albumartist"])
    genre = _first_tag(tags, ["genre"])
    bpm_tag = _parse_float(_first_tag(tags, ["bpm", "tbpm"]))
    key_tag = _first_tag(tags, ["initialkey", "key", "tkey"])

    return ImportedTrack(
        id=make_track_id(resolved),
        file_path=resolved,
        file_hash=compute_file_hash(resolved),
        title=title,
        artist=artist,
        genre=genre,
        duration_seconds=duration_seconds,
        bpm_tag=bpm_tag,
        key_tag=key_tag,
    )
