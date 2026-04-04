from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


SUPPORTED_DJ_LIBRARY_SOURCES = {"rekordbox", "traktor", "serato"}
SUPPORTED_AUDIO_EXTENSIONS = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


@dataclass(frozen=True)
class DJPlaylistTrack:
    file_path: Path
    title: str | None
    artist: str | None
    genre: str | None
    bpm_imported: float | None
    key_imported: str | None
    import_source: str


def normalize_dj_source(source: str) -> str:
    clean = source.strip().lower()
    if clean not in SUPPORTED_DJ_LIBRARY_SOURCES:
        allowed = ", ".join(sorted(SUPPORTED_DJ_LIBRARY_SOURCES))
        raise ValueError(f"Unsupported DJ library source '{source}'. Expected one of: {allowed}")
    return clean


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _normalize_playlist_name(value: str) -> str:
    return value.strip().casefold()


def _resolve_playlist_choice(available: dict[str, object], requested: str) -> object:
    normalized = _normalize_playlist_name(requested)
    exact = {name: value for name, value in available.items() if _normalize_playlist_name(name) == normalized}
    if exact:
        return next(iter(exact.values()))

    leaf_matches = {
        name: value
        for name, value in available.items()
        if _normalize_playlist_name(name.split(" / ")[-1]) == normalized
    }
    if len(leaf_matches) == 1:
        return next(iter(leaf_matches.values()))
    if len(leaf_matches) > 1:
        joined = ", ".join(sorted(leaf_matches))
        raise ValueError(f"Playlist name '{requested}' is ambiguous. Use one of: {joined}")
    raise ValueError(f"Playlist '{requested}' was not found.")


def _decode_file_url(raw_value: str) -> Path | None:
    if not raw_value:
        return None
    parsed = urlparse(raw_value)
    if parsed.scheme != "file":
        return None
    candidate = unquote(parsed.path or "")
    if re.match(r"^/[A-Za-z]:/", candidate):
        candidate = candidate[1:]
    return Path(candidate).resolve()


def _clean_rekordbox_path(raw_value: str | None) -> Path | None:
    if not raw_value:
        return None
    decoded = _decode_file_url(raw_value)
    if decoded is not None:
        return decoded
    return Path(unquote(raw_value)).expanduser().resolve()


def _walk_rekordbox_nodes(node: ET.Element, parents: list[str]) -> Iterable[tuple[str, ET.Element]]:
    name = (node.attrib.get("Name") or "").strip()
    node_type = (node.attrib.get("Type") or "").strip()
    current = [*parents]
    if name:
        current.append(name)
    if node_type == "1":
        yield (" / ".join(current), node)
    for child in node.findall("NODE"):
        yield from _walk_rekordbox_nodes(child, current)


def _load_rekordbox_tree(library_path: Path) -> ET.Element:
    return ET.parse(library_path).getroot()


def list_rekordbox_playlists(library_path: Path) -> list[str]:
    root = _load_rekordbox_tree(library_path)
    playlists_root = root.find("PLAYLISTS")
    if playlists_root is None:
        return []
    names: list[str] = []
    for node in playlists_root.findall("NODE"):
        names.extend(name for name, _ in _walk_rekordbox_nodes(node, []))
    return sorted(dict.fromkeys(names))


def load_rekordbox_playlist(library_path: Path, playlist_name: str) -> list[DJPlaylistTrack]:
    root = _load_rekordbox_tree(library_path)
    collection = root.find("COLLECTION")
    playlists_root = root.find("PLAYLISTS")
    if collection is None or playlists_root is None:
        raise ValueError("Rekordbox XML is missing COLLECTION or PLAYLISTS.")

    track_index: dict[str, DJPlaylistTrack] = {}
    for track in collection.findall("TRACK"):
        track_id = (track.attrib.get("TrackID") or track.attrib.get("ID") or "").strip()
        file_path = _clean_rekordbox_path(track.attrib.get("Location"))
        if not track_id or file_path is None:
            continue
        track_index[track_id] = DJPlaylistTrack(
            file_path=file_path,
            title=(track.attrib.get("Name") or "").strip() or None,
            artist=(track.attrib.get("Artist") or "").strip() or None,
            genre=(track.attrib.get("Genre") or "").strip() or None,
            bpm_imported=_parse_float(track.attrib.get("AverageBpm")),
            key_imported=(track.attrib.get("Tonality") or "").strip() or None,
            import_source="rekordbox_xml",
        )

    playlist_index: dict[str, ET.Element] = {}
    for node in playlists_root.findall("NODE"):
        for resolved_name, playlist_node in _walk_rekordbox_nodes(node, []):
            playlist_index[resolved_name] = playlist_node

    playlist_node = _resolve_playlist_choice(playlist_index, playlist_name)
    imported_tracks: list[DJPlaylistTrack] = []
    for track_ref in playlist_node.findall("TRACK"):
        key = (track_ref.attrib.get("Key") or track_ref.attrib.get("TrackID") or "").strip()
        track = track_index.get(key)
        if track is not None:
            imported_tracks.append(track)
    return imported_tracks


def _clean_traktor_dir(raw_value: str | None) -> str:
    if not raw_value:
        return ""
    clean = raw_value.replace("/:","\\").replace("/", "\\")
    while "\\\\" in clean:
        clean = clean.replace("\\\\", "\\")
    return clean


def _clean_traktor_path(raw_value: str | None, *, volume: str | None = None, directory: str | None = None, file_name: str | None = None) -> Path | None:
    if raw_value:
        decoded = _decode_file_url(raw_value)
        if decoded is not None:
            return decoded
        if re.match(r"^[A-Za-z]:[\\\\/]", raw_value):
            return Path(raw_value).expanduser().resolve()
    if volume and file_name is not None:
        directory_part = _clean_traktor_dir(directory)
        return Path(f"{volume}{directory_part}{file_name}").resolve()
    return None


def _traktor_key_from_entry(entry: ET.Element) -> str | None:
    for candidate_path in (
        "./INFO",
        "./MUSICAL_KEY",
    ):
        node = entry.find(candidate_path)
        if node is None:
            continue
        for attribute in ("KEY", "VALUE", "TONAL_KEY"):
            value = (node.attrib.get(attribute) or "").strip()
            if value and not value.isdigit():
                return value
    return None


def _load_traktor_tree(library_path: Path) -> ET.Element:
    return ET.parse(library_path).getroot()


def _walk_traktor_nodes(node: ET.Element, parents: list[str]) -> Iterable[tuple[str, ET.Element]]:
    name = (node.attrib.get("NAME") or "").strip()
    current = [*parents]
    if name:
        current.append(name)
    playlist = node.find("PLAYLIST")
    if playlist is not None:
        yield (" / ".join(current), node)
    for child in node.findall("./SUBNODES/NODE"):
        yield from _walk_traktor_nodes(child, current)


def list_traktor_playlists(library_path: Path) -> list[str]:
    root = _load_traktor_tree(library_path)
    playlist_root = root.find("PLAYLISTS")
    if playlist_root is None:
        return []
    names: list[str] = []
    for node in playlist_root.findall("NODE"):
        names.extend(name for name, _ in _walk_traktor_nodes(node, []))
    return sorted(dict.fromkeys(names))


def load_traktor_playlist(library_path: Path, playlist_name: str) -> list[DJPlaylistTrack]:
    root = _load_traktor_tree(library_path)
    collection = root.find("COLLECTION")
    playlist_root = root.find("PLAYLISTS")
    if collection is None or playlist_root is None:
        raise ValueError("Traktor NML is missing COLLECTION or PLAYLISTS.")

    track_index: dict[str, DJPlaylistTrack] = {}
    for entry in collection.findall("ENTRY"):
        location = entry.find("LOCATION")
        if location is None:
            continue
        file_path = _clean_traktor_path(
            None,
            volume=(location.attrib.get("VOLUME") or "").strip(),
            directory=(location.attrib.get("DIR") or "").strip(),
            file_name=(location.attrib.get("FILE") or "").strip(),
        )
        if file_path is None:
            continue
        track_index[file_path.as_posix().casefold()] = DJPlaylistTrack(
            file_path=file_path,
            title=(entry.attrib.get("TITLE") or "").strip() or None,
            artist=(entry.attrib.get("ARTIST") or "").strip() or None,
            genre=(entry.attrib.get("GENRE") or "").strip() or None,
            bpm_imported=_parse_float((entry.find("TEMPO").attrib.get("BPM") if entry.find("TEMPO") is not None else None)),
            key_imported=_traktor_key_from_entry(entry),
            import_source="traktor_nml",
        )

    playlist_index: dict[str, ET.Element] = {}
    for node in playlist_root.findall("NODE"):
        for resolved_name, playlist_node in _walk_traktor_nodes(node, []):
            playlist_index[resolved_name] = playlist_node

    playlist_node = _resolve_playlist_choice(playlist_index, playlist_name)
    playlist = playlist_node.find("PLAYLIST")
    if playlist is None:
        return []

    imported_tracks: list[DJPlaylistTrack] = []
    for entry in playlist.findall("ENTRY"):
        primary_key = entry.find("PRIMARYKEY")
        resolved_path: Path | None = None
        if primary_key is not None:
            key_value = (primary_key.attrib.get("KEY") or "").strip()
            resolved_path = _clean_traktor_path(key_value)
        if resolved_path is None:
            location = entry.find("LOCATION")
            if location is not None:
                resolved_path = _clean_traktor_path(
                    None,
                    volume=(location.attrib.get("VOLUME") or "").strip(),
                    directory=(location.attrib.get("DIR") or "").strip(),
                    file_name=(location.attrib.get("FILE") or "").strip(),
                )
        if resolved_path is None:
            continue
        track = track_index.get(resolved_path.as_posix().casefold())
        if track is not None:
            imported_tracks.append(track)
    return imported_tracks


SERATO_CRATE_PATH_RE = re.compile(
    r"([A-Za-z]:[\\/][^\x00\r\n]+?\.(?:aac|aif|aiff|flac|m4a|mp3|ogg|wav))",
    re.IGNORECASE,
)


def _list_serato_crate_files(library_path: Path) -> list[Path]:
    if library_path.is_file():
        return [library_path.resolve()]
    return sorted(
        [path.resolve() for path in library_path.rglob("*.crate") if path.is_file()],
        key=lambda path: path.as_posix().casefold(),
    )


def list_serato_playlists(library_path: Path) -> list[str]:
    crate_files = _list_serato_crate_files(library_path)
    base = library_path.resolve() if library_path.is_dir() else library_path.resolve().parent
    names: list[str] = []
    for crate_file in crate_files:
        try:
            relative = crate_file.relative_to(base)
            names.append(relative.with_suffix("").as_posix())
        except ValueError:
            names.append(crate_file.stem)
    return sorted(dict.fromkeys(names))


def load_serato_playlist(library_path: Path, playlist_name: str) -> list[DJPlaylistTrack]:
    crate_files = _list_serato_crate_files(library_path)
    if not crate_files:
        return []
    base = library_path.resolve() if library_path.is_dir() else library_path.resolve().parent
    crate_index: dict[str, Path] = {}
    for crate_file in crate_files:
        try:
            name = crate_file.relative_to(base).with_suffix("").as_posix()
        except ValueError:
            name = crate_file.stem
        crate_index[name] = crate_file

    crate_file = _resolve_playlist_choice(crate_index, playlist_name)
    decoded = crate_file.read_bytes().decode("utf-8", errors="ignore")
    discovered_paths: list[Path] = []
    for match in SERATO_CRATE_PATH_RE.findall(decoded):
        resolved = Path(match).expanduser().resolve()
        if resolved.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
            discovered_paths.append(resolved)

    unique_paths: dict[str, Path] = {}
    for path in discovered_paths:
        unique_paths[path.as_posix().casefold()] = path

    return [
        DJPlaylistTrack(
            file_path=path,
            title=path.stem,
            artist=None,
            genre=None,
            bpm_imported=None,
            key_imported=None,
            import_source="serato_crate",
        )
        for path in unique_paths.values()
    ]


def list_dj_playlists(source: str, library_path: Path) -> list[str]:
    normalized = normalize_dj_source(source)
    resolved_library_path = library_path.expanduser().resolve()
    if normalized == "rekordbox":
        return list_rekordbox_playlists(resolved_library_path)
    if normalized == "traktor":
        return list_traktor_playlists(resolved_library_path)
    return list_serato_playlists(resolved_library_path)


def load_dj_playlist(source: str, library_path: Path, playlist_name: str) -> list[DJPlaylistTrack]:
    normalized = normalize_dj_source(source)
    resolved_library_path = library_path.expanduser().resolve()
    if normalized == "rekordbox":
        return load_rekordbox_playlist(resolved_library_path, playlist_name)
    if normalized == "traktor":
        return load_traktor_playlist(resolved_library_path, playlist_name)
    return load_serato_playlist(resolved_library_path, playlist_name)
