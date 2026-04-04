from __future__ import annotations

from pathlib import Path

from cuemate_analysis.dj_import import list_dj_playlists, load_dj_playlist


def test_rekordbox_playlist_import_extracts_metadata(tmp_path: Path) -> None:
    track_path = tmp_path / "music" / "track.mp3"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_bytes(b"fake-audio")

    library_path = tmp_path / "rekordbox.xml"
    library_path.write_text(
        f"""
<DJ_PLAYLISTS>
  <COLLECTION>
    <TRACK
      TrackID="1"
      Name="Test Track"
      Artist="Test Artist"
      Genre="House"
      AverageBpm="128.0"
      Tonality="8A"
      Location="{track_path.resolve().as_uri()}"
    />
  </COLLECTION>
  <PLAYLISTS>
    <NODE Name="Folder" Type="0">
      <NODE Name="Main" Type="1">
        <TRACK Key="1" />
      </NODE>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
""".strip(),
        encoding="utf-8",
    )

    playlists = list_dj_playlists("rekordbox", library_path)
    imported_tracks = load_dj_playlist("rekordbox", library_path, "Main")

    assert playlists == ["Folder / Main"]
    assert len(imported_tracks) == 1
    assert imported_tracks[0].file_path == track_path.resolve()
    assert imported_tracks[0].bpm_imported == 128.0
    assert imported_tracks[0].key_imported == "8A"
    assert imported_tracks[0].import_source == "rekordbox_xml"


def test_traktor_playlist_import_extracts_metadata(tmp_path: Path) -> None:
    track_path = tmp_path / "music" / "track.mp3"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_bytes(b"fake-audio")
    resolved_track = track_path.resolve()
    directory = "\\" + "\\".join(resolved_track.parts[1:-1]) + "\\"

    library_path = tmp_path / "collection.nml"
    library_path.write_text(
        f"""
<NML>
  <COLLECTION>
    <ENTRY TITLE="Test Track" ARTIST="Test Artist" GENRE="House">
      <LOCATION VOLUME="{resolved_track.drive}" DIR="{directory}" FILE="{resolved_track.name}" />
      <TEMPO BPM="130.0" />
      <MUSICAL_KEY VALUE="9A" />
    </ENTRY>
  </COLLECTION>
  <PLAYLISTS>
    <NODE NAME="Folder">
      <SUBNODES>
        <NODE NAME="Main">
          <PLAYLIST>
            <ENTRY>
              <PRIMARYKEY KEY="{resolved_track.as_posix()}" />
            </ENTRY>
          </PLAYLIST>
        </NODE>
      </SUBNODES>
    </NODE>
  </PLAYLISTS>
</NML>
""".strip(),
        encoding="utf-8",
    )

    playlists = list_dj_playlists("traktor", library_path)
    imported_tracks = load_dj_playlist("traktor", library_path, "Main")

    assert playlists == ["Folder / Main"]
    assert len(imported_tracks) == 1
    assert imported_tracks[0].file_path == resolved_track
    assert imported_tracks[0].bpm_imported == 130.0
    assert imported_tracks[0].key_imported == "9A"
    assert imported_tracks[0].import_source == "traktor_nml"


def test_serato_playlist_import_extracts_track_paths(tmp_path: Path) -> None:
    track_path = tmp_path / "music" / "track.mp3"
    track_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.write_bytes(b"fake-audio")

    crates_dir = tmp_path / "serato"
    crates_dir.mkdir()
    crate_path = crates_dir / "My Crate.crate"
    crate_path.write_bytes(b"header\x00" + str(track_path.resolve()).encode("utf-8") + b"\x00footer")

    playlists = list_dj_playlists("serato", crates_dir)
    imported_tracks = load_dj_playlist("serato", crates_dir, "My Crate")

    assert playlists == ["My Crate"]
    assert len(imported_tracks) == 1
    assert imported_tracks[0].file_path == track_path.resolve()
    assert imported_tracks[0].bpm_imported is None
    assert imported_tracks[0].key_imported is None
    assert imported_tracks[0].import_source == "serato_crate"
