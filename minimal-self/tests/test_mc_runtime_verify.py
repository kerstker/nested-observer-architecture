"""The deployment probe may read production memory, so verify its no-write behavior."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from src.levels.l2_memory.embedding import Embedder
from src.levels.l8_agency.mc_game_memory import GameMemory

spec = importlib.util.spec_from_file_location("verify_mc_runtime", Path(__file__).resolve().parents[2] / "scripts" / "verify_mc_runtime.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_vector_probe_reads_without_mutation_or_default_text_export(tmp_path):
    path = tmp_path / "game.sqlite"
    emb = Embedder()
    memory = GameMemory(str(path), embedder=emb)
    memory.remember("家在白桦树林，箱子里有铁锭", kind="place")
    memory.close()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = probe.read_vectors(path, emb, ["家在白桦树林，箱子里有铁锭"])
    assert result["matching_vectors"] == 1
    assert result["queries"][0]["hits"][0]["score"] > 0.95
    assert "game_text" not in result["queries"][0]["hits"][0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert list(tmp_path.iterdir()) == [path]
    assert not probe.read_vectors(tmp_path / "missing.sqlite", emb, ["家"])["available"]
    assert not (tmp_path / "missing.sqlite").exists()


def test_console_probe_rejects_remote_and_filters_private_payload(tmp_path, monkeypatch):
    for url in ("https://example.org", "http://user:secret@localhost", "http://localhost?token=secret"):
        with pytest.raises(ValueError):
            probe.live_status(url, tmp_path / "missing")
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"inner": "private", "game": {"connected": True,
                "journal": ["private"], "username": "private", "quests": {"quests": 143},
                "game_memory": {"shared_embedder": True}}}).encode()
    class Opener:
        def open(self, request, timeout):
            assert request.method == "GET"
            return Response()
    monkeypatch.setattr(probe.urllib.request, "build_opener", lambda *args: Opener())
    result = probe.live_status("http://127.0.0.1:8080", tmp_path / "missing")
    assert result["quests"]["quests"] == 143
    assert "private" not in json.dumps(result)


def test_vector_probe_includes_live_wal_without_touching_source_files(tmp_path):
    path = tmp_path / "running.sqlite"
    emb = Embedder()
    memory = GameMemory(str(path), embedder=emb)
    try:
        memory.remember("刚刚在矿洞发现铁矿", kind="lesson")
        before = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                  for p in tmp_path.iterdir()}
        assert "running.sqlite-wal" in before
        result = probe.read_vectors(path, emb, ["刚刚在矿洞发现铁矿"])
        assert result["records"] == 1 and result["queries"][0]["hits"]
        after = {p.name: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in tmp_path.iterdir()}
        assert before == after
    finally:
        memory.close()
