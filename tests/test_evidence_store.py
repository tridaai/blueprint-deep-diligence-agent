from deep_diligence.backend import EvidenceBackend
from deep_diligence.evidence import evidence_text
from deep_diligence.objects import sha256_hex


def test_evidence_bytes_round_trip(service, case):
    ev = service.store.list_evidence(case["case_id"])[0]
    text = evidence_text(service.objects, ev)
    assert sha256_hex(text.encode()) == ev["sha256"]
    assert ev["title"] in text


def test_backend_file_round_trip(service, case):
    backend = EvidenceBackend(service.store, service.objects, case["case_id"])
    assert backend.write("/notes/scratch.md", "first draft").error is None
    assert backend.write("/notes/scratch.md", "again").error is not None  # exists
    read = backend.read("/notes/scratch.md")
    assert read.error is None
    assert "first draft" in read.file_data["content"]
    edited = backend.edit("/notes/scratch.md", "first", "second")
    assert edited.error is None and edited.occurrences == 1
    assert "second draft" in backend.read("/notes/scratch.md").file_data["content"]
    listing = backend.ls("/notes/")
    assert [e["path"] for e in listing.entries] == ["/notes/scratch.md"]
    assert backend.glob("*.md", "/notes").matches == ["/notes/scratch.md"]
    grep = backend.grep("second", "/notes/")
    assert grep.matches and grep.matches[0]["path"] == "/notes/scratch.md"
    assert backend.delete("/notes/scratch.md").error is None
    assert backend.read("/notes/scratch.md").error is not None


def test_backend_mounts_evidence_read_only(service, case):
    backend = EvidenceBackend(service.store, service.objects, case["case_id"])
    listing = backend.ls("/evidence/")
    paths = [e["path"] for e in listing.entries]
    assert "/evidence/EV-0001.md" in paths
    read = backend.read("/evidence/EV-0001.md")
    assert "Key facts" in read.file_data["content"]
    assert backend.write("/evidence/EV-0001.md", "tampered").error is not None
    assert backend.edit("/evidence/EV-0001.md", "a", "b").error is not None
    assert backend.delete("/evidence/EV-0001.md").error is not None
