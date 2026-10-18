"""The Deep Agents virtual filesystem, backed by the evidence store.

`create_deep_agent` talks to a BackendProtocol. This backend serves one
case: its files are objects in the object store with an index row in
the case store, and the case's evidence is mounted read-only under
/evidence/<EV-ID>.md so the agent reads source documents through the
same filesystem tools it uses for its own notes. Writes under
/evidence/ are refused. The skills the case's approved plan pins
(skills.py) mount the same way under /skills/<name>.md — the method
text verbatim, read-only: a workstream reads its method from the
filesystem, and no agent write can reach it.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import datetime, timezone

from deepagents.backends.protocol import (
    DeleteResult,
    FileData,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
    EditResult,
)
from deepagents.backends.utils import create_file_data, slice_read_response

from .evidence import evidence_text
from .objects import ObjectStore, sha256_hex
from .skills import SKILLS_PREFIX
from .store import Store

EVIDENCE_PREFIX = "/evidence/"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvidenceBackend:
    """One case's filesystem: case files writable, evidence and
    pinned skills read-only."""

    def __init__(
        self,
        store: Store,
        objects: ObjectStore,
        case_id: str,
        skill_files: dict[str, str] | None = None,
    ):
        self.store = store
        self.objects = objects
        self.case_id = case_id
        # The pinned skills mounted at /skills/<name>.md for this
        # case (path -> method text), resolved by the service from
        # the approved plan's pins. Not case files: they live in
        # the library, are shared across cases, and are read-only
        # here exactly as the evidence mount is.
        self.skill_files = dict(skill_files or {})

    # -- key helpers ------------------------------------------------------
    def _object_key(self, path: str) -> str:
        return f"cases/{self.case_id}/files{path}"

    def _is_evidence(self, path: str) -> bool:
        return path.startswith(EVIDENCE_PREFIX)

    def _is_skill(self, path: str) -> bool:
        return path.startswith(SKILLS_PREFIX)

    def _all_files(self) -> dict[str, FileData]:
        files: dict[str, FileData] = {}
        for meta in self.store.list_files(self.case_id):
            data = self.objects.get(self._object_key(meta["path"]))
            files[meta["path"]] = create_file_data(
                data.decode("utf-8"), created_at=meta["updated_at"]
            )
        for ev in self.store.list_evidence(self.case_id):
            path = f"{EVIDENCE_PREFIX}{ev['evidence_id']}.md"
            files[path] = create_file_data(
                evidence_text(self.objects, ev), created_at=ev["retrieved_at"]
            )
        for path, method in self.skill_files.items():
            files[path] = create_file_data(method)
        return files

    def _write_bytes(self, path: str, data: bytes) -> None:
        self.objects.put(self._object_key(path), data)
        self.store.put_file_meta(self.case_id, path, sha256_hex(data), len(data))

    # Shell-side helpers (not part of the agent-facing protocol).
    def write_text_direct(self, path: str, content: str) -> None:
        self._write_bytes(path, content.encode("utf-8"))

    def read_text_direct(self, path: str) -> str | None:
        meta = self.store.get_file_meta(self.case_id, path)
        if meta is None:
            return None
        return self.objects.get(self._object_key(path)).decode("utf-8")

    def delete_direct(self, path: str) -> None:
        if self.store.get_file_meta(self.case_id, path) is not None:
            self.store.delete_file_meta(self.case_id, path)

    # -- protocol: ls -------------------------------------------------------
    def ls(self, path: str) -> LsResult:
        path = path or "/"
        if not path.startswith("/"):
            path = "/" + path
        if not path.endswith("/"):
            path += "/"
        files = self._all_files()
        entries: list[dict] = []
        seen_dirs: set[str] = set()
        for file_path, file_data in sorted(files.items()):
            if not file_path.startswith(path):
                continue
            rest = file_path[len(path):]
            if "/" in rest:
                dir_name = rest.split("/", 1)[0]
                dir_path = f"{path}{dir_name}/"
                if dir_path not in seen_dirs:
                    seen_dirs.add(dir_path)
                    entries.append({"path": dir_path, "is_dir": True})
            else:
                entries.append(
                    {
                        "path": file_path,
                        "is_dir": False,
                        "size": len(file_data["content"].encode("utf-8")),
                        "modified_at": file_data.get("modified_at", ""),
                    }
                )
        if path != "/" and not entries and not any(
            p.startswith(path) for p in files
        ):
            return LsResult(error=f"Path '{path}' not found")
        return LsResult(entries=entries)

    # -- protocol: read -------------------------------------------------------
    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        files = self._all_files()
        file_data = files.get(file_path)
        if file_data is None:
            return ReadResult(error=f"File '{file_path}' not found")
        return slice_read_response(file_data, offset, limit)

    # -- protocol: write / edit / delete ---------------------------------------
    def write(self, file_path: str, content: str) -> WriteResult:
        if self._is_evidence(file_path):
            return WriteResult(error="Evidence files are read-only")
        if self._is_skill(file_path):
            return WriteResult(error="Skill files are read-only")
        if self.store.get_file_meta(self.case_id, file_path) is not None:
            return WriteResult(
                error=f"File '{file_path}' already exists. Use edit to modify it."
            )
        self._write_bytes(file_path, content.encode("utf-8"))
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if self._is_evidence(file_path):
            return EditResult(error="Evidence files are read-only")
        if self._is_skill(file_path):
            return EditResult(error="Skill files are read-only")
        content = self.read_text_direct(file_path)
        if content is None:
            return EditResult(error=f"File '{file_path}' not found")
        occurrences = content.count(old_string)
        if occurrences == 0:
            return EditResult(error="Old string not found in file")
        if occurrences > 1 and not replace_all:
            return EditResult(
                error=f"Old string appears {occurrences} times; use replace_all"
            )
        new_content = content.replace(old_string, new_string)
        self._write_bytes(file_path, new_content.encode("utf-8"))
        return EditResult(path=file_path, occurrences=occurrences)

    def delete(self, file_path: str) -> DeleteResult:
        if self._is_evidence(file_path):
            return DeleteResult(error="Evidence files are read-only")
        if self._is_skill(file_path):
            return DeleteResult(error="Skill files are read-only")
        if self.store.get_file_meta(self.case_id, file_path) is None:
            return DeleteResult(error=f"File '{file_path}' not found")
        self.store.delete_file_meta(self.case_id, file_path)
        return DeleteResult(path=file_path)

    # -- protocol: glob / grep ---------------------------------------------------
    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        base = (path or "/").rstrip("/")
        files = self._all_files()
        matches = [
            p
            for p in sorted(files)
            if (not base or p.startswith(base + "/") or p == base)
            and (
                fnmatch.fnmatch(p, pattern)
                or fnmatch.fnmatch(p.lstrip("/"), pattern)
                or fnmatch.fnmatch(p, "/" + pattern)
            )
        ]
        return GlobResult(matches=matches)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return GrepResult(error=f"Invalid pattern: {exc}")
        base = (path or "/").rstrip("/")
        matches: list[dict] = []
        for file_path, file_data in sorted(self._all_files().items()):
            if base and not (file_path.startswith(base + "/") or file_path == base):
                continue
            if glob and not fnmatch.fnmatch(file_path, glob):
                continue
            for lineno, line in enumerate(file_data["content"].splitlines(), 1):
                if regex.search(line):
                    matches.append(
                        {"path": file_path, "line": lineno, "text": line.strip()}
                    )
                    if max_count and len(matches) >= max_count:
                        return GrepResult(matches=matches, truncated=True)
        return GrepResult(matches=matches)

    # -- protocol: upload / download ---------------------------------------------
    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        responses = []
        for path, data in files:
            if self._is_evidence(path) or self._is_skill(path):
                responses.append(
                    FileUploadResponse(path=path, error="permission_denied")
                )
                continue
            self._write_bytes(path, data)
            responses.append(FileUploadResponse(path=path, error=None))
        return responses

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        files = self._all_files()
        responses = []
        for path in paths:
            file_data = files.get(path)
            if file_data is None:
                responses.append(
                    FileDownloadResponse(path=path, content=None, error="file_not_found")
                )
            else:
                responses.append(
                    FileDownloadResponse(
                        path=path,
                        content=file_data["content"].encode("utf-8"),
                        error=None,
                    )
                )
        return responses

    # -- async wrappers (the shell runs the sync API; these delegate) --------------
    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.read(file_path, offset, limit)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.glob(pattern, path)

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None, *, max_count: int | None = None) -> GrepResult:
        return self.grep(pattern, path, glob, max_count=max_count)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return self.upload_files(files)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return self.download_files(paths)

    async def adelete(self, file_path: str) -> DeleteResult:
        return self.delete(file_path)
