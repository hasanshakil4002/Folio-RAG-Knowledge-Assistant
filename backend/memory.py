"""Per-session conversation memory (stored on disk, survives restarts)."""
import json
import threading

from . import config as cfg


class Memory:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.data: dict[str, list[dict]] = {}
        if cfg.SESSIONS_FILE.exists():
            try:
                self.data = json.loads(cfg.SESSIONS_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}

    def _save(self) -> None:
        cfg.SESSIONS_FILE.write_text(json.dumps(self.data, ensure_ascii=False),
                                     encoding="utf-8")

    def get(self, sid: str) -> list[dict]:
        return list(self.data.get(sid, []))

    def add(self, sid: str, question: str, answer: str) -> None:
        with self.lock:
            msgs = self.data.setdefault(sid, [])
            msgs.append({"role": "user", "content": question})
            msgs.append({"role": "assistant", "content": answer})
            limit = cfg.MAX_HISTORY_TURNS * 4  # storage cap (fewer go into the prompt)
            self.data[sid] = msgs[-limit:]
            self._save()

    def clear(self, sid: str) -> None:
        with self.lock:
            self.data.pop(sid, None)
            self._save()
