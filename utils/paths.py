from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Sequence, Union

_SENTINELS: Sequence[str] = ("config.yaml", "README.md", "requirements.txt", ".git", "models", "weights")

def _find_project_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path(__file__).resolve()).parent
    for _ in range(20):
        if any((cur / s).exists() for s in _SENTINELS):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return (start or Path(__file__).resolve()).parent

class _Paths:
    def __init__(self):
        self.base: Optional[Path] = None 

    def init(self, base: Optional[Union[str, Path]] = None) -> Path:
        # Ưu tiên: base truyền vào → env TSIGN_HOME → auto tìm gốc
        if base:
            b = Path(str(base)).expanduser().resolve()
        else:
            env = os.environ.get("TSIGN_HOME")
            b = Path(env).expanduser().resolve() if env else _find_project_root()
        self.base = b
        return b

    @staticmethod
    def _strip_aliases(s: str) -> str:
        # Hỗ trợ cả 'local/...', 'local:...', 'local_root/...', 'local_root:...'
        if s.startswith("local_root/"):
            return s[len("local_root/") :]
        if s.startswith("local_root:"):
            rest = s[len("local_root:") :]
            return rest[1:] if rest.startswith("/") else rest
        if s.startswith("local/"):
            return s[len("local/") :]
        if s.startswith("local:"):
            rest = s[len("local:") :]
            return rest[1:] if rest.startswith("/") else rest
        return s

    def resolve(self, pathlike: Union[str, Path, int, None]) -> Union[str, int, None]:
        """Chuyển alias → tuyệt đối dựa trên base; giữ nguyên URL; int → webcam id."""
        if pathlike is None:
            return None
        if isinstance(pathlike, int):
            return pathlike

        s = str(pathlike).strip()
        if s.isdigit():
            return int(s)
        if s.startswith(("http://", "https://", "rtsp://")):
            return s

        s = self._strip_aliases(s)
        p = Path(s).expanduser()
        if p.is_absolute():
            return str(p)

        if self.base is None:
            self.init()
        return str((self.base or Path.cwd()).joinpath(p))

    def ensure_parent_dir(self, pathlike: Union[str, Path]) -> None:
        Path(str(pathlike)).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    def p(self, *parts: Union[str, Path]) -> str:
        if self.base is None:
            self.init()
        return str(self.base.joinpath(*map(str, parts)))

PATHS = _Paths()
