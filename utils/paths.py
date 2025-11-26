"""
paths.py
Refactor & Fix version – hỗ trợ:
- Alias "local/", "local_root/"
- Webcam ID (int or numeric string)
- URL (http/https/rtsp)
- Windows absolute paths (C:/..., D:/...)
- Auto-detect project root
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Sequence, Union


# Các file/thư mục đặc trưng giúp xác định project root
_SENTINELS: Sequence[str] = (
    "config.yaml",
    "README.md",
    "requirements.txt",
    "models",
    "weights",
    ".git",
)


# ===============================================================
# TÌM PROJECT ROOT (auto detect)
# ===============================================================

def _find_project_root(start: Optional[Path] = None) -> Path:
    """
    Đi ngược từ file hiện tại lên 20 tầng thư mục để tìm root project:
    - Nếu có bất kỳ sentinel nào → đó là root
    - Nếu tìm không thấy → fallback về start
    """
    cur = (start or Path(__file__).resolve()).parent

    for _ in range(20):
        if any((cur / s).exists() for s in _SENTINELS):
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    return (start or Path(__file__).resolve()).parent


# ===============================================================
# CLASS PATHS (điều phối mọi đường dẫn trong project)
# ===============================================================

class _Paths:
    def __init__(self):
        self.base: Optional[Path] = None

    # -----------------------------------------------------------

    def init(self, base: Optional[Union[str, Path]] = None) -> Path:
        """
        Ưu tiên:
        1. base truyền vào
        2. TSIGN_HOME từ môi trường
        3. Auto detect project root
        """
        if base:
            b = Path(str(base)).expanduser().resolve()
        else:
            env = os.environ.get("TSIGN_HOME")
            if env:
                b = Path(env).expanduser().resolve()
            else:
                b = _find_project_root()

        self.base = b
        return b

    # -----------------------------------------------------------
    # ALIAS HANDLING
    # -----------------------------------------------------------

    @staticmethod
    def _strip_aliases(s: str) -> str:
        """
        Xử lý các alias:
          - local/xxx
          - local:xxx
          - local_root/xxx
          - local_root:xxx
        Trả về phần path phía sau alias.
        """
        # local_root/
        if s.startswith("local_root/"):
            return s[len("local_root/"):]
        if s.startswith("local_root:"):
            rest = s[len("local_root:"):]
            return rest[1:] if rest.startswith("/") else rest

        # local/
        if s.startswith("local/"):
            return s[len("local/"):]
        if s.startswith("local:"):
            rest = s[len("local:"):]
            return rest[1:] if rest.startswith("/") else rest

        return s

    # -----------------------------------------------------------
    # PATH RESOLUTION
    # -----------------------------------------------------------

    def resolve(self, pathlike: Union[str, Path, int, None]) -> Union[str, int, None]:
        """
        Chuyển đường dẫn "tương đối" – "alias" – "int" → thành:
        - webcam id (int)
        - URL (giữ nguyên)
        - absolute path (chuẩn Windows + Linux)
        """
        # Trường hợp None
        if pathlike is None:
            return None

        # Webcam ID dạng int (0, 1...)
        if isinstance(pathlike, int):
            return pathlike

        s = str(pathlike).strip()

        # Webcam ID dạng string số ("0", "1")
        if s.isdigit():
            return int(s)

        # URL
        if s.startswith(("http://", "https://", "rtsp://")):
            return s

        # Windows absolute path: C:/ABC/..., D:\ABC\...
        # Path.is_absolute() nhận diện đúng cả Windows + Linux
        s = self._strip_aliases(s)
        p = Path(s).expanduser()

        if p.is_absolute():
            return str(p)

        # Nếu là relative path → join vào project base
        if self.base is None:
            self.init()

        return str(self.base.joinpath(p))

    # -----------------------------------------------------------

    def ensure_parent_dir(self, pathlike: Union[str, Path]) -> None:
        """
        Đảm bảo thư mục cha tồn tại trước khi lưu file.
        """
        Path(str(pathlike)).expanduser().resolve().parent.mkdir(
            parents=True, exist_ok=True
        )

    # -----------------------------------------------------------

    def p(self, *parts: Union[str, Path]) -> str:
        """
        Generate absolute path từ base + parts
        """
        if self.base is None:
            self.init()
        return str(self.base.joinpath(*map(str, parts)))


# Global instance
PATHS = _Paths()
