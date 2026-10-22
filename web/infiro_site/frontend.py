from pathlib import Path
from urllib.parse import unquote, urlsplit


def resolve_built_path(root: Path, url: str) -> Path | None:
    """Resolve only real files inside the build, including pretty page URLs."""
    path = unquote(urlsplit(url).path).strip("/")
    parts = path.split("/") if path else []
    if "\\" in path or "\x00" in path or any(part.startswith(".") for part in parts):
        return None
    root = Path(root).resolve()
    candidate = root.joinpath(*parts)
    if not candidate.suffix:
        candidate /= "index.html"
    candidate = candidate.resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return candidate
    return None
