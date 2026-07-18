"""Load markdown docs and split them into citable section chunks.

Each chunk keeps its source filename and section heading so every answer can
cite exactly where a claim came from: [02-pricing-sheet.md § Financing].
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    doc: str      # source filename, e.g. "02-pricing-sheet.md"
    section: str  # section heading, e.g. "Diagnostic & Service Call Fees"
    text: str     # section body

    @property
    def citation(self) -> str:
        return f"[{self.doc} § {self.section}]"


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown document on '## ' headings.

    Returns a list of (heading, body). Content before the first '## ' heading
    becomes an "Overview" section (minus the top-level '# ' title line).
    """
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    body: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("## "):
            text = "\n".join(body).strip()
            if text:
                sections.append((heading, text))
            heading = line[3:].strip()
            body = []
        elif line.startswith("# "):
            continue  # top-level title, not a section
        else:
            body.append(line)

    text = "\n".join(body).strip()
    if text:
        sections.append((heading, text))
    return sections


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Load every .md file in docs_dir and return its section chunks."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        for heading, body in _split_sections(path.read_text(encoding="utf-8")):
            chunks.append(Chunk(doc=path.name, section=heading, text=body))
    if not chunks:
        raise FileNotFoundError(f"No markdown documents found in {docs_dir}")
    return chunks
