"""Write a condition spec that fixes one protein chain from a downloaded CIF.

The spec is the design problem ``rfd4-generate`` already reads. A protein binder
is 60-110 residues and a peptide binder is 8-20. The target chain keeps its
backbone internal coordinates and its sequence. A deposit that does not name an
interface does not get a hotspot. Ligand and nucleic binders do not get a
protein contig. The class label stays on the annotated CIF, not on extra atoms.
"""

from __future__ import annotations

import gzip
import zlib
from pathlib import Path

_BACKBONE = frozenset({"N", "CA", "C", "O"})
_LENGTHS = {"protein": "60-110", "peptide": "8-20"}
_MISSING = frozenset({".", "?"})
#: What one structure's CIF text may expand to, for a compressed file only. A
#: compressed file's size on disk does not bound what it decompresses into, so
#: the bound is taken after. A plain file is read whole, as it always was.
_TEXT_LIMIT = 64 * 1024 * 1024


def cif_text(source_cif: Path) -> str | None:
    """A structure's CIF text, decompressed by the suffix the file carries.

    The screen reads ``*_generated.cif.zst`` and ``*_generated.cif.gz`` designs,
    so a compressed path reaches this module through ``follow_up``, and
    ``inputs=...monomer-short.cif.zst`` is the documented way to run the model
    in the first place. Read as plain text a compressed design raised
    ``UnicodeDecodeError``, and the inverse-folding spec was silently not
    written for it.

    None when the file will not open, will not decompress, or will not decode.
    ``zstandard`` is not a declared dependency of this package -- it arrives
    with the model stack -- so a ``.zst`` file on a machine without it reads as
    unreadable rather than raising.
    """
    name = source_cif.name
    compressed = name.endswith((".zst", ".gz", ".gzip"))
    errors: tuple[type[BaseException], ...] = (OSError, EOFError, ValueError, MemoryError, zlib.error)
    try:
        if name.endswith(".zst"):
            try:
                import zstandard
            except ImportError:
                return None
            errors = (*errors, zstandard.ZstdError)
            with source_cif.open("rb") as handle:
                # ``stream_reader`` rather than ``decompress``, which demands an
                # embedded content size these writers do not always set.
                raw = zstandard.ZstdDecompressor().stream_reader(handle).read(_TEXT_LIMIT + 1)
        elif name.endswith((".gz", ".gzip")):
            with gzip.open(source_cif, "rb") as handle:
                raw = handle.read(_TEXT_LIMIT + 1)
        else:
            raw = source_cif.read_bytes()
    except errors:
        return None
    if compressed and len(raw) > _TEXT_LIMIT:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeError:
        return None


def structure_stem(source_cif: Path) -> str:
    """A structure's name without its format suffix or its compression.

    ``Path.stem`` takes one suffix off, so ``binder_0_generated.cif.zst`` named
    its spec ``binder_0_generated.cif.yaml``.
    """
    name = source_cif.name
    for compression in (".zst", ".gzip", ".gz"):
        if name.endswith(compression):
            name = name[: -len(compression)]
            break
    return Path(name).stem


def write_design_spec(
    source_cif: Path,
    work_dir: Path,
    *,
    chain: str,
    binder_kind: str,
) -> Path | None:
    """Write ``{stem}.yaml`` beside ``source_cif``, or return None when no spec applies.

    ``chain`` is the RCSB entity id, or a label or author chain id. The contig
    uses the label chain id, which is what a preserved segment addresses.
    """
    length = _LENGTHS.get(binder_kind.casefold())
    if length is None or not chain.strip():
        return None
    quoted = _yaml_quote(source_cif.name)
    if quoted is None:
        return None
    text = cif_text(source_cif)
    if text is None:
        return None
    span = _polymer_span(text, chain.strip())
    if span is None:
        return None
    asym, start, end = span
    work_dir.mkdir(parents=True, exist_ok=True)
    dest = work_dir / f"{structure_stem(source_cif)}.yaml"
    dest.write_text(_render(quoted, length, asym, start, end), encoding="utf-8", newline="\n")
    return dest


def _polymer_span(text: str, chain: str) -> tuple[str, int, int] | None:
    residues = _backbone_residues(text)
    chosen = [row for row in residues if row[1] == chain]
    if not chosen:
        chosen = [row for row in residues if chain in {row[0], row[2]}]
    if not chosen:
        return None
    by_asym: dict[str, set[int]] = {}
    order: list[str] = []
    for asym, _entity, _auth, seq in chosen:
        if asym not in by_asym:
            order.append(asym)
            by_asym[asym] = set()
        by_asym[asym].add(seq)
    asym = max(order, key=lambda name: (len(by_asym[name]), -order.index(name)))
    if not asym.isascii() or not asym.isalpha():
        return None
    seq_ids = sorted(by_asym[asym])
    # A missing residue number stays inside the span. The contig keeps every
    # residue that exists in that range and does not require a filled sequence.
    return asym, seq_ids[0], seq_ids[-1]


def _backbone_residues(text: str) -> list[tuple[str, str, str, int]]:
    """Label chain, entity id, author chain, and sequence id for complete backbone residues."""
    grouped: dict[tuple[str, int], set[str]] = {}
    identity: dict[tuple[str, int], tuple[str, str]] = {}
    for row in _atom_site_rows(text):
        if row.get("group_PDB") != "ATOM":
            continue
        asym = row.get("label_asym_id", "")
        entity = row.get("label_entity_id", "")
        auth = row.get("auth_asym_id", "")
        seq = _seq_id(row.get("label_seq_id", ""))
        atom = row.get("label_atom_id", "")
        if not asym or seq is None or atom not in _BACKBONE:
            continue
        key = (asym, seq)
        grouped.setdefault(key, set()).add(atom)
        identity.setdefault(key, (entity, auth))
    residues: list[tuple[str, str, str, int]] = []
    for (asym, seq), atoms in grouped.items():
        if atoms != _BACKBONE:
            continue
        entity, auth = identity[(asym, seq)]
        residues.append((asym, entity, auth, seq))
    return residues


def _atom_site_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    index = 0
    found: list[dict[str, str]] = []
    while index < len(lines):
        if lines[index].strip() != "loop_":
            index += 1
            continue
        index += 1
        tags: list[str] = []
        while index < len(lines) and lines[index].lstrip().startswith("_"):
            tags.append(lines[index].strip().split()[0])
            index += 1
        width = len(tags)
        is_atom = bool(tags) and tags[0].startswith("_atom_site.")
        columns = [tag.removeprefix("_atom_site.") for tag in tags]
        pending: list[str] = []
        while True:
            if width and len(pending) >= width:
                row, pending = pending[:width], pending[width:]
                if is_atom:
                    found.append(dict(zip(columns, row, strict=True)))
                continue
            if index >= len(lines):
                break
            line = lines[index]
            if not pending and _ends_loop(line):
                break
            if line.startswith(";"):
                index = _consume_text_field(lines, index)
                pending.append("")
                continue
            pending.extend(_tokenize(line))
            index += 1
    return found


def _ends_loop(line: str) -> bool:
    stripped = line.strip()
    return stripped in {"", "#"} or stripped.startswith(("loop_", "data_", "save_", "_", "global_"))


def _consume_text_field(lines: list[str], index: int) -> int:
    index += 1
    while index < len(lines) and lines[index].strip() != ";":
        index += 1
    if index < len(lines):
        index += 1
    return index


def _tokenize(line: str) -> list[str]:
    tokens: list[str] = []
    cursor = 0
    while cursor < len(line):
        if line[cursor].isspace():
            cursor += 1
            continue
        if line[cursor] in {'"', "'"}:
            quote = line[cursor]
            cursor += 1
            start = cursor
            while cursor < len(line) and line[cursor] != quote:
                cursor += 1
            tokens.append(line[start:cursor])
            cursor += 1
            continue
        start = cursor
        while cursor < len(line) and not line[cursor].isspace():
            cursor += 1
        tokens.append(line[start:cursor])
    return tokens


def _seq_id(token: str) -> int | None:
    if token in _MISSING:
        return None
    try:
        value = int(token)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


def _yaml_quote(value: str) -> str | None:
    if any(char in value for char in '"\\\n\r'):
        return None
    return f'"{value}"'


def backbone_spans(source_cif: Path) -> dict[str, tuple[int, int]]:
    """Label chain to the first and last residue that has N, CA, C, and O."""
    text = cif_text(source_cif)
    if text is None:
        raise ValueError(f"{source_cif.name} could not be read.")
    grouped: dict[str, list[int]] = {}
    for asym, _entity, _auth, seq in _backbone_residues(text):
        if not asym.isascii() or not asym.isalpha():
            continue
        grouped.setdefault(asym, []).append(seq)
    spans: dict[str, tuple[int, int]] = {}
    for asym, seq_ids in grouped.items():
        ordered = sorted(set(seq_ids))
        spans[asym] = (ordered[0], ordered[-1])
    return spans


def write_inverse_folding_spec(
    generated_cif: Path,
    fresh_dir: Path,
    *,
    designed_chain: str,
) -> Path:
    """Write a condition spec that keeps this backbone and redesigns the designed chain.

    The designed chain gets backbone internal coordinates only. Its sequence and its
    sidechains stay free. Each other chain keeps its backbone and its sequence. The
    contig is those existing residue spans. No hotspot is set, and the file is written
    only inside ``fresh_dir``.
    """
    if not generated_cif.is_file():
        raise ValueError(f"{generated_cif.name} is not a generated structure.")
    spans = backbone_spans(generated_cif)
    designed = _resolve_chain(spans, designed_chain)
    if designed is None:
        raise ValueError("The designed chain is not in the generated structure.")
    targets = sorted(name for name in spans if name != designed)
    if not targets:
        raise ValueError("The generated structure has no target chain to keep.")
    fresh_dir.mkdir(parents=True, exist_ok=True)
    dest = fresh_dir / "inverse_folding.yaml"
    if dest.resolve() == generated_cif.resolve():
        raise ValueError("The inverse-folding spec would overwrite the generated structure.")
    quoted = _spec_input(generated_cif, fresh_dir)
    dest.write_text(_render_inverse(quoted, spans, designed, targets), encoding="utf-8", newline="\n")
    return dest


def _resolve_chain(spans: dict[str, tuple[int, int]], chain: str) -> str | None:
    text = chain.strip()
    if text in spans:
        return text
    letter = text.split("_", 1)[0]
    if letter in spans:
        return letter
    return None


def _spec_input(generated_cif: Path, fresh_dir: Path) -> str:
    source = generated_cif.resolve()
    try:
        relative = source.relative_to(fresh_dir.resolve(), walk_up=True)
    except ValueError:
        relative = source
    quoted = _yaml_quote(relative.as_posix())
    if quoted is None:
        raise ValueError("The generated structure path cannot be stored in the condition spec.")
    return quoted


def _span_token(asym: str, start: int, end: int) -> str:
    return f"{asym}{start}" if start == end else f"{asym}{start}-{end}"


def _render_inverse(
    quoted_name: str,
    spans: dict[str, tuple[int, int]],
    designed: str,
    targets: list[str],
) -> str:
    tokens = [_span_token(designed, *spans[designed])]
    for target in targets:
        tokens.append("/0")
        tokens.append(_span_token(target, *spans[target]))
    backbone = [
        f"        - select: \"sel('{chain}/*/*/[N,CA,C,O]') and not has_nan_coord()\"" for chain in (designed, *targets)
    ]
    sequence = [f"        - select: \"sel('{target}')\"" for target in targets]
    lines = [
        "# Inverse folding keeps the designed backbone and redesigns its sequence.",
        "# The target keeps its backbone and its sequence. No hotspot is set.",
        "binder:",
        f"  input: {quoted_name}",
        f'  contig: "{", ".join(tokens)}"',
        "  conditions:",
        "    C_CRDGRP:",
        "      1:",
        *backbone,
        "    C_SEQ:",
        "      true:",
        *sequence,
        "",
    ]
    return "\n".join(lines)


def _render(quoted_name: str, length: str, asym: str, start: int, end: int) -> str:
    span = f"{asym}{start}" if start == end else f"{asym}{start}-{end}"
    backbone = f"sel('{asym}/*/*/[N,CA,C,O]') and not has_nan_coord()"
    sequence = f"sel('{asym}')"
    return (
        "# No hotspot is set because the deposit does not name an interface.\n"
        "binder:\n"
        f"  input: {quoted_name}\n"
        f'  contig: "{length}, /0, {span}"\n'
        "  conditions:\n"
        "    C_CRDGRP:\n"
        "      1:\n"
        f'        - select: "{backbone}"\n'
        "    C_SEQ:\n"
        "      true:\n"
        f'        - select: "{sequence}"\n'
    )
