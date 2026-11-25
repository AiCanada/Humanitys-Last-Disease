"""Read bounded brightfield image fields and their operator-supplied binary ROIs."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import zlib
from contextlib import ExitStack
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

MAX_PIXELS = 4_194_304
MAX_FILE_BYTES = 20 * 1024 * 1024
_FORMATS = {".png": "PNG", ".tif": "TIFF", ".tiff": "TIFF"}
_LIMIT = "cellprofiler_image_budget_exceeded"
_UNAVAILABLE = "cellprofiler_file_unavailable"
_PATH = "cellprofiler_invalid_image_path"


class ImageError(ValueError):
    """A named image input failure safe to include in a screening record."""


@dataclass(frozen=True, slots=True)
class RegionImage:
    rgb: np.ndarray
    mask: np.ndarray
    width: int
    height: int
    roi_pixels: int
    image_sha256: str
    mask_sha256: str
    pixel_sha256: str
    source_byte_count: int


def _path(directory: Path, value: str) -> Path:
    if not isinstance(value, str) or not 0 < len(value) <= 512:
        raise ImageError(_PATH)
    parts = value.replace("\\", "/").split("/")
    if any(
        not p
        or p in {".", ".."}
        or p.endswith((" ", "."))
        or not p.isprintable()
        or any(c in '<>:"|?*' for c in p)
        or PureWindowsPath(p).is_reserved()
        for p in parts
    ):
        raise ImageError(_PATH)
    path = directory.absolute().joinpath(*parts)
    if path.suffix.lower() not in _FORMATS:
        raise ImageError(_PATH)
    return path


def _meta(path: Path) -> os.stat_result:
    try:
        if path.resolve(strict=False) != path:
            raise ImageError(_PATH)
        info = path.lstat()
    except (OSError, RuntimeError) as exc:
        raise ImageError(_UNAVAILABLE) from exc
    if stat.S_ISLNK(info.st_mode) or info.st_nlink != 1:
        raise ImageError(_PATH)
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ImageError(_UNAVAILABLE)
    if info.st_size > MAX_FILE_BYTES:
        raise ImageError(_LIMIT)
    return info


def check_region_files(directory: Path, image: str, mask: str) -> int:
    """Check locality and file limits, returning the combined encoded size without decoding."""
    image_info, mask_info = (_meta(_path(directory, name)) for name in (image, mask))
    if os.path.samestat(image_info, mask_info):
        raise ImageError(_PATH)
    return image_info.st_size + mask_info.st_size


def _signature(info: os.stat_result) -> tuple[int, ...]:
    return info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _snapshot(path: Path) -> bytes:
    before = _signature(_meta(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        with ExitStack() as stack:
            if os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY"):
                # Anchor each directory to a descriptor so a concurrent symlink swap cannot redirect the read.
                parent_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
                parent = os.open(path.anchor, parent_flags)
                stack.callback(os.close, parent)
                for component in path.parts[1:-1]:
                    parent = os.open(component, parent_flags, dir_fd=parent)
                    stack.callback(os.close, parent)
                fd = os.open(path.name, flags, dir_fd=parent)
            else:
                fd = os.open(path, flags)
            stream = stack.enter_context(os.fdopen(fd, "rb"))
            opened = os.fstat(fd)
            if _signature(opened) != before:
                raise ImageError(_UNAVAILABLE)
            raw = stream.read(opened.st_size + 1)
            if len(raw) != opened.st_size or _signature(os.fstat(fd)) != before or _signature(_meta(path)) != before:
                raise ImageError(_UNAVAILABLE)
            return raw
    except OSError as exc:
        raise ImageError(_UNAVAILABLE) from exc


def _size(width: int, height: int) -> None:
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise ImageError(_LIMIT)


def _png(raw: bytes, mask: bool, invalid: str) -> None:
    # Pillow presents RGB16 PNGs as RGB8 and scales low-bit greyscale to 8-bit. Check the native IHDR.
    if len(raw) < 33 or raw[:16] != b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR":
        raise ImageError(invalid)
    _size(*struct.unpack_from(">II", raw, 16))
    if (raw[24], raw[25]) not in ({(1, 0), (8, 0)} if mask else {(8, 2)}):
        raise ImageError(invalid)
    data = memoryview(raw)
    offset = 8
    while offset + 12 <= len(raw):
        length, chunk = struct.unpack_from(">I4s", data, offset)
        end = offset + 8 + length
        if end + 4 > len(raw) or zlib.crc32(data[offset + 4 : end]) != struct.unpack_from(">I", data, end)[0]:
            raise ImageError(invalid)
        if chunk in {b"acTL", b"fcTL", b"fdAT", b"tRNS"}:
            raise ImageError(invalid)
        offset = end + 4
        if chunk == b"IEND":
            if length or offset != len(raw):
                raise ImageError(invalid)
            return
    raise ImageError(invalid)


def _tiff(raw: bytes, invalid: str) -> None:
    # Guard directory and strip/tile tables before the decoder can allocate a tile list.
    endian = {b"II*\x00": "<", b"MM\x00*": ">"}.get(raw[:4])
    if endian is None or len(raw) < 8:
        raise ImageError(invalid)
    offset = struct.unpack_from(endian + "I", raw, 4)[0]
    if not 8 <= offset <= len(raw) - 2:
        raise ImageError(invalid)
    count = struct.unpack_from(endian + "H", raw, offset)[0]
    end = offset + 2 + 12 * count
    if not 0 < count <= 4096 or end + 4 > len(raw) or struct.unpack_from(endian + "I", raw, end)[0]:
        raise ImageError(invalid)
    seen = set()
    for tag, _kind, size, _entry in struct.iter_unpack(endian + "HHI4s", raw[offset + 2 : end]):
        if tag in seen or tag == 330:  # SubIFDs identify a hierarchy, often a multi-resolution or multi-plane file.
            raise ImageError(invalid)
        if tag in {273, 279, 324, 325} and size > 65_536:
            raise ImageError(_LIMIT)
        seen.add(tag)


def _tiff_encoding(frame: Image.Image, length: int, mask: bool, invalid: str) -> None:
    tags = frame.tag_v2
    if tags.get(274, 1) != 1:
        raise ImageError("cellprofiler_grid_mismatch")
    bits = tuple(tags.get(258, (1,)))
    if (
        tags.get(259, 1) not in {1, 5, 8, 32946, 32773, *({3, 4} if mask else set())}
        or tags.get(262) != (1 if mask else 2)
        or tags.get(277, 1) != (1 if mask else 3)
        or bits not in ({(1,), (8,)} if mask else {(8, 8, 8)})
        or any(v != 1 for v in tags.get(339, (1,)))
        or tags.get(338)
    ):
        raise ImageError(invalid)
    if tags.get(284, 1) not in (1, 2) or (324 in tags) == (273 in tags):
        raise ImageError(invalid)
    width, height = frame.size
    planes = tags[277] if tags.get(284, 1) == 2 else 1
    if 324 in tags:
        tile_width, tile_height = tags[322], tags[323]
        _size(tile_width, tile_height)
        segments = ((height + tile_height - 1) // tile_height) * ((width + tile_width - 1) // tile_width)
        offsets, lengths = tags[324], tags[325]
    else:
        rows = tags.get(278, height)
        if rows < 1:
            raise ImageError(invalid)
        segments = (height + rows - 1) // rows
        offsets, lengths = tags[273], tags[279]
    if (
        len(offsets) != len(lengths)
        or len(offsets) != segments * planes
        or any(start < 8 or size <= 0 or start + size > length for start, size in zip(offsets, lengths, strict=True))
    ):
        raise ImageError(invalid)


def _decode(raw: bytes, fmt: str, *, mask: bool) -> np.ndarray:
    try:
        import numpy as np
        from PIL import Image
    except (ImportError, OSError) as exc:
        raise ImageError("cellprofiler_image_reader_unavailable") from exc
    invalid = "cellprofiler_mask_format" if mask else "cellprofiler_image_format"
    try:
        if fmt == "PNG":
            _png(raw, mask, invalid)
        else:
            _tiff(raw, invalid)
        with Image.open(BytesIO(raw), formats=[fmt]) as frame:
            _size(*frame.size)
            if fmt == "TIFF":
                _tiff_encoding(frame, len(raw), mask, invalid)
            if frame.mode not in ({"L", "1"} if mask else {"RGB"}) or frame.is_animated:
                raise ImageError(invalid)
            if fmt == "PNG" and frame.getexif().get(274, 1) != 1:
                raise ImageError("cellprofiler_grid_mismatch")
            pixels = np.asarray(frame)
    except ImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageError(_LIMIT) from exc
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        SyntaxError,
        EOFError,
        struct.error,
    ) as exc:
        raise ImageError(invalid) from exc
    if mask:
        if pixels.ndim != 2 or pixels.dtype not in (np.uint8, np.bool_):
            raise ImageError(invalid)
        on = pixels.max()
        if on not in (1, 255) or np.any((pixels != 0) & (pixels != on)):
            raise ImageError(invalid)
        return pixels.astype(np.bool_, copy=False)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ImageError(invalid)
    return pixels


def read_region(directory: Path, image: str, mask: str) -> RegionImage:
    """Decode a native RGB field and exactly aligned nonempty binary region. Return no source metadata."""
    image_path, mask_path = _path(directory, image), _path(directory, mask)
    if image_path == mask_path:
        raise ImageError(_PATH)
    rgb_raw, mask_raw = _snapshot(image_path), _snapshot(mask_path)
    rgb = _decode(rgb_raw, _FORMATS[image_path.suffix.lower()], mask=False)
    region = _decode(mask_raw, _FORMATS[mask_path.suffix.lower()], mask=True)
    if rgb.shape[:2] != region.shape:
        raise ImageError("cellprofiler_grid_mismatch")
    height, width = region.shape
    pixels = hashlib.sha256(struct.pack(">II", width, height))
    pixels.update(memoryview(rgb).cast("B"))
    return RegionImage(
        rgb,
        region,
        width,
        height,
        int(region.sum()),
        hashlib.sha256(rgb_raw).hexdigest(),
        hashlib.sha256(mask_raw).hexdigest(),
        pixels.hexdigest(),
        len(rgb_raw) + len(mask_raw),
    )
