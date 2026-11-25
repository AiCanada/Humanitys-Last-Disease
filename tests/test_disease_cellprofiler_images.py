"""Exercise original RGB precision, field attribution, and region boundaries using encoded images."""

import hashlib
import os
import struct
import zlib

import pytest

from rfproteina.disease import cellprofiler_images as cp

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def fields(tmp_path):
    images = tmp_path / "fields"
    images.mkdir()
    rgb = np.array(
        [[[255, 251, 252], [102, 80, 48], [95, 101, 161]], [[120, 112, 150], [170, 132, 90], [0, 0, 0]]], np.uint8
    )
    roi = np.array([[0, 255, 255], [0, 255, 0]], np.uint8)
    Image.fromarray(rgb).save(images / "field.png")
    Image.fromarray(roi).save(images / "mask.png")
    return tmp_path, images, rgb, roi


@pytest.mark.parametrize("compression", ["raw", "tiff_lzw", "tiff_adobe_deflate", "packbits"])
def test_region_native_values_hashes_and_duplicate_field_across_containers(fields, compression):
    root, images, rgb, roi = fields
    Image.fromarray(rgb).save(images / "field.tif", compression=compression)
    Image.fromarray(roi == 255).save(images / "mask.tiff", compression=compression)
    png = cp.read_region(root, "fields/field.png", "fields/mask.png")
    tiff = cp.read_region(root, r"fields\field.tif", "fields/mask.tiff")
    assert png.pixel_sha256 == tiff.pixel_sha256 and png.image_sha256 != tiff.image_sha256
    assert png.image_sha256 == hashlib.sha256((images / "field.png").read_bytes()).hexdigest()
    assert png.mask_sha256 == hashlib.sha256((images / "mask.png").read_bytes()).hexdigest()
    for region in (png, tiff):
        assert (region.width, region.height, region.roi_pixels) == (3, 2, 3)
        np.testing.assert_array_equal(region.rgb, rgb)
        np.testing.assert_array_equal(region.mask, roi == 255)
        assert region.rgb.dtype == np.uint8 and region.mask.dtype == np.bool_


@pytest.mark.parametrize("mode", ["bit", "zero_one", "all"])
def test_binary_mask_supported_encodings(fields, mode):
    root, images, _, roi = fields
    data = (
        roi.astype(bool)
        if mode == "bit"
        else ((roi != 0).astype(np.uint8) if mode == "zero_one" else np.ones_like(roi))
    )
    Image.fromarray(data).save(images / "mask.png")
    assert cp.read_region(root, "fields/field.png", "fields/mask.png").roi_pixels == int(data.sum())


@pytest.mark.parametrize("values", [[0, 1, 255], [0, 128, 255], [0, 0, 0], [2, 2, 2]])
def test_reject_empty_soft_and_mixed_mask_conventions(fields, values):
    root, images, _, _ = fields
    Image.fromarray(np.array([values, values], np.uint8)).save(images / "mask.png")
    with pytest.raises(cp.ImageError, match="cellprofiler_mask_format"):
        cp.read_region(root, "fields/field.png", "fields/mask.png")


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png16():
    hdr = struct.pack(">IIBBBBB", 3, 2, 16, 2, 0, 0, 0)
    rows = (b"\0" + b"\x12\x34" * 9) * 2
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", hdr) + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b"")


@pytest.mark.parametrize("style", ["palette", "alpha", "rgb16", "animated", "transparent", "jpeg", "truncated", "crc"])
def test_no_color_conversion_frame_selection_or_truncation(fields, style):
    root, images, rgb, _ = fields
    source = images / "field.png"
    field = Image.fromarray(rgb)
    if style == "palette":
        field.convert("P").save(source)
    elif style == "alpha":
        field.convert("RGBA").save(source)
    elif style == "rgb16":
        source.write_bytes(_png16())
    elif style == "animated":
        field.save(source, save_all=True, append_images=[Image.fromarray(255 - rgb)])
    elif style == "transparent":
        field.save(source, transparency=(255, 251, 252))
    elif style == "jpeg":
        field.save(source, format="JPEG")
    elif style == "truncated":
        source.write_bytes(source.read_bytes()[:-1])
    else:
        data = bytearray(source.read_bytes())
        data[data.index(b"IDAT") + 4] ^= 1
        source.write_bytes(data)
    with pytest.raises(cp.ImageError, match="cellprofiler_image_format"):
        cp.read_region(root, "fields/field.png", "fields/mask.png")


@pytest.mark.parametrize(
    "style",
    [
        "rotated_png",
        "rotated_tiff",
        "wrong_size",
        "multipage_tiff",
        "jpeg_tiff",
        "truncated_tiff",
        "rgb16_tiff",
        "missing_strip",
    ],
)
def test_reject_misaligned_or_ambiguous_tiff_sources(fields, style):
    root, images, rgb, _ = fields
    path = "fields/field.tif"
    field = Image.fromarray(rgb)
    code = "cellprofiler_image_format"
    if style in {"rotated_png", "rotated_tiff"}:
        options = Image.Exif()
        options[274] = 6
        path = "fields/field.png" if style == "rotated_png" else path
        field.save(root / path, exif=options)
        code = "cellprofiler_grid_mismatch"
    elif style == "wrong_size":
        field.save(root / path)
        Image.fromarray(np.ones((3, 2), np.uint8)).save(images / "mask.png")
        code = "cellprofiler_grid_mismatch"
    elif style == "multipage_tiff":
        field.save(root / path, save_all=True, append_images=[Image.fromarray(255 - rgb)])
    elif style == "jpeg_tiff":
        field.save(root / path, compression="jpeg")
    elif style == "truncated_tiff":
        field.save(root / path)
        (root / path).write_bytes((root / path).read_bytes()[:-2])
    else:
        # Start from the writer's single uncompressed strip and update its native directory entries.
        field.save(root / path)
        data = bytearray((root / path).read_bytes())
        directory = struct.unpack_from("<I", data, 4)[0]
        count = struct.unpack_from("<H", data, directory)[0]
        entries = {
            struct.unpack_from("<H", data, pos)[0]: pos + 8
            for pos in range(directory + 2, directory + 2 + 12 * count, 12)
        }
        if style == "missing_strip":
            struct.pack_into("<I", data, entries[278], 1)
        else:
            bits = struct.unpack_from("<I", data, entries[258])[0]
            struct.pack_into("<3H", data, bits, 16, 16, 16)
            pixels = (rgb.astype("<u2") * 257).tobytes()
            struct.pack_into("<I", data, entries[279], len(pixels))
            strip = struct.unpack_from("<I", data, entries[273])[0]
            data[strip:] = pixels
        (root / path).write_bytes(data)
    with pytest.raises(cp.ImageError, match=code):
        cp.read_region(root, path, "fields/mask.png")


@pytest.mark.parametrize(
    "name",
    [
        "../field.png",
        "fields/../fields/field.png",
        "",
        "/tmp/field.png",
        r"\\host\share\field.png",
        r"C:field.png",
        r"C:\field.png",
        "fields/field:stream.png",
        "fields/NUL.tif",
        "fields/field.png.",
        "fields//field.png",
        "fields/field.jpeg",
    ],
)
def test_reject_filename_aliases_and_foreign_paths(fields, name):
    root, *_ = fields
    with pytest.raises(cp.ImageError, match="cellprofiler_invalid_image_path"):
        cp.check_region_files(root, name, "fields/mask.png")


def test_preflight_and_reader_withhold_linked_files(fields):
    root, images, *_ = fields
    (root / "linked").symlink_to(images, target_is_directory=True)
    (root / "leaf.png").symlink_to(images / "field.png")
    for link in ("linked/field.png", "leaf.png"):
        for reader in (cp.read_region, cp.check_region_files):
            with pytest.raises(cp.ImageError, match="cellprofiler_invalid_image_path"):
                reader(root, link, "fields/mask.png")
    os.link(images / "field.png", root / "hard.png")
    with pytest.raises(cp.ImageError, match="cellprofiler_invalid_image_path"):
        cp.read_region(root, "hard.png", "fields/mask.png")


def test_stat_preflight_and_limits_are_enforced_before_decode(fields, monkeypatch):
    root, images, *_ = fields
    (images / "mask.png").write_bytes(b"not an image")
    assert cp.check_region_files(root, "fields/field.png", "fields/mask.png") == (
        images / "field.png"
    ).stat().st_size + len(b"not an image")
    with pytest.raises(cp.ImageError, match="cellprofiler_mask_format"):
        cp.read_region(root, "fields/field.png", "fields/mask.png")
    with pytest.raises(cp.ImageError, match="cellprofiler_file_unavailable"):
        cp.check_region_files(root, "fields/gone.png", "fields/mask.png")
    monkeypatch.setattr(cp, "MAX_PIXELS", 5)
    with pytest.raises(cp.ImageError, match="cellprofiler_image_budget_exceeded"):
        cp.read_region(root, "fields/field.png", "fields/mask.png")
    monkeypatch.setattr(cp, "MAX_FILE_BYTES", 10)
    with pytest.raises(cp.ImageError, match="cellprofiler_image_budget_exceeded"):
        cp.check_region_files(root, "fields/field.png", "fields/mask.png")


def test_single_file_cannot_be_registered_as_both_image_and_mask(fields):
    root, *_ = fields
    for reader in (cp.read_region, cp.check_region_files):
        with pytest.raises(cp.ImageError, match="cellprofiler_invalid_image_path"):
            reader(root, "fields/field.png", "fields/field.png")


def test_execution_accounts_for_bytes_in_current_snapshot_even_after_reencoding(fields):
    from PIL import PngImagePlugin

    root, images, rgb, _ = fields
    args = (root, "fields/field.png", "fields/mask.png")
    before = cp.check_region_files(*args)
    assert before == cp.read_region(*args).source_byte_count
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Description", "Synthetic field exported again")
    Image.fromarray(rgb).save(images / "field.png", pnginfo=metadata)
    expected = sum((images / name).stat().st_size for name in ("field.png", "mask.png"))
    assert expected > before
    assert cp.read_region(*args).source_byte_count == expected
