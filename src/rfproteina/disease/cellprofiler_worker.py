"""Private H-DAB measurement worker for the official CellProfiler 4.2.8 image."""

import hashlib
import json
import struct
import sys
from contextlib import closing, redirect_stdout
from importlib.metadata import version
from typing import BinaryIO


def measure(stream: BinaryIO) -> dict:
    import numpy as np
    from cellprofiler_core.preferences import set_headless

    set_headless()
    from cellprofiler.modules.measureimageintensity import (
        F_MEAN_INTENSITY,
        F_MEDIAN_INTENSITY,
        F_TOTAL_AREA,
        MeasureImageIntensity,
    )
    from cellprofiler.modules.unmixcolors import CHOICE_DAB, CHOICE_HEMATOXYLIN, UnmixColors
    from cellprofiler_core.image import Image, ImageSetList
    from cellprofiler_core.measurement import Measurements
    from cellprofiler_core.object import ObjectSet
    from cellprofiler_core.pipeline import Pipeline
    from cellprofiler_core.workspace import Workspace

    pipeline = Pipeline()
    unmix = UnmixColors()
    unmix.set_module_num(1)
    unmix.input_image_name.value = "Brightfield"
    unmix.add_image()
    for index, dye in enumerate((CHOICE_HEMATOXYLIN, CHOICE_DAB)):
        output = unmix.outputs[index]
        output.image_name.value = dye
        output.stain_choice.value = dye
    intensity = MeasureImageIntensity()
    intensity.set_module_num(2)
    intensity.images_list.value = CHOICE_DAB
    pipeline.add_module(unmix)
    pipeline.add_module(intensity)

    (count,) = struct.unpack(">I", stream.read(4))
    if not 1 <= count <= 4:
        raise ValueError("invalid batch size")
    results = []
    for _ in range(count):
        header = stream.read(8)
        height, width = struct.unpack(">II", header)
        size = height * width
        if not 1 <= size <= 4_194_304:
            raise ValueError("invalid field size")
        rgb_bytes = stream.read(3 * size)
        mask_bytes = stream.read(size)
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
        mask_values = np.frombuffer(mask_bytes, dtype=np.uint8).reshape((height, width))
        if not np.any(mask_values) or np.max(mask_values) != 1:
            raise ValueError("invalid region mask")
        mask = mask_values.view(bool)
        digest = hashlib.sha256(header)
        digest.update(rgb_bytes)
        digest.update(mask_bytes)
        image_sets = ImageSetList()
        image_set = image_sets.get_image_set(0)
        image_set.add("Brightfield", Image(rgb, mask=mask))
        objects = ObjectSet()
        with closing(Measurements(mode="memory")) as measurements:
            for module in (unmix, intensity):
                module.run(Workspace(pipeline, module, image_set, objects, measurements, image_sets))
            # UnmixColors deconvolves the RGB field. The child image inherits the
            # explicit mask and MeasureImageIntensity selects only its ROI pixels.
            results.append(
                {
                    "region_sha256": digest.hexdigest(),
                    "roi_pixels": int(measurements.get_current_image_measurement(F_TOTAL_AREA % CHOICE_DAB)),
                    "mean_dab": float(measurements.get_current_image_measurement(F_MEAN_INTENSITY % CHOICE_DAB)),
                    "median_dab": float(measurements.get_current_image_measurement(F_MEDIAN_INTENSITY % CHOICE_DAB)),
                }
            )
    if stream.read(1):
        raise ValueError("unexpected trailing fields")
    return {"version": version("cellprofiler"), "regions": results}


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        reply = measure(sys.stdin.buffer)
    print(json.dumps(reply, allow_nan=False))
