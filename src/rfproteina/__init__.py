import logging
import os

from environs import Env

# Load settings before AtomWorks reads its environment during import.
from rfproteina.utils.env import load_dotenv

load_dotenv()

_env = Env()
_env.read_env()
should_debug = _env.bool("DEBUG", default=False)
# Read by utils/torch.py to gate its NaN assertions.
should_check_nans = _env.bool("NAN_CHECK", default=True)

logger = logging.getLogger("rfproteina")
logger.setLevel(logging.DEBUG if should_debug else logging.INFO)

# cuEquivariance tuning defaults. These two variables are read by cuequivariance itself
# when a fused kernel runs, and by nothing in this package, so putting them in the
# environment is all that has to happen here.
#
# Importing torch to ask whether CUDA is present, and importing cuequivariance_torch to
# find out whether it is installed, cost about forty seconds together and made every
# module in this package wait for them -- including modules that touch neither, because
# importing any submodule runs this file first. Neither answer was kept: the fused-kernel
# switch is decided independently by nn/modules/triangle.py,
# nn/modules/openfold_modules/triangular_multiplicative_update.py and
# nn/modules/pair_bias_attn.py, each of which reads DISABLE_CUEQUIVARIANCE and imports the
# library itself, and each of which logs what it actually chose. Without CUDA, or with the
# library absent, these variables are inert rather than wrong: nothing reads them unless a
# kernel is running, which requires both.
if not _env.bool("DISABLE_CUEQUIVARIANCE", default=False):
    os.environ["CUEQ_DISABLE_AOT_TUNING"] = _env.str("CUEQ_DISABLE_AOT_TUNING", default="1")
    os.environ["CUEQ_DEFAULT_CONFIG"] = _env.str("CUEQ_DEFAULT_CONFIG", default="1")

# Both are namespace packages with no module body, so these cost nothing and exist to
# make the names in __all__ resolvable after a bare ``import rfproteina``. The annotation
# registration that used to sit beside them is now imported by the modules that read the
# registry, each of which already has atomworks loaded.
import rfproteina.datapipes  # noqa: E402
import rfproteina.utils  # noqa: E402

__all__ = ["datapipes", "utils"]
