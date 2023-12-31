import threestudio
from packaging.version import Version

if hasattr(threestudio, "__version__") and Version(threestudio.__version__) >= Version(
    "0.2.2"
):
    pass
else:
    if hasattr(threestudio, "__version__"):
        print(f"[INFO] threestudio version: {threestudio.__version__}")
    raise ValueError(
        "threestudio version must be >= 0.2.2, please update threestudio by pulling the latest version from github"
    )

from .guidance import (
    diffusers_deep_floyd_guidance,
    diffusers_guidance,
    diffusers_lcm_xl_guidance,
    diffusers_stable_diffusion_guidance,
    diffusers_stable_diffusion_xl_guidance,
)
from .sampler import ism_sampler, lcm_sampler, sds_sampler
