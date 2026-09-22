"""Utility modules moved out of the anysole top level (2026-09-22 repo
cleanup): geometry (FK / 6D / heading), losses (training objectives),
diffusion (legacy V1 sampling), eval_protocol (protocol metrics).

eval_protocol is intentionally NOT imported here: it imports anysole.eval
at module load (mutual dependency that works when eval.py is the entry
point), and an eager re-export from this package would invert that order.
"""

from anysole.utils.diffusion import GaussianDiffusion  # noqa: F401
from anysole.utils.geometry import (  # noqa: F401
    f2_to_world,
    fk_pose6d,
    heading_from_root_np,
    rot6d_to_rotmat,
    rotmat_to_6d,
)
from anysole.utils.losses import compute_losses, soft_contact_from_keypoints  # noqa: F401
