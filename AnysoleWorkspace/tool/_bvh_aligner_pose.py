"""BVH pose conversion shared by the display renderers.

Re-export of the canonical ``Baselines/utils/bvh_aligner.py`` (which also
handles the 2026-10-24 BVH24 export); this module keeps its historical name
so tool-layer consumers (contact_labels.py, contact_adapter.py) import
unchanged.
"""
from Baselines.utils.bvh_aligner import parse_bvh_aligner  # noqa: F401
