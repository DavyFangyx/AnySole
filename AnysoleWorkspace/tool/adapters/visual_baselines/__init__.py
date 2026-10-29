"""Export package-saved WHAM / SAM3DB / GVHMR predictions to the unified contract.

Registered in Baselines/决策/11_20261029_视觉基线集成.md.  The chain mirrors
``AnysoleWorkspace/tool/eval_rgb_smpl.py`` (the validated anysole_aligned
geometry) and extends it with the checkerboard -> mocap-world inverse and the
project protocol FK, so archives land on the canonical 40 Hz frame grid.
"""
