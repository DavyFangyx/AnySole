"""V8_Cv7_PhysIMU + frozen V gate + window-level V/VT decision gate.

The decision gate outputs s (one logit per window); p = sigmoid(s) is the
probability of selecting the complete VT output over the complete V output.
It is trained by the gain-weighted decision loss |E_VT - E_V| * BCE(s, y),
y = 1[E_VT < E_V], jointly with the T-side representation.
"""
from __future__ import annotations

from anysole.models.v8_cv7_physimu_gate import STRUCTURE as GATE_STRUCTURE
from anysole.models.v8_cv7_physimu_gate import build as gate_build

STRUCTURE = {**GATE_STRUCTURE, "v7_decision_gate": True,
             "v7_decision_p0": 0.8}
CONFIG_EXTRA: dict = {"lambda_tanchor": 0.0}


def build(config: dict, part_joints=None):
    return gate_build({**config, **STRUCTURE}, part_joints)
