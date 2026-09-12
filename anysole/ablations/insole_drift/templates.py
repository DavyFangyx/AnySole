from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def load_template_bank(path):
    data = json.loads(Path(path).read_text())
    return np.asarray(data["templates"], dtype=np.float32), data["subject_to_index"]
