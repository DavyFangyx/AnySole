"""Explicit V5 training input availability, independent of measured signal values.

legacy remains the default. availability_v1 is a new training recipe, never a
retroactive label for old checkpoints. V/T/VT are native readouts of one model.
"""
from pathlib import Path
import torch
from anysole.types import CONFIG_V, CONFIG_T, CONFIG_VT

POLICIES = ('legacy', 'availability_v1')

def conditioning_policy(config):
    value = config.get('v5_conditioning', 'legacy')
    if value not in POLICIES:
        raise ValueError('Unknown v5_conditioning: %r' % value)
    return value

def require_fresh_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError('availability_v1 requires a fresh --out-dir; preserve old outputs: %s' % path)

def _flag(value, config_id, name):
    if value is None:
        return torch.zeros_like(config_id, dtype=torch.bool)
    value = torch.as_tensor(value, device=config_id.device)
    if value.shape != config_id.shape or not bool(((value == 0) | (value == 1)).all()):
        raise ValueError('%s must contain one boolean/0/1 per window' % name)
    return value.bool()

def availability_masks(config_id, batch=None, random_v_drop=None):
    if config_id.ndim != 1 or not bool(((config_id == CONFIG_V) | (config_id == CONFIG_T) | (config_id == CONFIG_VT)).all()):
        raise ValueError('config_id must be a vector of V/T/VT conditions')
    batch = {} if batch is None else batch
    v = config_id != CONFIG_T
    t = config_id != CONFIG_V
    for key in ('v_zero', 'occ_v_zeroed'):
        v = v & ~_flag(batch.get(key), config_id, key)
    for key in ('t_zero', 'occ_t_zeroed'):
        t = t & ~_flag(batch.get(key), config_id, key)
    v = v & ~_flag(random_v_drop, config_id, 'random_v_drop')
    # Explicit masks may remove additional observations, never restore a
    # modality that config_id or a structured dropout declared absent.
    if 'v_available' in batch:
        v = v & _flag(batch['v_available'], config_id, 'v_available')
    if 't_available' in batch:
        t = t & _flag(batch['t_available'], config_id, 't_available')
    return v, t

def apply_availability(inputs, config_id, batch, random_v_drop):
    """Use the same masks to gate inputs and supervise branches. Preserve GT.
    Zero-valued but observed pressure remains available (e.g. genuine flight).
    """
    v, t = availability_masks(config_id, batch, random_v_drop)
    values=[]
    for x, present in zip(inputs, (v, t, t, t)):
        if x.shape[0] != len(config_id):
            raise ValueError('Input batch does not match availability masks')
        shape=(len(config_id),)+(1,)*(x.ndim-1)
        values.append(torch.where(present.view(shape), x, torch.zeros_like(x)))
    masks={'v_available':v, 't_available':t, 'v_zero':~v, 't_zero':~t}
    return tuple(values), masks
