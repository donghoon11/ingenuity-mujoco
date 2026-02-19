"""
Ingenuity Mars Helicopter - Shared Utilities

Re-exports functions from hover_basic.py and provides new helpers
for the blade optimization testbed.
"""

import sys
import csv
import json
import numpy as np
import mujoco
from pathlib import Path

from config import DESIGN_DIR, CODE_0219, SCENE_MARS_XML, RESULTS_DIR, INIT_POS_Z

# ============================================================================
# Import from existing codebase (hover_basic.py)
# ============================================================================
_0219_str = str(CODE_0219)
if _0219_str not in sys.path:
    sys.path.insert(0, _0219_str)

from hover_basic import (
    quat_to_euler,
    get_sensor_data,
    get_rotor_joint_indices,
    update_rotor_visual,
    update_tracking_camera,
    apply_gust,
)

# Re-export for convenience
__all__ = [
    'quat_to_euler', 'get_sensor_data', 'get_rotor_joint_indices',
    'update_rotor_visual', 'update_tracking_camera', 'apply_gust',
    'load_mars_model', 'get_body_id', 'extract_state', 'log_to_csv',
    'ensure_results_dir', 'save_json', 'load_json',
]


# ============================================================================
# New Helpers
# ============================================================================

def load_mars_model(xml_path: str = None) -> tuple:
    """
    Load the Mars-patched MuJoCo model.

    Returns (model, data) tuple.
    """
    if xml_path is None:
        xml_path = SCENE_MARS_XML
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return model, data


def get_body_id(model, name: str) -> int:
    """Get body ID by name."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def extract_state(data) -> dict:
    """
    Extract full state from MuJoCo data for freejoint body.

    Returns dict with:
        'pos'   : [x, y, z]
        'quat'  : [w, x, y, z]
        'vel'   : [vx, vy, vz]
        'omega' : [wx, wy, wz]
        'euler' : [roll, pitch, yaw]
    """
    pos = data.qpos[:3].copy()
    quat = data.qpos[3:7].copy()
    vel = data.qvel[:3].copy()
    omega = data.qvel[3:6].copy()

    w, qx, qy, qz = quat
    roll, pitch, yaw = quat_to_euler(w, qx, qy, qz)

    return {
        'pos': pos,
        'quat': quat,
        'vel': vel,
        'omega': omega,
        'euler': np.array([roll, pitch, yaw]),
    }


def reset_model(model, data, z_init: float = None):
    """Reset MuJoCo data to initial state."""
    mujoco.mj_resetData(model, data)
    if z_init is not None:
        data.qpos[2] = z_init
    else:
        data.qpos[2] = INIT_POS_Z


def ensure_results_dir(experiment_name: str) -> Path:
    """Create results subdirectory if it doesn't exist."""
    d = RESULTS_DIR / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_to_csv(filepath: str, headers: list, rows: list):
    """Write data to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def save_json(filepath: str, data: dict):
    """Save dict to JSON file."""
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, cls=NpEncoder, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    """Load dict from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
