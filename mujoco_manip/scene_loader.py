"""Shared scene loading for MuJoCo environments."""

import os
import re
import tempfile

import mujoco

from mujoco_manip.data import PANDA_DIR as _DEFAULT_PANDA_DIR


def load_scene(
    xml_path: str,
    panda_dir: str = _DEFAULT_PANDA_DIR,
    add_wrist_camera: bool = False,
) -> mujoco.MjModel:
    """Load a scene XML, resolving robot meshes from *panda_dir*.

    MuJoCo resolves ``<include>`` and ``meshdir`` relative to the loading
    file's directory. We write a temp copy of the scene XML into the panda
    directory so that ``panda.xml``'s internal ``meshdir="assets"`` resolves
    correctly.

    Args:
        xml_path: Path to the scene XML file.
        panda_dir: Directory containing ``panda.xml``.
        add_wrist_camera: If True, inject a wrist camera on the hand body
            via MjSpec.

    Returns:
        Compiled MuJoCo model.
    """
    with open(xml_path, encoding="utf-8") as f:
        xml = f.read()

    # Normalize any include path ending in panda.xml to just "panda.xml"
    xml = re.sub(r'file="[^"]*panda\.xml"', 'file="panda.xml"', xml)
    # Remove duplicate compiler directives (panda.xml has its own)
    xml = re.sub(r'<compiler\s+angle="radian"\s*/>\s*', "", xml)

    abs_panda_dir = os.path.abspath(panda_dir)
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=abs_panda_dir)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(xml)

        if add_wrist_camera:
            spec = mujoco.MjSpec.from_file(tmp_path)
            hand = spec.body("hand")
            cam = hand.add_camera()
            cam.name = "wrist"
            cam.pos = [-0.07, 0.0, 0.055]
            cam.quat = [
                -0.0616,
                -0.7044,
                0.7044,
                0.0616,
            ]
            cam.fovy = 128.0
            return spec.compile()
        else:
            return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
