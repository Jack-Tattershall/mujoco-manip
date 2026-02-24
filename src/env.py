"""MuJoCo environment wrapper for the pick-and-place scene."""

import os
import tempfile

import mujoco
import mujoco.viewer
import numpy as np

# Placeholder in the scene XML for the panda include path
_PANDA_INCLUDE = 'file="third_party/mujoco_menagerie/franka_emika_panda/panda.xml"'


def _load_scene(
    xml_path: str, panda_dir: str, add_wrist_camera: bool = False
) -> mujoco.MjModel:
    """Load the scene XML from the panda model directory.

    MuJoCo resolves <include> and meshdir relative to the loading file's
    directory. We write a temp copy of the scene XML into the panda directory
    so that panda.xml's internal meshdir="assets" resolves correctly.

    If add_wrist_camera is True, uses MjSpec to inject a camera on the hand body.
    """
    with open(xml_path) as f:
        xml = f.read()

    # Rewrite include to local panda.xml (will be in the same directory)
    xml = xml.replace(_PANDA_INCLUDE, 'file="panda.xml"')
    # Remove our compiler tag (let panda.xml's compiler handle meshdir)
    xml = xml.replace('<compiler angle="radian"/>\n\n', "")

    # Write into panda directory and load from there
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
            cam.pos = [0.0, 0.0, 0.04]
            cam.quat = [
                0.0,
                0.0,
                1.0,
                0.0,
            ]  # 180° around y so camera looks along hand +z (downward)
            return spec.compile()
        else:
            return mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)


class PickPlaceEnv:
    """Loads the MJCF scene, manages simulation stepping and viewer."""

    def __init__(
        self,
        xml_path: str,
        panda_dir: str | None = None,
        add_wrist_camera: bool = False,
    ):
        if panda_dir is None:
            project_root = os.path.dirname(os.path.abspath(xml_path))
            panda_dir = os.path.join(
                project_root, "third_party", "mujoco_menagerie", "franka_emika_panda"
            )
        self.model = _load_scene(xml_path, panda_dir, add_wrist_camera=add_wrist_camera)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

    def launch_viewer(self):
        """Open the passive viewer window."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset_to_keyframe(self, name: str = "scene_start"):
        """Reset simulation state to a named keyframe."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise ValueError(f"Keyframe '{name}' not found")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def sync(self):
        """Sync viewer with current simulation state."""
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self) -> bool:
        """Check if the viewer window is still open."""
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    def get_body_pos(self, name: str) -> np.ndarray:
        """Get the 3D position of a named body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found")
        return self.data.xpos[body_id].copy()

    def get_body_xmat(self, name: str) -> np.ndarray:
        """Get the 3x3 rotation matrix of a named body."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found")
        return self.data.xmat[body_id].reshape(3, 3).copy()
