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
    with open(xml_path) as f:
        xml = f.read()

    xml = xml.replace(_PANDA_INCLUDE, 'file="panda.xml"')
    xml = xml.replace('<compiler angle="radian"/>\n\n', "")

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
    ) -> None:
        """Initialise the environment.

        Args:
            xml_path: Path to the MuJoCo scene XML.
            panda_dir: Directory containing ``panda.xml``. Defaults to
                ``third_party/mujoco_menagerie/franka_emika_panda`` relative
                to *xml_path*.
            add_wrist_camera: If True, inject a wrist camera on the hand body.
        """
        if panda_dir is None:
            project_root = os.path.dirname(os.path.abspath(xml_path))
            panda_dir = os.path.join(
                project_root, "third_party", "mujoco_menagerie", "franka_emika_panda"
            )
        self.model: mujoco.MjModel = _load_scene(
            xml_path, panda_dir, add_wrist_camera=add_wrist_camera
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        self.viewer = None

    def launch_viewer(self) -> None:
        """Open the passive MuJoCo viewer window."""
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset_to_keyframe(self, name: str = "scene_start") -> None:
        """Reset simulation state to a named keyframe.

        Args:
            name: Keyframe name defined in the MJCF.

        Raises:
            ValueError: If the keyframe is not found.
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise ValueError(f"Keyframe '{name}' not found")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        """Advance simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def sync(self) -> None:
        """Sync viewer with current simulation state."""
        if self.viewer is not None:
            self.viewer.sync()

    def is_running(self) -> bool:
        """Return True if the viewer is open (or no viewer is attached)."""
        if self.viewer is None:
            return True
        return self.viewer.is_running()

    def get_body_pos(self, name: str) -> np.ndarray:
        """Return world position (3,) of a named body.

        Args:
            name: MuJoCo body name.

        Raises:
            ValueError: If the body is not found.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found")
        return self.data.xpos[body_id].copy()

    def get_body_xmat(self, name: str) -> np.ndarray:
        """Return rotation matrix (3, 3) of a named body.

        Args:
            name: MuJoCo body name.

        Raises:
            ValueError: If the body is not found.
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found")
        return self.data.xmat[body_id].reshape(3, 3).copy()
