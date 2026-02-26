"""MuJoCo environment wrapper for the pick-and-place scene."""

import os
import tempfile

import mujoco
import mujoco.viewer
import numpy as np

from .data import PANDA_DIR as _DEFAULT_PANDA_DIR
from .data import SCENE_XML as _DEFAULT_SCENE_XML
from .randomization import randomize_object_positions


def _load_scene(
    xml_path: str, panda_dir: str, add_wrist_camera: bool = False
) -> mujoco.MjModel:
    """Load the scene XML, resolving robot meshes from *panda_dir*.

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

    # If the scene still references the old third_party path, fix it.
    xml = xml.replace(
        'file="third_party/mujoco_menagerie/franka_emika_panda/panda.xml"',
        'file="panda.xml"',
    )
    # Also handle the bundled data/ layout.
    xml = xml.replace('file="franka_emika_panda/panda.xml"', 'file="panda.xml"')
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


class PickPlaceEnv:
    """Loads the MJCF scene, manages simulation stepping and viewer."""

    def __init__(
        self,
        xml_path: str | None = None,
        panda_dir: str | None = None,
        add_wrist_camera: bool = False,
    ) -> None:
        """Initialise the environment.

        Args:
            xml_path: Path to the MuJoCo scene XML. Defaults to the bundled
                scene included with the package.
            panda_dir: Directory containing ``panda.xml``. Defaults to the
                bundled Franka Panda model included with the package.
            add_wrist_camera: If True, inject a wrist camera on the hand body.
        """
        if xml_path is None:
            xml_path = _DEFAULT_SCENE_XML
        if panda_dir is None:
            panda_dir = _DEFAULT_PANDA_DIR
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

    def randomize_objects(
        self, rng: np.random.Generator, **kwargs
    ) -> dict[str, np.ndarray]:
        """Randomize object positions and propagate to xpos.

        Args:
            rng: NumPy random generator for reproducibility.
            **kwargs: Forwarded to ``randomize_object_positions``.

        Returns:
            Dict mapping joint name to new (x, y, z) position.
        """
        result = randomize_object_positions(self.model, self.data, rng, **kwargs)
        mujoco.mj_forward(self.model, self.data)
        return result

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
