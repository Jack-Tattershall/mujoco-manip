"""Bundled data files for mujoco-manip."""

from pathlib import Path

DATA_DIR = Path(__file__).parent
SCENE_XML = str(DATA_DIR / "pick_and_place_scene.xml")
PANDA_DIR = str(DATA_DIR / "franka_emika_panda")
