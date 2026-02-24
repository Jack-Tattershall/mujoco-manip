"""Tests for SE(3) pose utilities."""

import numpy as np
import pytest

from src.pose_utils import (
    pos_rotmat_to_se3,
    quat_xyzw_to_rotmat,
    rotmat_from_6d,
    rotmat_to_6d,
    rotmat_to_quat_xyzw,
    se3_from_8dof,
    se3_from_10dof,
    se3_to_8dof,
    se3_to_10dof,
    se3_to_pos_rotmat,
)


def _random_rotation() -> np.ndarray:
    """Generate a random valid rotation matrix via QR decomposition."""
    H = np.random.randn(3, 3)
    Q, R = np.linalg.qr(H)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _random_se3() -> np.ndarray:
    """Generate a random SE(3) matrix."""
    return pos_rotmat_to_se3(np.random.randn(3), _random_rotation())


# ---------------------------------------------------------------------------
# pos_rotmat_to_se3 / se3_to_pos_rotmat
# ---------------------------------------------------------------------------


class TestSE3Construction:
    def test_roundtrip(self):
        pos = np.array([1.0, 2.0, 3.0])
        R = _random_rotation()
        T = pos_rotmat_to_se3(pos, R)
        pos_out, R_out = se3_to_pos_rotmat(T)
        np.testing.assert_allclose(pos_out, pos)
        np.testing.assert_allclose(R_out, R)

    def test_bottom_row(self):
        T = pos_rotmat_to_se3(np.zeros(3), np.eye(3))
        np.testing.assert_array_equal(T[3, :], [0, 0, 0, 1])

    def test_identity(self):
        T = pos_rotmat_to_se3(np.zeros(3), np.eye(3))
        np.testing.assert_allclose(T, np.eye(4))


# ---------------------------------------------------------------------------
# Quaternion ↔ rotation matrix
# ---------------------------------------------------------------------------


class TestQuaternion:
    def test_identity_quaternion(self):
        R = quat_xyzw_to_rotmat(np.array([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_roundtrip_identity(self):
        q = rotmat_to_quat_xyzw(np.eye(3))
        R = quat_xyzw_to_rotmat(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("_seed", range(10))
    def test_roundtrip_random(self, _seed: int):
        np.random.seed(_seed)
        R = _random_rotation()
        q = rotmat_to_quat_xyzw(R)
        R_recovered = quat_xyzw_to_rotmat(q)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_unit_norm(self):
        R = _random_rotation()
        q = rotmat_to_quat_xyzw(R)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)

    def test_90deg_rotation_around_z(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        q = rotmat_to_quat_xyzw(R)
        R_back = quat_xyzw_to_rotmat(q)
        np.testing.assert_allclose(R_back, R, atol=1e-12)


# ---------------------------------------------------------------------------
# 6D rotation (Zhou et al.)
# ---------------------------------------------------------------------------


class TestRotation6D:
    def test_roundtrip_identity(self):
        d6 = rotmat_to_6d(np.eye(3))
        R = rotmat_from_6d(d6)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("_seed", range(10))
    def test_roundtrip_random(self, _seed: int):
        np.random.seed(_seed)
        R = _random_rotation()
        d6 = rotmat_to_6d(R)
        R_recovered = rotmat_from_6d(d6)
        np.testing.assert_allclose(R_recovered, R, atol=1e-6)

    def test_output_shape(self):
        d6 = rotmat_to_6d(np.eye(3))
        assert d6.shape == (6,)
        assert d6.dtype == np.float32

    def test_recovered_is_orthogonal(self):
        np.random.seed(7)
        R = rotmat_from_6d(rotmat_to_6d(_random_rotation()))
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 8DOF ↔ SE(3)
# ---------------------------------------------------------------------------


class Test8DOF:
    def test_identity(self):
        dof8 = np.array([0, 0, 0, 0, 0, 0, 1, 0.5], dtype=np.float32)
        T = se3_from_8dof(dof8)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-6)

    @pytest.mark.parametrize("gripper", [0.0, 0.5, 1.0])
    def test_roundtrip(self, gripper: float):
        np.random.seed(42)
        T = _random_se3()
        dof8 = se3_to_8dof(T, gripper)
        T_recovered = se3_from_8dof(dof8)
        np.testing.assert_allclose(T_recovered, T, atol=1e-5)
        assert dof8[-1] == pytest.approx(gripper)

    def test_output_shape_and_dtype(self):
        T = np.eye(4)
        dof8 = se3_to_8dof(T, 1.0)
        assert dof8.shape == (8,)
        assert dof8.dtype == np.float32

    def test_position_preserved(self):
        pos = np.array([1.5, -2.0, 0.3])
        T = pos_rotmat_to_se3(pos, np.eye(3))
        dof8 = se3_to_8dof(T, 0.0)
        np.testing.assert_allclose(dof8[:3], pos, atol=1e-6)


# ---------------------------------------------------------------------------
# 10DOF ↔ SE(3)
# ---------------------------------------------------------------------------


class Test10DOF:
    def test_identity(self):
        dof10 = np.zeros(10, dtype=np.float32)
        dof10[3] = 1.0  # r11
        dof10[7] = 1.0  # r22
        T = se3_from_10dof(dof10)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-6)

    @pytest.mark.parametrize("gripper", [0.0, 0.5, 1.0])
    def test_roundtrip(self, gripper: float):
        np.random.seed(99)
        T = _random_se3()
        dof10 = se3_to_10dof(T, gripper)
        T_recovered = se3_from_10dof(dof10)
        np.testing.assert_allclose(T_recovered, T, atol=1e-5)
        assert dof10[-1] == pytest.approx(gripper)

    def test_output_shape_and_dtype(self):
        T = np.eye(4)
        dof10 = se3_to_10dof(T, 1.0)
        assert dof10.shape == (10,)
        assert dof10.dtype == np.float32

    def test_position_preserved(self):
        pos = np.array([-0.5, 0.4, 1.2])
        T = pos_rotmat_to_se3(pos, np.eye(3))
        dof10 = se3_to_10dof(T, 0.0)
        np.testing.assert_allclose(dof10[:3], pos, atol=1e-6)


# ---------------------------------------------------------------------------
# 8DOF ↔ 10DOF cross-consistency
# ---------------------------------------------------------------------------


class TestCrossConsistency:
    @pytest.mark.parametrize("_seed", range(5))
    def test_same_se3_different_encoding(self, _seed: int):
        """8DOF and 10DOF encoding of the same SE(3) should reconstruct identically."""
        np.random.seed(_seed)
        T = _random_se3()
        T_from_8 = se3_from_8dof(se3_to_8dof(T, 0.5))
        T_from_10 = se3_from_10dof(se3_to_10dof(T, 0.5))
        np.testing.assert_allclose(T_from_8, T_from_10, atol=1e-5)


# ---------------------------------------------------------------------------
# Relative action reconstruction (the exact pattern used in replay/visualize)
# ---------------------------------------------------------------------------


class TestRelativeReconstruction:
    @pytest.mark.parametrize("_seed", range(5))
    def test_8dof_relative_roundtrip(self, _seed: int):
        """T_init @ se3_from_8dof(se3_to_8dof(inv(T_init) @ T_target)) ≈ T_target."""
        np.random.seed(_seed)
        T_init = _random_se3()
        T_target = _random_se3()
        T_rel = np.linalg.inv(T_init) @ T_target
        dof8_rel = se3_to_8dof(T_rel, 0.7)
        T_reconstructed = T_init @ se3_from_8dof(dof8_rel)
        np.testing.assert_allclose(T_reconstructed, T_target, atol=1e-4)

    @pytest.mark.parametrize("_seed", range(5))
    def test_10dof_relative_roundtrip(self, _seed: int):
        """T_init @ se3_from_10dof(se3_to_10dof(inv(T_init) @ T_target)) ≈ T_target."""
        np.random.seed(_seed)
        T_init = _random_se3()
        T_target = _random_se3()
        T_rel = np.linalg.inv(T_init) @ T_target
        dof10_rel = se3_to_10dof(T_rel, 0.3)
        T_reconstructed = T_init @ se3_from_10dof(dof10_rel)
        np.testing.assert_allclose(T_reconstructed, T_target, atol=1e-4)

    def test_position_matches_after_relative_reconstruction(self):
        """The XYZ from reconstructed T should match the original target position."""
        np.random.seed(123)
        T_init = _random_se3()
        target_pos = np.array([-0.15, 0.45, 0.36])
        T_target = pos_rotmat_to_se3(target_pos, _random_rotation())
        T_rel = np.linalg.inv(T_init) @ T_target
        dof8_rel = se3_to_8dof(T_rel, 1.0)
        T_abs = T_init @ se3_from_8dof(dof8_rel)
        np.testing.assert_allclose(T_abs[:3, 3], target_pos, atol=1e-5)
