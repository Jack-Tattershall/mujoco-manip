"""Tests for bottle packing constants and well geometry."""

import numpy as np
import pytest

from mujoco_manip.tasks.bottle_packing.constants import (
    CRATE_POS,
    NUM_WELLS,
    WELL_COLS,
    WELL_ROWS,
    WELL_SPACING,
    well_position,
    well_row_col,
)


class TestWellGrid:
    def test_num_wells(self):
        assert NUM_WELLS == WELL_ROWS * WELL_COLS
        assert NUM_WELLS == 20

    def test_well_row_col_range(self):
        for i in range(NUM_WELLS):
            row, col = well_row_col(i)
            assert 0 <= row < WELL_ROWS
            assert 0 <= col < WELL_COLS

    def test_well_row_col_unique(self):
        coords = [well_row_col(i) for i in range(NUM_WELLS)]
        assert len(set(coords)) == NUM_WELLS

    def test_well_row_col_covers_grid(self):
        rows = set()
        cols = set()
        for i in range(NUM_WELLS):
            r, c = well_row_col(i)
            rows.add(r)
            cols.add(c)
        assert rows == set(range(WELL_ROWS))
        assert cols == set(range(WELL_COLS))


class TestWellPosition:
    def test_shape(self):
        pos = well_position(0)
        assert pos.shape == (3,)

    def test_z_is_crate_floor(self):
        for i in range(NUM_WELLS):
            pos = well_position(i)
            assert pos[2] == pytest.approx(CRATE_POS[2])

    def test_spacing_between_adjacent_cols(self):
        """Wells in the same row, adjacent columns should be WELL_SPACING apart in X."""
        p0 = well_position(0)  # row 0, col 0
        p1 = well_position(1)  # row 0, col 1
        assert abs(p1[0] - p0[0]) == pytest.approx(WELL_SPACING, abs=1e-6)
        assert abs(p1[1] - p0[1]) == pytest.approx(0.0, abs=1e-6)

    def test_spacing_between_adjacent_rows(self):
        """Wells in the same col, adjacent rows should be WELL_SPACING apart in Y."""
        p0 = well_position(0)  # row 0, col 0
        p5 = well_position(5)  # row 1, col 0
        assert abs(p5[1] - p0[1]) == pytest.approx(WELL_SPACING, abs=1e-6)
        assert abs(p5[0] - p0[0]) == pytest.approx(0.0, abs=1e-6)

    def test_centered_on_crate(self):
        """The mean well position XY should be the crate center XY."""
        positions = np.array([well_position(i) for i in range(NUM_WELLS)])
        mean_xy = positions[:, :2].mean(axis=0)
        np.testing.assert_allclose(mean_xy, CRATE_POS[:2], atol=1e-6)

    def test_all_wells_unique(self):
        positions = [tuple(well_position(i)) for i in range(NUM_WELLS)]
        assert len(set(positions)) == NUM_WELLS
