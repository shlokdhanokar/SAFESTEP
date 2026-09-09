"""Preview overlay rendering.

The overlay is a debug aid the end user never sees, so these tests cover
mechanics rather than aesthetics -- with one exception. Caption placement is
tested properly because a real bug shipped here: a close obstacle fills the
frame, so its box touches y=0 and its caption was drawn straight over the HUD,
leaving both unreadable in exactly the situation that matters most.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_detection

from safestep.annotate import (
    HUD_RESERVED_PX,
    draw_detections,
    draw_hud,
    draw_zone_guides,
)
from safestep.spatial import Proximity, Zone


def bright_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """A white frame: any drawing shows up as a darkening."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


class TestDrawDetections:
    def test_does_not_mutate_the_input_frame(self, blank_frame):
        # The original code drew onto the caller's array while claiming to
        # return a new one, corrupting any downstream use of the raw frame.
        original = blank_frame.copy()
        draw_detections(blank_frame, [make_detection()])
        assert np.array_equal(blank_frame, original)

    def test_returns_a_frame_of_the_same_shape(self, blank_frame):
        canvas = draw_detections(blank_frame, [make_detection()])
        assert canvas.shape == blank_frame.shape

    def test_empty_detection_list_leaves_the_frame_unchanged(self, blank_frame):
        assert np.array_equal(draw_detections(blank_frame, []), blank_frame)

    def test_something_is_actually_drawn(self, blank_frame):
        canvas = draw_detections(blank_frame, [make_detection(x=100, y=200, w=80, h=120)])
        assert not np.array_equal(canvas, blank_frame)

    def test_urgency_changes_the_box_colour(self, blank_frame):
        near = draw_detections(blank_frame, [make_detection(proximity=Proximity.NEAR)])
        immediate = draw_detections(
            blank_frame, [make_detection(proximity=Proximity.IMMEDIATE)]
        )
        assert not np.array_equal(near, immediate)

    def test_handles_a_box_flush_against_every_edge(self, blank_frame):
        # Must not raise or write out of bounds.
        for box in (
            make_detection(x=0, y=0, w=50, h=50),
            make_detection(x=590, y=0, w=50, h=50),
            make_detection(x=0, y=430, w=50, h=50),
            make_detection(x=590, y=430, w=50, h=50),
        ):
            assert draw_detections(blank_frame, [box]).shape == blank_frame.shape

    def test_handles_a_full_frame_box(self, blank_frame):
        full = make_detection(x=0, y=0, w=640, h=480, proximity=Proximity.IMMEDIATE)
        assert draw_detections(blank_frame, [full]).shape == blank_frame.shape

    def test_draws_every_detection(self, blank_frame):
        one = draw_detections(blank_frame, [make_detection(x=50, y=100)])
        two = draw_detections(
            blank_frame, [make_detection(x=50, y=100), make_detection(x=400, y=100)]
        )
        assert not np.array_equal(one, two)


def widest_painted_run(canvas: np.ndarray, row: int) -> int:
    """Longest contiguous run of non-white pixels in ``row``.

    Distinguishes a box border (a few pixels per edge) from a caption's filled
    background (a hundred or more), which is what these tests need to tell
    apart -- a box legitimately draws its top edge at y=0.
    """
    painted = np.any(canvas[row] != 255, axis=-1)
    best = run = 0
    for pixel in painted:
        run = run + 1 if pixel else 0
        best = max(best, run)
    return best


class TestCaptionPlacement:
    #: Box borders are at most a few pixels wide; a caption block is far wider.
    CAPTION_WIDTH_FLOOR = 40

    def _caption_in_hud_band(self, canvas: np.ndarray) -> bool:
        # Skip the topmost rows, where a box's own top border legitimately sits.
        return any(
            widest_painted_run(canvas, row) > self.CAPTION_WIDTH_FLOOR
            for row in range(6, HUD_RESERVED_PX)
        )

    def test_caption_for_a_top_edge_box_stays_out_of_the_hud_band(self):
        # The regression: a box at y=0 put its caption on top of the HUD.
        canvas = draw_detections(
            bright_frame(), [make_detection(x=10, y=0, w=600, h=470)]
        )
        assert not self._caption_in_hud_band(canvas)

    def test_caption_for_a_full_frame_box_stays_out_of_the_hud_band(self):
        canvas = draw_detections(
            bright_frame(),
            [make_detection(x=0, y=0, w=640, h=480, proximity=Proximity.IMMEDIATE)],
        )
        assert not self._caption_in_hud_band(canvas)

    def test_the_caption_is_still_drawn_somewhere(self):
        # Guard against "fixing" the overlap by dropping the caption entirely.
        canvas = draw_detections(
            bright_frame(), [make_detection(x=10, y=0, w=600, h=470)]
        )
        assert any(
            widest_painted_run(canvas, row) > self.CAPTION_WIDTH_FLOOR
            for row in range(HUD_RESERVED_PX, 480)
        )

    def test_caption_for_a_low_box_is_drawn_above_it(self):
        canvas = draw_detections(bright_frame(), [make_detection(x=100, y=300, w=80, h=100)])
        # Region just above the box should have been painted.
        assert not np.all(canvas[270:300, 100:180] == 255)

    def test_caption_never_runs_off_the_right_edge(self):
        canvas = draw_detections(
            bright_frame(), [make_detection(x=635, y=200, w=5, h=50, label="refrigerator")]
        )
        assert canvas.shape == (480, 640, 3)


class TestHud:
    def test_returns_the_same_frame_object(self, blank_frame):
        assert draw_hud(blank_frame, "yolo", 30.0, 2) is blank_frame

    def test_draws_into_the_reserved_band(self):
        frame = bright_frame()
        draw_hud(frame, "yolo", 30.0, 2)
        assert not np.all(frame[:HUD_RESERVED_PX] == 255)

    def test_dimmed_band_keeps_text_legible_over_a_white_scene(self):
        frame = bright_frame()
        draw_hud(frame, "yolo", 30.0, 2)
        # The band is darkened rather than left white.
        assert frame[5, 5].mean() < 200

    def test_last_announcement_adds_a_second_line(self):
        without = bright_frame()
        draw_hud(without, "yolo", 30.0, 1)
        with_alert = bright_frame()
        draw_hud(with_alert, "yolo", 30.0, 1, "person, close, ahead")
        assert not np.array_equal(without, with_alert)

    def test_survives_a_tiny_frame(self):
        tiny = np.full((60, 80, 3), 255, dtype=np.uint8)
        assert draw_hud(tiny, "color", 5.0, 0).shape == (60, 80, 3)


def guide_centres(frame: np.ndarray, row: int = 300):
    """Centre column of each vertical guide in ``row``.

    Antialiasing spreads a nominally 1 px line across two columns, so adjacent
    marked columns are grouped into a single guide before measuring.
    """
    marked = np.where(np.any(frame[row] != 255, axis=-1))[0]
    if len(marked) == 0:
        return []

    groups, current = [], [marked[0]]
    for column in marked[1:]:
        if column - current[-1] <= 1:
            current.append(column)
        else:
            groups.append(current)
            current = [column]
    groups.append(current)
    return [sum(g) / len(g) for g in groups]


class TestZoneGuides:
    def test_draws_two_vertical_guides(self):
        frame = bright_frame()
        draw_zone_guides(frame, 0.34)
        assert len(guide_centres(frame)) == 2

    def test_guides_are_symmetric_about_the_centre(self):
        frame = bright_frame()
        draw_zone_guides(frame, 0.34)
        left, right = guide_centres(frame)
        assert abs((320 - left) - (right - 320)) <= 1.5

    def test_guides_match_the_configured_band_width(self):
        frame = bright_frame()
        draw_zone_guides(frame, 0.34)
        left, right = guide_centres(frame)
        assert (right - left) == pytest.approx(0.34 * 640, abs=2)

    def test_wider_centre_band_pushes_the_guides_apart(self):
        narrow, wide = bright_frame(), bright_frame()
        draw_zone_guides(narrow, 0.20)
        draw_zone_guides(wide, 0.60)
        narrow_left, narrow_right = guide_centres(narrow)
        wide_left, wide_right = guide_centres(wide)
        assert (wide_right - wide_left) > (narrow_right - narrow_left)

    def test_guides_span_the_full_frame_height(self):
        frame = bright_frame()
        draw_zone_guides(frame, 0.34)
        assert len(guide_centres(frame, row=10)) == 2
        assert len(guide_centres(frame, row=470)) == 2


class TestCombinedOverlay:
    def test_full_overlay_renders_without_error(self):
        frame = bright_frame()
        canvas = draw_detections(
            frame,
            [
                make_detection(x=0, y=0, w=640, h=400, proximity=Proximity.IMMEDIATE),
                make_detection(x=400, y=200, w=100, h=150, zone=Zone.RIGHT),
            ],
        )
        draw_zone_guides(canvas, 0.34)
        draw_hud(canvas, "yolo", 12.5, 2, "person, very close, ahead")
        assert canvas.shape == (480, 640, 3)
