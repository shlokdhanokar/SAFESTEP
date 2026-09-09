"""IoU tracker identity behaviour."""

from __future__ import annotations

import pytest
from conftest import make_detection

from safestep.tracking import IoUTracker


class TestIdentityAssignment:
    def test_first_frame_mints_new_ids(self):
        tracker = IoUTracker()
        tracked = tracker.update([make_detection(x=0), make_detection(x=300)])
        assert [d.track_id for d in tracked] == [1, 2]

    def test_slowly_moving_object_keeps_its_id(self):
        tracker = IoUTracker()
        first = tracker.update([make_detection(x=100, y=100, w=50, h=100)])
        second = tracker.update([make_detection(x=105, y=100, w=50, h=100)])
        assert first[0].track_id == second[0].track_id == 1

    def test_object_that_jumps_gets_a_new_id(self):
        tracker = IoUTracker()
        tracker.update([make_detection(x=0, y=0, w=50, h=50)])
        moved = tracker.update([make_detection(x=500, y=400, w=50, h=50)])
        assert moved[0].track_id == 2

    def test_two_objects_do_not_swap_identities(self):
        tracker = IoUTracker()
        tracker.update([make_detection(x=0, w=60, h=60), make_detection(x=400, w=60, h=60)])
        second = tracker.update(
            [make_detection(x=5, w=60, h=60), make_detection(x=405, w=60, h=60)]
        )
        assert second[0].track_id == 1
        assert second[1].track_id == 2

    def test_ids_are_never_reused_after_a_track_dies(self):
        tracker = IoUTracker(max_missing=1)
        tracker.update([make_detection(x=0, w=50, h=50)])
        for _ in range(4):
            tracker.update([])
        fresh = tracker.update([make_detection(x=0, w=50, h=50)])
        assert fresh[0].track_id == 2

    def test_empty_update_returns_empty(self):
        assert IoUTracker().update([]) == []

    def test_detections_are_returned_in_input_order(self):
        tracker = IoUTracker()
        tracked = tracker.update(
            [make_detection(x=400, label="chair"), make_detection(x=0, label="person")]
        )
        assert [d.label for d in tracked] == ["chair", "person"]


class TestHitCounting:
    def test_hits_accumulate_across_frames(self):
        tracker = IoUTracker()
        for _ in range(3):
            tracker.update([make_detection(x=100, w=50, h=100)])
        assert tracker.hits_for(1) == 3

    def test_new_track_starts_at_one_hit(self):
        tracker = IoUTracker()
        tracker.update([make_detection()])
        assert tracker.hits_for(1) == 1

    def test_unknown_track_reports_zero(self):
        assert IoUTracker().hits_for(999) == 0


class TestTrackExpiry:
    def test_track_survives_brief_dropout(self):
        tracker = IoUTracker(max_missing=3)
        tracker.update([make_detection(x=100, w=50, h=100)])
        for _ in range(3):
            tracker.update([])
        recovered = tracker.update([make_detection(x=100, w=50, h=100)])
        assert recovered[0].track_id == 1

    def test_track_expires_after_max_missing(self):
        tracker = IoUTracker(max_missing=2)
        tracker.update([make_detection(x=100, w=50, h=100)])
        for _ in range(3):
            tracker.update([])
        assert tracker.hits_for(1) == 0
        assert len(tracker.active_tracks) == 0

    def test_reset_clears_everything(self):
        tracker = IoUTracker()
        tracker.update([make_detection()])
        tracker.reset()
        assert len(tracker.active_tracks) == 0
        assert tracker.update([make_detection()])[0].track_id == 1


class TestValidation:
    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.1])
    def test_invalid_iou_threshold_rejected(self, threshold):
        with pytest.raises(ValueError):
            IoUTracker(iou_threshold=threshold)
