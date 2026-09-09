"""Alert selection: relevance, priority, cooldown and rate limiting."""

from __future__ import annotations

from conftest import FakeClock, make_detection

from safestep.alerts import AlertPolicy, describe
from safestep.spatial import Proximity, Zone


def policy(clock: FakeClock, **kwargs) -> AlertPolicy:
    defaults = dict(cooldown_s=4.0, min_gap_s=1.2, min_hits=1, min_proximity=Proximity.MODERATE)
    defaults.update(kwargs)
    return AlertPolicy(clock=clock, **defaults)


class TestDescribe:
    def test_centre_obstacle_omits_direction_unless_urgent(self):
        detection = make_detection(label="chair", zone=Zone.CENTER, proximity=Proximity.NEAR)
        assert describe(detection) == "chair, close"

    def test_immediate_centre_obstacle_states_direction(self):
        detection = make_detection(label="person", zone=Zone.CENTER, proximity=Proximity.IMMEDIATE)
        assert describe(detection) == "person, very close, ahead"

    def test_side_obstacle_states_direction(self):
        detection = make_detection(label="bicycle", zone=Zone.LEFT, proximity=Proximity.NEAR)
        assert describe(detection) == "bicycle, close, on your left"

    def test_label_comes_first(self):
        # The most decision-relevant word must arrive before the rest.
        assert describe(make_detection(label="car")).startswith("car")


class TestRelevanceFilter:
    def test_far_obstacles_are_ignored(self, clock):
        result = policy(clock).select([make_detection(proximity=Proximity.FAR)])
        assert result is None

    def test_moderate_obstacles_are_announced_by_default(self, clock):
        result = policy(clock).select([make_detection(proximity=Proximity.MODERATE)])
        assert result is not None

    def test_threshold_is_configurable(self, clock):
        strict = policy(clock, min_proximity=Proximity.IMMEDIATE)
        assert strict.select([make_detection(proximity=Proximity.NEAR)]) is None
        assert strict.select([make_detection(proximity=Proximity.IMMEDIATE)]) is not None

    def test_empty_input_is_silent(self, clock):
        assert policy(clock).select([]) is None


class TestConfirmation:
    def test_unconfirmed_track_is_not_announced(self, clock):
        p = policy(clock, min_hits=3)
        detection = make_detection(track_id=7)
        assert p.select([detection], hits_for=lambda _tid: 1) is None

    def test_confirmed_track_is_announced(self, clock):
        p = policy(clock, min_hits=3)
        detection = make_detection(track_id=7)
        assert p.select([detection], hits_for=lambda _tid: 3) is not None

    def test_confirmation_skipped_when_no_tracker_supplied(self, clock):
        p = policy(clock, min_hits=99)
        assert p.select([make_detection(track_id=7)]) is not None


class TestPriority:
    def test_closer_obstacle_wins(self, clock):
        far = make_detection(track_id=1, proximity=Proximity.MODERATE, label="chair")
        near = make_detection(track_id=2, proximity=Proximity.IMMEDIATE, label="person")
        result = policy(clock).select([far, near])
        assert result.label == "person"

    def test_centre_wins_over_side_at_equal_proximity(self, clock):
        side = make_detection(track_id=1, zone=Zone.LEFT, label="chair", proximity=Proximity.NEAR)
        centre = make_detection(track_id=2, zone=Zone.CENTER, label="person", proximity=Proximity.NEAR)
        result = policy(clock).select([side, centre])
        assert result.label == "person"

    def test_larger_wins_when_proximity_and_zone_tie(self, clock):
        small = make_detection(track_id=1, w=20, h=20, label="bottle", proximity=Proximity.NEAR)
        large = make_detection(track_id=2, w=200, h=200, label="car", proximity=Proximity.NEAR)
        result = policy(clock).select([small, large])
        assert result.label == "car"

    def test_only_one_announcement_per_pass(self, clock):
        detections = [
            make_detection(track_id=i, proximity=Proximity.NEAR, label=f"obj{i}") for i in range(5)
        ]
        result = policy(clock).select(detections)
        assert result is not None and result.label.startswith("obj")


class TestCooldown:
    def test_same_object_is_not_repeated_within_cooldown(self, clock):
        p = policy(clock, min_gap_s=0.0)
        detection = make_detection(track_id=1)

        assert p.select([detection]) is not None
        clock.advance(1.0)
        assert p.select([detection]) is None
        clock.advance(1.0)
        assert p.select([detection]) is None

    def test_same_object_is_repeated_after_cooldown(self, clock):
        p = policy(clock, cooldown_s=4.0, min_gap_s=0.0)
        detection = make_detection(track_id=1)

        assert p.select([detection]) is not None
        clock.advance(4.1)
        assert p.select([detection]) is not None

    def test_a_different_object_is_announced_immediately(self, clock):
        p = policy(clock, min_gap_s=0.0)
        assert p.select([make_detection(track_id=1, label="chair")]) is not None
        clock.advance(0.1)
        second = p.select([make_detection(track_id=2, label="person")])
        assert second is not None and second.label == "person"

    def test_untracked_objects_dedupe_by_class_and_zone(self, clock):
        p = policy(clock, min_gap_s=0.0)
        first = make_detection(track_id=None, label="obstacle", zone=Zone.LEFT)
        assert p.select([first]) is not None
        clock.advance(0.5)
        # Same class and zone, different pixels: still the same thing to a user.
        again = make_detection(track_id=None, label="obstacle", zone=Zone.LEFT, x=110)
        assert p.select([again]) is None

    def test_suppressed_primary_falls_through_to_next_candidate(self, clock):
        p = policy(clock, min_gap_s=0.0)
        loud = make_detection(track_id=1, proximity=Proximity.IMMEDIATE, label="person")
        quiet = make_detection(track_id=2, proximity=Proximity.NEAR, label="chair")

        assert p.select([loud, quiet]).label == "person"
        clock.advance(0.1)
        # 'person' is still cooling down, so the next-best obstacle is reported.
        assert p.select([loud, quiet]).label == "chair"

    def test_reset_clears_cooldowns(self, clock):
        p = policy(clock, min_gap_s=0.0)
        detection = make_detection(track_id=1)
        p.select([detection])
        p.reset()
        assert p.select([detection]) is not None


class TestGlobalRateLimit:
    def test_min_gap_blocks_rapid_announcements(self, clock):
        p = policy(clock, min_gap_s=1.2)
        assert p.select([make_detection(track_id=1, label="chair")]) is not None
        clock.advance(0.3)
        assert p.select([make_detection(track_id=2, label="person")]) is None

    def test_announcement_resumes_after_the_gap(self, clock):
        p = policy(clock, min_gap_s=1.2)
        assert p.select([make_detection(track_id=1, label="chair")]) is not None
        clock.advance(1.3)
        assert p.select([make_detection(track_id=2, label="person")]) is not None


class TestMemoryHygiene:
    def test_cooldown_table_does_not_grow_without_bound(self, clock):
        p = policy(clock, cooldown_s=1.0, min_gap_s=0.0)
        for track_id in range(200):
            p.select([make_detection(track_id=track_id)])
            clock.advance(0.5)
        # Entries older than 4x the cooldown are pruned on each announcement.
        assert len(p._last_announced_at) < 50
