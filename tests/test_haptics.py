"""Vibration-motor mapping."""

from __future__ import annotations

import pytest
from conftest import make_detection

from safestep.haptics import IDLE, cue_for, cue_for_detection, strongest_cue
from safestep.spatial import Proximity, Zone, proximity_from_distance


class TestIntensity:
    def test_urgency_raises_intensity_monotonically(self):
        bands = [Proximity.FAR, Proximity.MODERATE, Proximity.NEAR, Proximity.IMMEDIATE]
        values = [cue_for(b, Zone.CENTER).intensity for b in bands]
        assert values == sorted(values)

    def test_far_obstacles_do_not_buzz(self):
        cue = cue_for(Proximity.FAR, Zone.CENTER)
        assert cue.intensity == 0.0
        assert cue.active is False

    def test_immediate_is_full_strength(self):
        assert cue_for(Proximity.IMMEDIATE, Zone.CENTER).intensity == 1.0

    def test_intensities_stay_in_range(self):
        for band in Proximity:
            for zone in Zone:
                cue = cue_for(band, zone)
                assert 0.0 <= cue.left <= 1.0
                assert 0.0 <= cue.right <= 1.0


class TestPulseRate:
    def test_urgency_raises_pulse_rate_monotonically(self):
        bands = [Proximity.FAR, Proximity.MODERATE, Proximity.NEAR, Proximity.IMMEDIATE]
        rates = [cue_for(b, Zone.CENTER).pulse_hz for b in bands]
        assert rates == sorted(rates)

    def test_far_has_no_pulse(self):
        assert cue_for(Proximity.FAR, Zone.CENTER).pulse_hz == 0.0


class TestDirection:
    def test_left_obstacle_drives_the_left_motor_hardest(self):
        cue = cue_for(Proximity.NEAR, Zone.LEFT)
        assert cue.left > cue.right

    def test_right_obstacle_drives_the_right_motor_hardest(self):
        cue = cue_for(Proximity.NEAR, Zone.RIGHT)
        assert cue.right > cue.left

    def test_centre_obstacle_is_balanced(self):
        cue = cue_for(Proximity.NEAR, Zone.CENTER)
        assert cue.left == pytest.approx(cue.right)

    def test_off_side_motor_is_damped_not_silenced(self):
        # If one motor loses skin contact the user should still feel something.
        cue = cue_for(Proximity.IMMEDIATE, Zone.LEFT)
        assert cue.right > 0.0

    def test_left_and_right_are_mirror_images(self):
        left = cue_for(Proximity.NEAR, Zone.LEFT)
        right = cue_for(Proximity.NEAR, Zone.RIGHT)
        assert left.left == pytest.approx(right.right)
        assert left.right == pytest.approx(right.left)


class TestStrongest:
    def test_no_detections_is_idle(self):
        assert strongest_cue([]) is IDLE

    def test_none_detection_is_idle(self):
        assert cue_for_detection(None) is IDLE

    def test_most_urgent_obstacle_wins(self):
        cue = strongest_cue([
            make_detection(proximity=Proximity.MODERATE, zone=Zone.LEFT),
            make_detection(proximity=Proximity.IMMEDIATE, zone=Zone.RIGHT),
        ])
        assert cue.proximity is Proximity.IMMEDIATE
        assert cue.zone is Zone.RIGHT

    def test_centre_breaks_a_proximity_tie(self):
        cue = strongest_cue([
            make_detection(proximity=Proximity.NEAR, zone=Zone.LEFT),
            make_detection(proximity=Proximity.NEAR, zone=Zone.CENTER),
        ])
        assert cue.zone is Zone.CENTER

    def test_larger_object_breaks_a_remaining_tie(self):
        cue = strongest_cue([
            make_detection(proximity=Proximity.NEAR, zone=Zone.LEFT, w=10, h=10),
            make_detection(proximity=Proximity.NEAR, zone=Zone.RIGHT, w=200, h=200),
        ])
        assert cue.zone is Zone.RIGHT

    def test_only_one_obstacle_drives_the_motors(self):
        # Superimposing patterns produces something the wearer cannot decode.
        cue = strongest_cue([
            make_detection(proximity=Proximity.NEAR, zone=Zone.LEFT),
            make_detection(proximity=Proximity.NEAR, zone=Zone.RIGHT),
            make_detection(proximity=Proximity.IMMEDIATE, zone=Zone.CENTER),
        ])
        assert cue.proximity is Proximity.IMMEDIATE
        assert cue.zone is Zone.CENTER


class TestSerialisation:
    def test_as_dict_carries_every_ui_field(self):
        payload = cue_for(Proximity.IMMEDIATE, Zone.LEFT).as_dict()
        assert set(payload) == {
            "left", "right", "pulseHz", "intensity", "strength", "proximity", "zone", "active",
        }

    def test_strength_is_the_peak_motor_value(self):
        cue = cue_for(Proximity.IMMEDIATE, Zone.LEFT)
        assert cue.as_dict()["strength"] == pytest.approx(max(cue.left, cue.right))

    def test_idle_serialises_as_inactive(self):
        assert IDLE.as_dict()["active"] is False


class TestDistanceScaledStrength:
    """Strength varies continuously with distance, not in three fixed steps.

    Banding alone meant an obstacle closing from 4 m to 1 m felt identical
    until it crossed a threshold; the wearer could not feel it approaching.
    """

    def test_strength_falls_off_monotonically_with_distance(self):
        values = [
            cue_for(proximity_from_distance(d), Zone.CENTER, d).intensity
            for d in (0.5, 1.0, 2.0, 3.0, 4.0, 4.9)
        ]
        assert values == sorted(values, reverse=True)

    def test_full_power_at_very_close_range(self):
        assert cue_for(Proximity.IMMEDIATE, Zone.CENTER, 0.3).intensity == 1.0
        assert cue_for(Proximity.IMMEDIATE, Zone.CENTER, 0.5).intensity == 1.0

    def test_silent_beyond_maximum_range(self):
        assert cue_for(Proximity.FAR, Zone.CENTER, 6.0).intensity == 0.0

    def test_mid_range_obstacle_is_partial_strength(self):
        strength = cue_for(Proximity.NEAR, Zone.CENTER, 2.0).intensity
        assert 0.4 < strength < 0.9

    def test_close_obstacle_beats_a_distant_one(self):
        near = cue_for(proximity_from_distance(0.8), Zone.CENTER, 0.8)
        far = cue_for(proximity_from_distance(4.0), Zone.CENTER, 4.0)
        assert near.intensity > far.intensity * 2

    def test_pulse_rate_also_rises_as_the_obstacle_closes(self):
        rates = [
            cue_for(proximity_from_distance(d), Zone.CENTER, d).pulse_hz
            for d in (0.5, 2.0, 4.0)
        ]
        assert rates == sorted(rates, reverse=True)

    def test_band_floor_keeps_a_reported_obstacle_perceptible(self):
        # A 4.9 m obstacle is still worth announcing, so it must still be felt.
        assert cue_for(Proximity.MODERATE, Zone.CENTER, 4.9).intensity >= 0.2

    def test_falls_back_to_the_band_without_a_distance(self):
        assert cue_for(Proximity.NEAR, Zone.CENTER, None).intensity == 0.65

    def test_far_band_stays_silent_even_with_a_close_reading(self):
        # Guards against a bad distance overriding the considered band.
        assert cue_for(Proximity.FAR, Zone.CENTER, 0.2).intensity == 0.0

    def test_direction_balance_still_applies(self):
        cue = cue_for(Proximity.IMMEDIATE, Zone.LEFT, 0.5)
        assert cue.left > cue.right

    def test_detection_distance_drives_the_cue(self):
        close = cue_for_detection(
            make_detection(proximity=Proximity.NEAR, distance_m=1.0)
        )
        distant = cue_for_detection(
            make_detection(proximity=Proximity.NEAR, distance_m=2.4)
        )
        assert close.intensity > distant.intensity
