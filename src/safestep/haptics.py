"""Mapping obstacles onto vibration motors.

A wearable SafeStep carries two vibration motors -- one on each side of a belt,
strap or headband. Haptics are the third feedback channel alongside earcons and
speech, and in some ways the best of the three: vibration is instantaneous,
survives a noisy street, and does not occupy the user's hearing, which a blind
person depends on for everything else.

Two properties carry the message:

* **Intensity and pulse rate encode urgency.** Closer is stronger and faster,
  matching the earcon design so the two channels reinforce rather than compete.
* **Left/right balance encodes direction.** An obstacle on the left buzzes the
  left motor hardest, mirroring the stereo panning of the audio cue.

This module is pure arithmetic with no hardware dependency, so it drives the
on-screen meters in the web UI today and would drive real GPIO PWM channels
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .spatial import Detection, Proximity, Zone

#: Distance at which the motors reach full power, and the range beyond which
#: they fall silent. Between the two, strength varies continuously.
FULL_POWER_M = 0.5
MAX_RANGE_M = 5.0

#: Overall motor intensity (0..1) per urgency band. Used when no metric
#: distance is available, and as a floor so that anything worth announcing is
#: still perceptible through a coat or a strap.
_INTENSITY: Dict[Proximity, float] = {
    Proximity.FAR: 0.0,
    Proximity.MODERATE: 0.35,
    Proximity.NEAR: 0.65,
    Proximity.IMMEDIATE: 1.0,
}

#: Lower bound per band once a continuous distance reading is in play.
_INTENSITY_FLOOR: Dict[Proximity, float] = {
    Proximity.FAR: 0.0,
    Proximity.MODERATE: 0.20,
    Proximity.NEAR: 0.45,
    Proximity.IMMEDIATE: 0.75,
}


def intensity_for_distance(distance_m: float) -> float:
    """Continuous motor strength from a metric distance.

    Ramps linearly from full power at :data:`FULL_POWER_M` down to nothing at
    :data:`MAX_RANGE_M`. Banding alone made the motors step between three fixed
    levels, so an obstacle closing from 4 m to 1 m felt identical until it
    crossed a threshold; a continuous ramp lets the wearer feel it approaching.
    """
    if distance_m <= FULL_POWER_M:
        return 1.0
    if distance_m >= MAX_RANGE_M:
        return 0.0
    return (MAX_RANGE_M - distance_m) / (MAX_RANGE_M - FULL_POWER_M)


def pulse_hz_for_distance(distance_m: float) -> float:
    """Continuous pulse rate: faster as the obstacle gets closer."""
    return 1.0 + 6.0 * intensity_for_distance(distance_m)


#: Pulses per second per urgency band. Matches the earcon pulse counts so the
#: audio and haptic channels beat together rather than against each other.
_PULSE_HZ: Dict[Proximity, float] = {
    Proximity.FAR: 0.0,
    Proximity.MODERATE: 1.5,
    Proximity.NEAR: 3.0,
    Proximity.IMMEDIATE: 6.0,
}

#: (left, right) motor weighting per zone. The off-side is damped rather than
#: silenced so the user still feels a cue if one motor loses skin contact.
_BALANCE: Dict[Zone, Tuple[float, float]] = {
    Zone.LEFT: (1.0, 0.2),
    Zone.CENTER: (1.0, 1.0),
    Zone.RIGHT: (0.2, 1.0),
}


@dataclass(frozen=True)
class HapticCue:
    """A vibration pattern for one obstacle."""

    left: float          # left motor duty cycle, 0..1
    right: float         # right motor duty cycle, 0..1
    pulse_hz: float      # pulses per second; 0 means steady/off
    intensity: float     # overall urgency level, 0..1
    proximity: Proximity
    zone: Zone

    @property
    def active(self) -> bool:
        return self.intensity > 0.0

    @property
    def strongest(self) -> float:
        """Peak duty cycle across both motors -- the headline 'strength' figure."""
        return max(self.left, self.right)

    def as_dict(self) -> dict:
        return {
            "left": round(self.left, 3),
            "right": round(self.right, 3),
            "pulseHz": round(self.pulse_hz, 2),
            "intensity": round(self.intensity, 3),
            "strength": round(self.strongest, 3),
            "proximity": self.proximity.name,
            "zone": self.zone.value,
            "active": self.active,
        }


IDLE = HapticCue(
    left=0.0,
    right=0.0,
    pulse_hz=0.0,
    intensity=0.0,
    proximity=Proximity.FAR,
    zone=Zone.CENTER,
)


def cue_for(
    proximity: Proximity, zone: Zone, distance_m: Optional[float] = None
) -> HapticCue:
    """Build the vibration pattern for an obstacle.

    With a metric ``distance_m`` the strength varies continuously and the band
    only sets a floor, so the wearer feels an obstacle closing rather than
    three discrete steps. Without one, the band alone drives the motors.
    """
    if distance_m is None:
        intensity = _INTENSITY[proximity]
        pulse = _PULSE_HZ[proximity]
    elif proximity is Proximity.FAR:
        # Too far to be worth reporting at all, whatever the raw number says.
        intensity, pulse = 0.0, 0.0
    else:
        intensity = max(intensity_for_distance(distance_m), _INTENSITY_FLOOR[proximity])
        pulse = pulse_hz_for_distance(distance_m)

    left_weight, right_weight = _BALANCE[zone]
    return HapticCue(
        left=intensity * left_weight,
        right=intensity * right_weight,
        pulse_hz=pulse,
        intensity=intensity,
        proximity=proximity,
        zone=zone,
    )


def cue_for_detection(detection: Optional[Detection]) -> HapticCue:
    """Vibration pattern for a single detection, or :data:`IDLE` for ``None``."""
    if detection is None:
        return IDLE
    return cue_for(detection.proximity, detection.zone, detection.distance_m)


def strongest_cue(detections) -> HapticCue:
    """Pick the cue for the most urgent obstacle in ``detections``.

    Only one obstacle drives the motors at a time. Superimposing several
    patterns produces a muddle the wearer cannot decode -- the same reason the
    alert policy announces one obstacle rather than a list.
    """
    best: Optional[Detection] = None
    for detection in detections:
        if best is None or detection.proximity > best.proximity:
            best = detection
        elif detection.proximity == best.proximity:
            # Break ties toward what is straight ahead, then toward the larger
            # object -- the same priority order the alert policy uses.
            best_centre = best.zone is Zone.CENTER
            this_centre = detection.zone is Zone.CENTER
            if (this_centre and not best_centre) or (
                this_centre == best_centre and detection.bbox.area > best.bbox.area
            ):
                best = detection

    return cue_for_detection(best)
