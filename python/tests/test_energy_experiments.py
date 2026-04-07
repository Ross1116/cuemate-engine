import numpy as np

from cuemate_analysis.energy_experiments import build_energy_candidate_set


def test_energy_candidate_set_is_bounded_and_discriminative() -> None:
    sr = 22050
    duration_seconds = 4.0
    timeline = np.linspace(0.0, duration_seconds, int(sr * duration_seconds), endpoint=False, dtype=np.float32)

    calm = 0.04 * np.sin(2.0 * np.pi * 220.0 * timeline)
    driving = (0.28 * np.sin(2.0 * np.pi * 220.0 * timeline)) + (0.16 * np.sin(2.0 * np.pi * 2600.0 * timeline))
    pulse_centers = np.arange(0.35, duration_seconds, 0.5)
    width = int(sr * 0.02)
    pulses = np.zeros_like(timeline)
    for center in pulse_centers:
        start = max(int(center * sr) - (width // 2), 0)
        stop = min(start + width, pulses.size)
        pulses[start:stop] += np.hanning(max(stop - start, 2))[: stop - start]
    driving = driving + (0.30 * pulses.astype(np.float32))

    calm_candidates = build_energy_candidate_set(calm.astype(np.float32), sr)
    driving_candidates = build_energy_candidate_set(driving.astype(np.float32), sr)

    allowed_negative_fields = {"loudness_lufs"}
    for key, value in calm_candidates.to_payload().items():
        if isinstance(value, float):
            assert 0.0 <= value <= 1.0 or (key in allowed_negative_fields and value < 0.0)
    for key, value in driving_candidates.to_payload().items():
        if isinstance(value, float):
            assert 0.0 <= value <= 1.0 or (key in allowed_negative_fields and value < 0.0)

    assert driving_candidates.baseline > calm_candidates.baseline
    assert driving_candidates.club_fusion > calm_candidates.club_fusion
    assert driving_candidates.pressure_fusion > calm_candidates.pressure_fusion
    assert driving_candidates.consensus > calm_candidates.consensus
