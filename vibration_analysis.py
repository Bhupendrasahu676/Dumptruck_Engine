"""Small stdlib FFT feature extractor for real high-rate or synthetic waveforms.

Frequency bands are descriptive inputs, never bearing/root-cause diagnoses.
Use actual RPM and sensor bandwidth when deploying outside this simulator.
"""
import cmath
import math


def spectral_features(record):
    values = record.get("waveform", [])
    rate, rpm = record.get("sample_rate_hz"), record.get("rpm")
    if not isinstance(values, (list, tuple)) or not 8 <= len(values) <= 262144:
        raise ValueError("waveform needs 8..262144 samples")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values):
        raise ValueError("waveform values must be finite numbers")
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not isinstance(rpm, (int, float)) or isinstance(rpm, bool) or not math.isfinite(rpm) or rpm < 0:
        raise ValueError("rpm must be nonnegative and finite")
    n = len(values)
    mean = sum(values) / n
    centered = [v - mean for v in values]
    rms = math.sqrt(sum(v * v for v in centered) / n)
    size = 1 << (n - 1).bit_length()
    signal = [complex(v * (.5 - .5 * math.cos(2 * math.pi * i / (n - 1))))
              for i, v in enumerate(centered)] + [0j] * (size - n)
    # In-place radix-2 Cooley-Tukey, including zero-padding non-power-of-two data.
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            signal[i], signal[j] = signal[j], signal[i]
    width = 2
    while width <= size:
        root = cmath.exp(-2j * math.pi / width)
        for start in range(0, size, width):
            factor = 1 + 0j
            for offset in range(width // 2):
                left = signal[start + offset]
                right = factor * signal[start + offset + width // 2]
                signal[start + offset] = left + right
                signal[start + offset + width // 2] = left - right
                factor *= root
        width *= 2
    powers = [abs(signal[i]) ** 2 for i in range(1, size // 2 + 1)]
    frequencies = [i * rate / size for i in range(1, size // 2 + 1)]
    total = sum(powers)
    def fraction(low, high):
        return sum(p for f, p in zip(frequencies, powers) if low <= f < high) / total if total else 0.
    shaft = rpm / 60
    peak_index = max(range(len(powers)), key=powers.__getitem__)
    features = {
        "vibration.rms_mm_s": rms,
        "vibration.peak_to_peak_mm_s": max(values) - min(values),
        "vibration.crest_factor": max(map(abs, centered)) / rms if rms else 0.,
        "vibration.kurtosis": sum(v ** 4 for v in centered) / n / rms ** 4 if rms else 0.,
        "vibration.dominant_frequency_hz": frequencies[peak_index] if total else 0.,
        "vibration.frequency_resolution_hz": rate / n,
        "vibration.shaft_frequency_hz": shaft,
        "vibration.order_1_energy_fraction": fraction(.8 * shaft, 1.2 * shaft) if shaft else 0.,
        "vibration.order_2_energy_fraction": fraction(1.8 * shaft, 2.2 * shaft) if shaft else 0.,
        "vibration.order_4_to_6_energy_fraction": fraction(4 * shaft, 6 * shaft) if shaft else 0.,
        "vibration.order_above_10_energy_fraction": fraction(10 * shaft, rate / 2) if shaft else 0.,
    }
    return features
