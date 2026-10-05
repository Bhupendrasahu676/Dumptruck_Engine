"""Name-only GNSS/brake ablation policy; no data or target values are read.

The default leaves inputs intact. Enabled filtering also checks the existing
feature dependency contract, so an innocuously named derived input cannot hide
an excluded sensor. This is an ablation policy, not an observation allowlist.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import re


_TOKENS = re.compile(r"[^a-z0-9]+")
_GNSS_ALIASES = frozenset({"gnss", "gps"})
_BRAKE_ALIASES = frozenset({"brake", "brakes", "braking"})


def _validate_name(name: str) -> None:
    if not isinstance(name, str) or not name or any(character.isspace() for character in name):
        raise ValueError("Input names must be nonempty strings without whitespace")


def _validated_names(names: Iterable[str]) -> list[str]:
    if isinstance(names, (str, bytes, Mapping, set, frozenset)):
        raise ValueError("Input names must be an ordered iterable of unique names")
    try:
        result = list(names)
    except TypeError as exc:
        raise ValueError("Input names must be an ordered iterable of unique names") from exc
    for name in result:
        _validate_name(name)
    if len(set(result)) != len(result):
        raise ValueError("Input names must contain unique names; duplicates are invalid")
    return result


def _direct_reason(name: str) -> str | None:
    # Delimiters cover dotted sensor names, masks/windows, one-hot context and
    # both endpoints of leadlag names. Whole tokens avoid e.g. 'gpsignal' or
    # 'brakefluidity' accidentally matching unrelated unknown channel names.
    tokens = set(_TOKENS.split(name.lower()))
    if tokens & _GNSS_ALIASES:
        return "GNSS/GPS-related input"
    if tokens & _BRAKE_ALIASES:
        return "Brake/braking-related input"
    return None


def excluded_feature_reason(name: str) -> str | None:
    """Explain a direct or declared-dependency exclusion, otherwise return None.

    Unknown names still receive direct alias checks, but are not sent to the
    dependency parser (which assumes valid exporter feature syntax). The heavy
    feature-selection module is imported only when dependency lookup is needed.
    """
    _validate_name(name)
    reason = _direct_reason(name)
    if reason is not None:
        return reason

    from analysis import fault_features

    if name not in fault_features.allowed_input_names():
        return None
    dependencies = fault_features.sensor_dependencies(name)
    for kind in ("sensors", "context"):
        for dependency in dependencies.get(kind, []):
            reason = _direct_reason(dependency)
            if reason is not None:
                return f"{reason}; required {kind} dependency: {dependency}"
    return None


def filter_input_names(names: Iterable[str], exclude_gnss_brakes: bool = False
                       ) -> tuple[list[str], list[dict[str, str]]]:
    """Return kept names and {'attribute', 'reason'} removals in source order."""
    inputs = _validated_names(names)
    if not isinstance(exclude_gnss_brakes, bool):
        raise ValueError("exclude_gnss_brakes must be a boolean")
    if not exclude_gnss_brakes:
        return inputs, []
    kept, removed = [], []
    for name in inputs:
        reason = excluded_feature_reason(name)
        if reason is None:
            kept.append(name)
        else:
            removed.append({"attribute": name, "reason": reason})
    return kept, removed


def assert_allowed_names(names: Iterable[str]) -> None:
    """Raise ValueError when any input violates the GNSS/brake policy."""
    _, removed = filter_input_names(names, exclude_gnss_brakes=True)
    if removed:
        raise ValueError("Excluded GNSS/brake inputs: " + "; ".join(
            f"{record['attribute']} ({record['reason']})" for record in removed))
