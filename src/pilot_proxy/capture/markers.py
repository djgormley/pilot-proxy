"""Where each band is read in a capture: its marker channel, or the declared target of a control band.

Capture products carry the band label as an integer in ``meta["channel"]``
and the coarse channel in ``meta["freq_id"]``; these helpers give the same
two numbers from the project profile, so no band list or channel map is
written in code.
"""
from __future__ import annotations


def marker_freq_ids(project) -> dict[int, int]:
    """Band label (as an integer) to the freq_id it is read at: the channel holding the marker, or the declared
    target of a control band (the control band's nominal marker position)."""
    return {int(b.label): int(project.target_freq_id(b)) for b in project.frequency_plan.bands()}


def control_bands(project) -> list[int]:
    """The control bands' labels as integers: bands with no licensed emitter, whose bins give the control level."""
    return [int(b.label) for b in project.frequency_plan.bands("control")]


def screened_bands(project) -> list[int]:
    return [int(b.label) for b in project.frequency_plan.bands("screened")]


def resolve_project(path=None):
    """The project at ``path``, or the default project."""
    from pilot_proxy.config.project import default_project, load_project
    return load_project(path) if path else default_project()


__all__ = ["control_bands", "marker_freq_ids", "resolve_project", "screened_bands"]
