"""Capture units: the per-frame covariance ratio of a class amplitude, and what the data say N and n_b are.

A class stack averages ``n_b`` redundant baseline products of ``N``-sample
frames. The capture's amplitude statistic is the noise-bias-free coherent
amplitude of that stack in units of the per-frame, per-product noise, so a net
amplitude ``A_net`` corresponds to the per-frame covariance ratio
``A_net^2 N n_b`` (amendment 18 item 1). The detection gate (``k`` control
scatters) converts the same way.

N is the detector configuration's frame length. n_b and the frames of a dump
are read from the product (``count`` and ``stacks.shape[0]``), never from a
table.
"""
from __future__ import annotations

import glob
import os

import numpy as np

# The ten baseline classes the capture reads, as (EW step, NS step), in the order of every capture table.
TEN_CLASSES = ((0, 1), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255), (1, 0), (1, 32), (2, 0), (3, 0))


def to_power(a, n_b, n_frame):
    """The per-frame covariance ratio ``A_net^2 N n_b`` of a net class amplitude. A reading at or below zero is
    not squared (it is not measured, and squaring would turn a deficit into an excess)."""
    return a * a * n_frame * n_b if a > 0 else a


def gate_power(gate, n_b, n_frame):
    """The detection gate (k s) becomes (k s)^2 N n_b."""
    return gate * gate * n_frame * n_b


def class_column(keys, cls, pol):
    """Column of the class stack (EW step, NS step) on the same-polarisation pair (pol, pol), or None."""
    k = np.asarray(keys)
    w = np.where((k[:, 0] == cls[0]) & (k[:, 1] == cls[1]) & (k[:, 2] == pol) & (k[:, 3] == pol))[0]
    return int(w[0]) if w.size else None


def class_baseline_count(product, cls, pol=0):
    """n_b: the redundant baseline products averaged into the class stack (the product's ``count``)."""
    j = class_column(product["keys"], cls, pol)
    if j is None:
        raise KeyError(f"class {cls} pol {pol} is not in the product")
    return int(product["count"][j])


def frames_per_dump(product):
    """Frames of a dump: the product's ``stacks`` axis 0."""
    return int(product["stacks"].shape[0])


def first_product(directory):
    """The first product file of a dump directory in name order (the file the frame count is read from)."""
    files = sorted(glob.glob(os.path.join(directory, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no products in {directory}")
    return files[0]


def frame_samples(project=None):
    """N, the samples per frame, from the project's detector configuration."""
    if project is None:
        from pilot_proxy.config.project import default_project
        project = default_project()
    return int(project.detector_config.nfft)


__all__ = ["TEN_CLASSES", "class_baseline_count", "class_column", "first_product", "frame_samples", "frames_per_dump",
           "gate_power", "to_power"]
