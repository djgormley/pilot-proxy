"""Per-telescope channelization and hardware, stored as data.

A telescope is a YAML file under ``pilot_proxy/instruments/``. It describes the band geometry
(freq_id 0 frequency, processed bandwidth, channel count, ordering, Nyquist zone)
and the Datatrail baseband `scopes` its data lives under. Any frequency<->freq_id
mapping falls out of this geometry, so adding a telescope that shares an existing
channelization is a YAML entry (often just `extends:` plus a feed count and scope),
not a code change.

Nyquist zone is a telescope property. Odd zones use the normal baseband direction;
even zones are inverted. Current CHIME-family instruments in this repository use
the second Nyquist zone.

Which freq_ids are actually occupied is geography, not configuration, and is
discovered by a survey, not declared here.

This module is intentionally analysis-agnostic: what a given analysis looks for
(a particular carrier, an RFI model, a feed list) is a property of the analyzer,
not the telescope. The instrument is pure geometry plus data access metadata.
"""
from __future__ import annotations

import glob
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional

from pilot_proxy.archive.names import validate_identifier

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

_INSTRUMENT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "instruments")
DEFAULT_NFFT = 16384
_INSTRUMENT_KEYS = frozenset({
    "name", "band", "nyquist_zone", "n_feeds", "nfft", "scopes", "reader", "site",
    "feed_layout",
})
_SITE_KEYS = frozenset({"longitude_deg_east", "time_zone"})
_FEED_LAYOUT_KEYS = ("ew_spacing_m", "ns_spacing_m", "inputs_per_cylinder", "positions_per_cylinder")
_BAND_KEYS = frozenset({
    "f0_mhz", "bandwidth_mhz", "n_channels", "descending",
})


# --------------------------------------------------------------------------
# Instrument geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeedLayout:
    """A cylinder array's input layout: cylinder ``i // inputs_per_cylinder`` (EW step
    ``ew_spacing_m``) and position ``i % positions_per_cylinder`` (NS step
    ``ns_spacing_m``)."""
    ew_spacing_m: float
    ns_spacing_m: float
    inputs_per_cylinder: int
    positions_per_cylinder: int


@dataclass
class Instrument:
    name: str
    f0_mhz: float          # sky frequency at freq_id 0
    bandwidth_mhz: float   # total processed bandwidth
    n_channels: int        # number of frequency channels
    descending: bool       # True if higher freq_id => lower frequency (CHIME)
    nyquist_zone: int      # 1 = normal baseband direction, 2 = inverted (the
                           #   sign of the baseband->sky mapping; see nyquist_sign)
    n_feeds: int           # feeds/inputs incoherently combined
    nfft: int = DEFAULT_NFFT  # default analysis FFT length per frame
    scopes: tuple[str, ...] = ()  # datatrail baseband scope(s) this station registers;
                           # the default survey scope(s) when --scope is omitted.
    reader: str = ""       # canonical reader for this telescope
                           # (e.g. "chime-baseband"); the default `scan` uses
                           # when --reader is omitted. Pure data, like the rest.
    site_longitude_deg: Optional[float] = None  # east-positive site longitude;
                           # None when the file records no site
    local_time_zone: str = ""  # IANA zone of the site's civil time ("" if none)
    feed_layout: Optional[FeedLayout] = None  # None when the file records none

    @property
    def fs_hz(self) -> float:
        """Per-channel (complex) sample rate = channel spacing."""
        return self.bandwidth_mhz * 1e6 / self.n_channels

    @property
    def sample_rate_hz(self) -> float:
        """Complex sample rate of one channel (equal to the channel spacing)."""
        return self.fs_hz

    @property
    def channel_width_hz(self) -> float:
        """Spacing of adjacent channels in Hz."""
        return self.fs_hz

    @property
    def f0_hz(self) -> float:
        """Sky frequency of freq_id 0 in Hz (the band top when descending)."""
        return self.f0_mhz * 1e6

    @property
    def n_inputs(self) -> int:
        """Number of inputs summed incoherently (the feed count)."""
        return self.n_feeds

    def freq_id_of_hz(self, f_hz: float) -> int:
        """Nearest freq_id for a sky frequency in Hz, computed in Hz.

        This is the arithmetic of the former ``chime_freq_id_from_hz``:
        ``round((f0 - f) / width)`` for a descending channelization.
        """
        width = self.channel_width_hz
        if self.descending:
            return int(round((self.f0_hz - float(f_hz)) / width))
        return int(round((float(f_hz) - self.f0_hz) / width))

    def hz_of_freq_id(self, freq_id: float) -> float:
        """Channel-centre sky frequency in Hz for a freq_id, computed in Hz."""
        width = self.channel_width_hz
        if self.descending:
            return self.f0_hz - freq_id * width
        return self.f0_hz + freq_id * width

    @property
    def chan_step_mhz(self) -> float:
        sign = -1.0 if self.descending else 1.0
        return sign * self.bandwidth_mhz / self.n_channels

    def freq_of_freq_id(self, n: float) -> float:
        """Channel-center sky frequency (MHz) for freq_id n."""
        return self.f0_mhz + n * self.chan_step_mhz

    def freq_id_of_freq(self, f_mhz: float) -> int:
        """Nearest freq_id containing sky frequency f_mhz."""
        return int(round((f_mhz - self.f0_mhz) / self.chan_step_mhz))


def nyquist_sign(nyquist_zone: int) -> int:
    """The +1/-1 baseband->sky direction implied by a Nyquist zone.

    Odd zones keep the baseband direction (sky = f_center + baseband); even zones
    invert it (sky = f_center - baseband). This is the one place that rule lives.
    """
    return 1 if int(nyquist_zone) % 2 else -1


# --------------------------------------------------------------------------
# Loading + discovery
# --------------------------------------------------------------------------
def list_instrument_names(directory: Optional[str] = None) -> List[str]:
    directory = directory or _INSTRUMENT_DIR
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(directory, "*.yaml")))


def _load_yaml(name: str, directory: Optional[str] = None) -> dict:
    if yaml is None:
        raise RuntimeError("pyyaml not installed; "
                           "pip install pyyaml --break-system-packages")
    name = validate_identifier(name, label="instrument name")
    directory = directory or _INSTRUMENT_DIR
    path = os.path.join(directory, f"{name}.yaml")
    if not os.path.exists(path):
        opts = ", ".join(list_instrument_names(directory)) or "(none)"
        raise FileNotFoundError(f"no instrument config {name!r}. Available: {opts}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, over: dict) -> dict:
    """Merge `over` onto `base`. Nested dicts merge key-by-key; every other value
    (scalars, lists) is replaced wholesale by `over`."""
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_config(name: str, directory: Optional[str] = None,
                    _seen: Optional[set] = None) -> dict:
    """Load <name>.yaml, applying `extends: <parent>` inheritance.

    A telescope that shares another's geometry sets `extends: <parent>` and lists
    only what differs (feed count, scope, name); the parent's config is loaded
    first and the child's keys override it. This is what lets the CHIME outriggers
    be a few lines each instead of a near-duplicate of chime.yaml.
    """
    _seen = set() if _seen is None else _seen
    if name in _seen:
        chain = " -> ".join(list(_seen) + [name])
        raise ValueError(f"instrument '{name}': circular extends chain ({chain})")
    _seen.add(name)
    cfg = _load_yaml(name, directory)
    parent = cfg.get("extends")
    if parent:
        base = _resolve_config(str(parent), directory, _seen)
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "extends"})
    return cfg


def _coerce_scopes(value) -> tuple[str, ...]:
    """Accept a list or a comma-separated string of datatrail scopes."""
    if not value:
        return ()
    if isinstance(value, str):
        value = value.split(",")
    elif not isinstance(value, (list, tuple)):
        raise ValueError("instrument scopes must be a list or comma-separated string")
    if any(not isinstance(scope, str) for scope in value):
        raise ValueError("instrument scopes must contain only strings")
    return tuple(scope.strip() for scope in value if scope.strip())


def _positive_number(value, *, field: str, integer: bool = False):
    if isinstance(value, bool):
        kind = "an integer" if integer else "a number"
        raise ValueError(f"instrument {field} must be {kind}, got {value!r}")
    try:
        parsed_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"instrument {field} must be a number, got {value!r}") from exc
    if not math.isfinite(parsed_float):
        raise ValueError(f"instrument {field} must be finite, got {value!r}")
    if integer and not parsed_float.is_integer():
        raise ValueError(f"instrument {field} must be an integer, got {value!r}")
    parsed = int(parsed_float) if integer else parsed_float
    if parsed <= 0:
        raise ValueError(f"instrument {field} must be > 0, got {value!r}")
    return parsed


def _instrument_from_config(requested_name: str, cfg: Mapping) -> Instrument:
    """Validate one fully resolved config and construct its Instrument."""
    if not isinstance(cfg, Mapping):
        raise ValueError(f"instrument {requested_name!r}: YAML root must be a mapping")
    if "sense" in cfg:
        raise ValueError(
            f"instrument {requested_name!r}: unsupported configuration key "
            "'sense'; use nyquist_zone: 1 (normal) or 2 (inverted)")
    unknown = set(cfg) - _INSTRUMENT_KEYS
    if unknown:
        raise ValueError(
            f"instrument {requested_name!r}: unknown configuration key(s) "
            f"{sorted(map(str, unknown))}")
    configured_name = validate_identifier(
        cfg.get("name", requested_name), label="configured instrument name")
    if configured_name != requested_name:
        raise ValueError(
            f"instrument file {requested_name!r} declares name "
            f"{configured_name!r}; filename and configured name must match")
    band = cfg.get("band")
    if not isinstance(band, Mapping):
        raise ValueError(
            f"instrument {requested_name!r}: band must be a mapping containing "
            "f0_mhz, bandwidth_mhz, and n_channels")
    unknown_band = set(band) - _BAND_KEYS
    if unknown_band:
        raise ValueError(
            f"instrument {requested_name!r}: unknown band key(s) "
            f"{sorted(map(str, unknown_band))}")
    if "f0_mhz" not in band:
        raise ValueError(f"instrument {requested_name!r}: band.f0_mhz is required")
    if isinstance(band["f0_mhz"], bool):
        raise ValueError(
            f"instrument {requested_name!r}: band.f0_mhz must be numeric")
    try:
        f0_mhz = float(band["f0_mhz"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"instrument {requested_name!r}: band.f0_mhz must be numeric") from exc
    if not math.isfinite(f0_mhz):
        raise ValueError(
            f"instrument {requested_name!r}: band.f0_mhz must be finite")
    bandwidth_mhz = _positive_number(
        band.get("bandwidth_mhz"), field="band.bandwidth_mhz")
    n_channels = _positive_number(
        band.get("n_channels"), field="band.n_channels", integer=True)
    descending = band.get("descending", True)
    if not isinstance(descending, bool):
        raise ValueError(
            f"instrument {requested_name!r}: band.descending must be YAML true or "
            f"false, got {descending!r}")

    if cfg.get("nyquist_zone") is None:
        raise ValueError(
            f"instrument {requested_name!r}: nyquist_zone is required (1 for "
            "normal baseband direction, 2 for inverted)")
    nyquist_zone = _positive_number(
        cfg["nyquist_zone"], field="nyquist_zone", integer=True)
    n_feeds_raw = cfg.get("n_feeds", 0)
    try:
        n_feeds_float = float(n_feeds_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"instrument {requested_name!r}: n_feeds must be an integer") from exc
    if (not math.isfinite(n_feeds_float) or not n_feeds_float.is_integer()
            or isinstance(n_feeds_raw, bool)):
        raise ValueError(
            f"instrument {requested_name!r}: n_feeds must be an integer")
    n_feeds = int(n_feeds_float)
    if n_feeds < 0:
        raise ValueError(f"instrument {requested_name!r}: n_feeds must be >= 0")
    nfft = _positive_number(
        cfg.get("nfft", DEFAULT_NFFT), field="nfft", integer=True)
    reader = cfg.get("reader", "")
    if not isinstance(reader, str):
        raise ValueError(f"instrument {requested_name!r}: reader must be a string")
    if reader:
        reader = validate_identifier(reader, label="instrument reader")

    site_longitude_deg, local_time_zone = _site_from_config(
        requested_name, cfg.get("site"))
    feed_layout = _feed_layout_from_config(requested_name, cfg.get("feed_layout"))

    return Instrument(
        name=configured_name,
        f0_mhz=f0_mhz,
        bandwidth_mhz=bandwidth_mhz,
        n_channels=n_channels,
        descending=descending,
        nyquist_zone=nyquist_zone,
        n_feeds=n_feeds,
        nfft=nfft,
        scopes=_coerce_scopes(cfg.get("scopes")),
        reader=reader,
        site_longitude_deg=site_longitude_deg,
        local_time_zone=local_time_zone,
        feed_layout=feed_layout,
    )


def _feed_layout_from_config(requested_name: str, layout) -> Optional[FeedLayout]:
    """Validate the optional ``feed_layout`` block (``null``: none recorded)."""
    if layout is None:
        return None
    if not isinstance(layout, Mapping) or set(layout) != set(_FEED_LAYOUT_KEYS):
        raise ValueError(
            f"instrument {requested_name!r}: feed_layout must be null or a mapping of "
            f"exactly {list(_FEED_LAYOUT_KEYS)}")
    values = {}
    for key in _FEED_LAYOUT_KEYS:
        integer = not key.endswith("_m")
        values[key] = _positive_number(layout[key], field=f"feed_layout.{key}", integer=integer)
        if not integer:
            values[key] = float(values[key])
    return FeedLayout(**values)


def _site_from_config(requested_name: str, site) -> tuple[Optional[float], str]:
    """Validate the optional ``site`` block: longitude and civil time zone.

    ``site: null`` (as an outrigger that extends CHIME writes it) means the file
    records no site, so a child never inherits its parent's coordinates.
    """
    if site is None:
        return None, ""
    if not isinstance(site, Mapping):
        raise ValueError(f"instrument {requested_name!r}: site must be a mapping or null")
    unknown = set(site) - _SITE_KEYS
    if unknown:
        raise ValueError(
            f"instrument {requested_name!r}: unknown site key(s) "
            f"{sorted(map(str, unknown))}")
    longitude = site.get("longitude_deg_east")
    if isinstance(longitude, bool) or not isinstance(longitude, (int, float)):
        raise ValueError(
            f"instrument {requested_name!r}: site.longitude_deg_east must be a number")
    longitude = float(longitude)
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 180.0:
        raise ValueError(
            f"instrument {requested_name!r}: site.longitude_deg_east must be in [-180, 180]")
    zone = site.get("time_zone", "")
    if not isinstance(zone, str) or not zone.strip():
        raise ValueError(
            f"instrument {requested_name!r}: site.time_zone must be a non-empty string")
    return longitude, zone.strip()


def load_instrument(name: str, directory: Optional[str] = None) -> Instrument:
    """Load <name>.yaml (resolving `extends:`) into an Instrument.

    Configuration uses an explicit `nyquist_zone`; the unsupported `sense` key
    raises with a correction. The +1/-1 spectral direction is derived on demand
    via `nyquist_sign(instrument.nyquist_zone)`.
    """
    requested_name = validate_identifier(name, label="instrument name")
    return _instrument_from_config(
        requested_name, _resolve_config(requested_name, directory))


@dataclass(frozen=True)
class Readiness:
    """At-a-glance 'can I use this telescope yet?' summary for `list telescopes`."""
    name: str
    nyquist_zone_set: bool
    scopes_set: bool
    valid: bool = True
    problems: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        # Usable out of the box: geometry + a built-in default survey scope.
        return self.valid and self.nyquist_zone_set and self.scopes_set

    @property
    def status(self) -> str:
        if not self.valid:
            return "invalid"
        if self.ready:
            return "ready"
        # Valid instruments always have a nyquist zone; load_instrument rejects
        # incomplete geometry before a Readiness value can reach this branch.
        return "geometry-only"       # no built-in scope -> pass --scope

    def missing(self) -> List[str]:
        out = list(self.problems)
        if not self.nyquist_zone_set:
            out.append("nyquist_zone")
        if not self.scopes_set:
            out.append("scopes")
        return out

    def usable_for(self, needs_archive_config: bool) -> bool:
        """Can this telescope run with a source of the given kind?

        A local source needs only geometry + Nyquist zone. An archive source can
        survey any geometry-ready telescope, but `doctor`'s ready-combos require a
        built-in scope so they work with no extra args; a geometry-only telescope
        still works against an archive source if you pass --scope explicitly.
        """
        if not self.valid:
            return False
        if needs_archive_config:
            return self.ready
        return self.nyquist_zone_set


def instrument_readiness(name: str, directory: Optional[str] = None) -> Readiness:
    instrument = load_instrument(name, directory)
    return Readiness(
        name=instrument.name,
        nyquist_zone_set=True,
        scopes_set=bool(instrument.scopes),
    )
