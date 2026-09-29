"""Project profile: one directory that names everything a project fixes.

``project.yaml`` names the instrument (a hardware file under
``pilot_proxy/instruments/``), the detector adapter, and the profile files:
frequency plan, interference template, detector configuration, integration
model, era list and detector register. Each part is loaded on first use.

The default project is the profile shipped with the repository,
``projects/chime_atsc``. In an installed wheel the same files are carried as
package resources.

``records`` (optional) names the project's package of code of record under
``pilot_proxy.records`` (the named configurations that reproduce a campaign's
releases, the ROC populations); the characterization resolves them through
the project, not through a package it imports by name.
"""
from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

from pilot_proxy import paths

from ._files import ProfileError, check_keys, read_yaml, text
from .detector_config import DetectorConfig, load_detector_config
from .eras import EraList, load_era_list
from .frequency_plan import Band, FrequencyPlan, load_frequency_plan
from .instrument import Instrument, load_instrument
from .integration_model import IntegrationModel, load_integration_model
from .interference import InterferenceTemplate, load_interference_template
from .register import Register, load_register

PROJECT_FILE = "project.yaml"
RECORDS_PACKAGE = "pilot_proxy.records"
DEFAULT_PROJECT_NAME = "chime_atsc"
_FILE_KEYS = ("frequency_plan", "interference_template", "detector_config",
              "integration_model", "register")
_ERA_KEYS = ("author_eras", "transmitter_off", "states")


class Project:
    """A loaded project profile. Parts load lazily and are cached."""

    def __init__(self, directory: Path, spec: Mapping[str, Any]):
        self.directory = Path(directory)
        where = str(self.directory / PROJECT_FILE)
        check_keys(spec, required=("name", "instrument", "detector_adapter", "eras")
                   + _FILE_KEYS, optional=("records",), where=where)
        self.name = text(spec["name"], field="name", where=where)
        self.records = text(spec["records"], field="records", where=where) if "records" in spec else None
        if self.records is not None and not self.records.startswith(RECORDS_PACKAGE + "."):
            raise ProfileError(f"{where}: records {self.records!r} is not a package under {RECORDS_PACKAGE}")
        self.instrument_name = text(spec["instrument"], field="instrument", where=where)
        self.detector_adapter = text(spec["detector_adapter"], field="detector_adapter",
                                     where=where)
        eras = spec["eras"]
        if not isinstance(eras, Mapping):
            raise ProfileError(f"{where}: eras must be a mapping")
        check_keys(eras, required=_ERA_KEYS, where=f"{where}: eras")
        if not isinstance(eras["states"], list):
            raise ProfileError(f"{where}: eras.states must be a list")
        self.era_states = tuple(eras["states"])
        files = {key: text(spec[key], field=key, where=where) for key in _FILE_KEYS}
        files["author_eras"] = text(eras["author_eras"], field="eras.author_eras", where=where)
        files["transmitter_off"] = text(eras["transmitter_off"], field="eras.transmitter_off",
                                        where=where)
        self.files = {key: self._inside(value, where) for key, value in files.items()}

    def _inside(self, relative: str, where: str) -> Path:
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ProfileError(f"{where}: {relative!r} leaves the project directory")
        return path

    @cached_property
    def instrument(self) -> Instrument:
        return load_instrument(self.instrument_name)

    @cached_property
    def frequency_plan(self) -> FrequencyPlan:
        return load_frequency_plan(self.files["frequency_plan"])

    @cached_property
    def interference(self) -> InterferenceTemplate:
        template = load_interference_template(self.files["interference_template"])
        if template.kind != self.detector_adapter:
            raise ProfileError(
                f"{self.files['interference_template']}: kind {template.kind!r} does not "
                f"match the project's detector_adapter {self.detector_adapter!r}")
        return template

    @cached_property
    def detector_config(self) -> DetectorConfig:
        return load_detector_config(self.files["detector_config"])

    @cached_property
    def integration_model(self) -> IntegrationModel:
        return load_integration_model(self.files["integration_model"])

    @cached_property
    def eras(self) -> EraList:
        return load_era_list(self.files["author_eras"], self.files["transmitter_off"],
                             self.era_states)

    def era_list(self, author_eras: str, sha256: str | None = None) -> EraList:
        """Another author-dated era list of the project (a record's), with the profile's
        transmitter-off record and states; refused when its bytes are not ``sha256``."""
        path = self._inside(author_eras, str(self.directory / PROJECT_FILE))
        eras = load_era_list(path, self.files["transmitter_off"], self.era_states)
        if sha256 is not None and eras.source_sha256 != sha256:
            raise ProfileError(f"{path}: sha256 {eras.source_sha256} is not the recorded {sha256}")
        return eras

    @cached_property
    def register(self) -> Register:
        return load_register(self.files["register"], side="detector")

    def marker_hz(self, band: Band | str) -> float:
        """Nominal marker frequency of a band: lower edge plus the template offset."""
        band = band if isinstance(band, Band) else self.frequency_plan.band(band)
        marker = self.interference.marker
        if marker is None:
            raise ProfileError(f"template kind {self.interference.kind!r} has no marker")
        return band.low_hz + marker.offset_hz

    def target_freq_id(self, band: Band | str) -> int:
        """The instrument channel a band is read at: the declared target when the
        plan gives one (a control band), else the channel holding the marker."""
        band = band if isinstance(band, Band) else self.frequency_plan.band(band)
        if band.target_freq_id is not None:
            return band.target_freq_id
        return self.instrument.freq_id_of_hz(self.marker_hz(band))

    def record_module(self, part: str):
        """One module of the project's code of record (``<records>.<part>``); refused when none is named."""
        if self.records is None:
            raise ProfileError(f"{self.directory / PROJECT_FILE}: the project names no records package")
        return importlib.import_module(f"{self.records}.{part}")

    def file_sha256(self) -> dict[str, str]:
        """sha256 of every profile file, for run manifests."""
        names = {"project": self.directory / PROJECT_FILE, **self.files}
        return {key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in sorted(names.items())}

    def load_all(self) -> "Project":
        """Load and cross-check every part now (a profile check)."""
        for part in ("instrument", "frequency_plan", "interference", "detector_config",
                     "integration_model", "eras", "register"):
            getattr(self, part)
        if self.detector_config.nfft != self.instrument.nfft:
            raise ProfileError(
                f"{self.files['detector_config']}: nfft {self.detector_config.nfft} differs "
                f"from instrument {self.instrument.name!r} nfft {self.instrument.nfft}")
        for band in self.frequency_plan.bands():
            if band.target_freq_id is not None and not (
                    0 <= band.target_freq_id < self.instrument.n_channels):
                raise ProfileError(
                    f"band {band.label}: target_freq_id {band.target_freq_id} is outside "
                    f"instrument {self.instrument.name!r}")
        return self


def load_project(directory: Path | str) -> Project:
    directory = Path(directory)
    return Project(directory, read_yaml(directory / PROJECT_FILE))


def default_project_dir() -> Path:
    """The repository's project profile (or its copy inside an installed wheel)."""
    return paths.DATA_ROOT / "projects" / DEFAULT_PROJECT_NAME


@lru_cache(maxsize=1)
def default_project() -> Project:
    return load_project(default_project_dir())


__all__ = [
    "DEFAULT_PROJECT_NAME",
    "PROJECT_FILE",
    "RECORDS_PACKAGE",
    "Project",
    "default_project",
    "default_project_dir",
    "load_project",
]
