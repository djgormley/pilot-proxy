"""Write the capture's marker map for the numpy-only entry points.

Some capture measurements run under an interpreter without the profile's
YAML reader (the marker-to-in-band correction of record ran under such an
interpreter). This step, run under a full interpreter, writes what they need
from the project into one JSON file: each band's marker freq_id (the control
band's declared target), the frame length N, the control bands and the
capture record's science dumps.

usage: pilot-proxy capture marker-map --out JSON [--project DIR]
"""
from __future__ import annotations

import argparse
import json
import sys

from .markers import control_bands, marker_freq_ids, resolve_project

SCHEMA = "capture_marker_map_v1"


def marker_map(project) -> dict:
    record = project.record_module("capture_campaign")
    return {"schema": SCHEMA,
            "project": project.name,
            "markers": {str(band): fid for band, fid in marker_freq_ids(project).items()},
            "nfft": int(project.detector_config.nfft),
            "control_bands": [str(b) for b in control_bands(project)],
            "science_events": list(record.SCIENCE_EVENTS)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilot-proxy capture marker-map", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--project", default=None)
    args = ap.parse_args(argv)
    with open(args.out, "w") as fh:
        json.dump(marker_map(resolve_project(args.project)), fh, indent=1)
        fh.write("\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
