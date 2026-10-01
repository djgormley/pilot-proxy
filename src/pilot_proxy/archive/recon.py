"""Scope/dataset discovery map for the Datatrail-backed source."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .datatrail_client import DATATRAIL

_PRINTED_CHILD_LIMIT = 20


def match_terms(spec) -> List[str]:
    """Turn a comma-separated match option into ANDed lowercase terms."""
    if not spec:
        return []
    return [term.strip().lower() for term in str(spec).split(",")
            if term.strip()]


def _keep(text: str, terms: List[str]) -> bool:
    low = text.lower()
    return all(term in low for term in terms)


def recon(named_scopes, terms: List[str], out_dir: str,
          expand: bool = False, telescope=None,
          map_name: str = "scopes.jsonl") -> str:
    """Recursively list datasets without enumerating archive files.

    Explicit scopes win. Otherwise the live Datatrail namespace is optionally
    narrowed by the telescope's first scope component. Checked adapter calls
    keep outages distinct from genuine empty lists. With ``expand``, each
    matching container is opened one level and its resolvable children become
    the map rows.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    telescope_note = ""
    if named_scopes:
        scopes = list(named_scopes)
    else:
        scopes, ok = DATATRAIL.list_scopes_checked()
        if not ok:
            raise SystemExit(
                "datatrail could not list its scopes (service not "
                "responding?) -- there is nothing to walk. Re-run this recon "
                "when the service answers, or name scopes explicitly with "
                "--scope.")
        if not scopes:
            raise SystemExit(
                "datatrail reports zero scopes -- nothing to walk (an "
                "account/config problem, not an empty archive).")
        if telescope:
            telescope_name = str(telescope).lower()
            all_scope_count = len(scopes)
            scopes = [scope for scope in scopes
                      if str(scope).split(".")[0].lower() == telescope_name]
            if not scopes:
                raise SystemExit(
                    f"no datatrail scope has first component {telescope!r} "
                    f"({all_scope_count} scope(s) visible). Omit --telescope "
                    "to walk them all, or name scopes explicitly with --scope.")
            telescope_note = (
                f"; telescope={telescope} ({len(scopes)}/{all_scope_count} "
                "scope(s); omit --telescope to walk all)")

    print(f"[recon] listing datasets across {len(scopes)} scope(s)"
          + telescope_note
          + (f"; match={terms}" if terms else "")
          + ("; expanding matches one level" if expand else ""), flush=True)

    map_path = out / map_name
    row_count = expanded_count = 0
    failed: List[str] = []
    with open(map_path, "w") as handle:
        for index, scope in enumerate(scopes, 1):
            datasets, ok = DATATRAIL.list_datasets_checked(scope)
            if not ok:
                failed.append(f"datasets under scope {scope}")
                print(f"  [{index:>3}/{len(scopes)}] {scope}  -- NOT LISTED "
                      "(datatrail error; see [warn] below)", flush=True)
                continue
            kept = ([dataset for dataset in datasets
                     if _keep(f"{scope} {dataset}", terms)]
                    if terms else datasets)
            if not kept:
                continue
            print(f"  [{index:>3}/{len(scopes)}] {scope}  "
                  f"({len(kept)} dataset(s))", flush=True)
            for dataset in kept:
                children, children_ok = (
                    DATATRAIL.children_checked(scope, dataset)
                    if expand else ([], True))
                if expand and not children_ok:
                    failed.append(f"children of {scope} {dataset}")
                    print(f"        {dataset}  (children NOT listed -- "
                          "datatrail error; container row kept)")
                    handle.write(json.dumps(
                        {"scope": scope, "dataset": dataset}) + "\n")
                    row_count += 1
                    continue
                if children:
                    print(f"        {dataset}  ({len(children)} child(ren))")
                    for child in children[:_PRINTED_CHILD_LIMIT]:
                        print(f"            {child}")
                    if len(children) > _PRINTED_CHILD_LIMIT:
                        print(
                            f"            ... and "
                            f"{len(children) - _PRINTED_CHILD_LIMIT} more "
                            f"(all in {map_name})")
                    for child in children:
                        handle.write(json.dumps({
                            "scope": scope, "dataset": child,
                            "parent": dataset,
                        }) + "\n")
                        row_count += 1
                    expanded_count += 1
                else:
                    print(f"        {dataset}"
                          + ("  (no children listed)" if expand else ""))
                    handle.write(json.dumps(
                        {"scope": scope, "dataset": dataset}) + "\n")
                    row_count += 1

    tail = (
        "Every row resolves directly: `datatrail ps <scope> <dataset> -s`. "
        "Event-keyed scopes are then surveyed by re-running without "
        "--scopes-only; non-event products (timestamped acquisitions, "
        "calibration) need a per-product shape reader or a hand-written "
        "inventory -- survey's event walk will not see them."
        if expand else
        "For event-keyed data, pick the scope(s) you want and re-run survey "
        "without --scopes-only. A hit that is a CONTAINER of non-event "
        "children (timestamped acquisitions such as complex_gains) will NOT "
        "be walked by survey's event enumeration -- re-run this recon with "
        "--expand to list its children here.")
    print(f"\n[recon] wrote {map_path}: {row_count} rows"
          + (f" ({expanded_count} dataset(s) expanded)" if expand else "")
          + f". This is a discovery map, not the scan inventory. {tail}",
          flush=True)
    if failed:
        print("\n[warn] the map is INCOMPLETE -- datatrail did not answer for:"
              + "".join(f"\n    {item}" for item in failed)
              + f"\n  Rows for these are missing from (or unexpanded in) "
              f"{map_path}. Re-run this recon when the service responds; the "
              "walk is cheap and rebuilds the whole map.", flush=True)
    return str(map_path)
