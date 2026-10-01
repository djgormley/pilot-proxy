#!/usr/bin/env python3
"""Report local archive process, disk, heartbeat, GPU, and log health."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


LOG_PATTERNS = (
    re.compile(r"traceback", re.IGNORECASE),
    re.compile(r"(?:cuda|gpu).*(?:out of memory|error)", re.IGNORECASE),
    re.compile(r"out of memory", re.IGNORECASE),
    re.compile(r"no space left", re.IGNORECASE),
    re.compile(r"certificate.*(?:expired|expiry)", re.IGNORECASE),
    re.compile(r"fetch.*(?:failed|error)", re.IGNORECASE),
    re.compile(r"\bquarantine\b", re.IGNORECASE),
    re.compile(r"\b[1-9][0-9]* quarantined\b", re.IGNORECASE),
    re.compile(r"\bwas quarantined\b", re.IGNORECASE),
    re.compile(r"worker.*(?:failed|error)", re.IGNORECASE),
)


def result(check: str, status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"check": check, "status": status, "message": message, **details}


def archive_processes(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "cmdline").read_bytes().split(b"\0")
            arguments = [field.decode("utf-8", "replace") for field in fields if field]
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        joined = " ".join(arguments)
        if "pilot-proxy" in joined and "chime-scan" in arguments:
            matches.append({"pid": int(entry.name), "command": joined})
    return sorted(matches, key=lambda item: item["pid"])


def process_check(expect_running: bool, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    processes = archive_processes(proc_root)
    if expect_running and len(processes) != 1:
        return result(
            "process",
            "error",
            f"expected one archive scan, found {len(processes)}",
            processes=processes,
        )
    if not expect_running and processes:
        return result(
            "process",
            "warning",
            f"found {len(processes)} archive scan process(es)",
            processes=processes,
        )
    message = "one archive scan is running" if processes else "no archive scan is running"
    return result("process", "ok", message, processes=processes)


def nearest_existing(path: Path) -> Path:
    candidate = path.expanduser().resolve(strict=False)
    while not candidate.exists():
        if candidate == candidate.parent:
            raise FileNotFoundError(path)
        candidate = candidate.parent
    return candidate


def disk_checks(paths: Iterable[Path], min_free_gib: float) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for requested in paths:
        try:
            measured = nearest_existing(requested)
            usage = shutil.disk_usage(measured)
        except OSError as exc:
            checks.append(result("disk", "error", f"cannot measure {requested}: {exc}"))
            continue
        free_gib = usage.free / 1024**3
        status = "ok" if free_gib >= min_free_gib else "error"
        checks.append(
            result(
                "disk",
                status,
                f"{free_gib:.1f} GiB free for {requested}",
                path=str(requested.resolve(strict=False)),
                measured_path=str(measured),
                free_gib=round(free_gib, 3),
                minimum_free_gib=min_free_gib,
            )
        )
    return checks


def heartbeat_candidates(run_dir: Path, staging_dir: Path, log_path: Path) -> list[Path]:
    candidates = [log_path, run_dir / "scan_scope.json"]
    product_dir = run_dir / "_per_pilot"
    if product_dir.is_dir():
        candidates.extend(product_dir.glob("*.npz"))
    if staging_dir.is_dir():
        candidates.extend(path for path in staging_dir.rglob("*") if path.is_file())
    return [path for path in candidates if path.is_file()]


def heartbeat_check(
    run_dir: Path,
    staging_dir: Path,
    log_path: Path,
    stale_minutes: float,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    candidates = heartbeat_candidates(run_dir, staging_dir, log_path)
    if not candidates:
        return result("heartbeat", "error", "no run heartbeat file was found")
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    newest_mtime = newest.stat().st_mtime
    age_minutes = ((time.time() if now is None else now) - newest_mtime) / 60.0
    status = "ok" if age_minutes <= stale_minutes else "error"
    return result(
        "heartbeat",
        status,
        f"newest activity is {age_minutes:.1f} minutes old: {newest}",
        newest_path=str(newest.resolve()),
        newest_utc=dt.datetime.fromtimestamp(newest_mtime, dt.timezone.utc).isoformat(),
        age_minutes=round(age_minutes, 3),
        stale_minutes=stale_minutes,
    )


def gpu_check(
    min_free_mib: float,
    max_temperature_c: float,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    query = command or (
        "nvidia-smi",
        "--query-gpu=index,name,temperature.gpu,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        completed = subprocess.run(
            list(query),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return result("gpu", "error", f"GPU status failed: {exc}")
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        return result("gpu", "error", f"GPU status failed: {message}")
    devices: list[dict[str, Any]] = []
    try:
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 7:
                raise ValueError(f"unexpected row: {line}")
            devices.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "temperature_c": float(fields[2]),
                    "memory_total_mib": float(fields[3]),
                    "memory_used_mib": float(fields[4]),
                    "memory_free_mib": float(fields[5]),
                    "utilization_percent": float(fields[6]),
                }
            )
    except ValueError as exc:
        return result("gpu", "error", f"cannot parse GPU status: {exc}")
    if not devices:
        return result("gpu", "error", "no GPU was reported")
    failing = [
        device
        for device in devices
        if device["memory_free_mib"] < min_free_mib
        or device["temperature_c"] > max_temperature_c
    ]
    status = "error" if failing else "ok"
    summary = "; ".join(
        f"GPU {device['index']}: {device['temperature_c']:.0f} C, "
        f"{device['memory_free_mib']:.0f} MiB free, "
        f"{device['utilization_percent']:.0f}% use"
        for device in devices
    )
    return result(
        "gpu",
        status,
        summary,
        devices=devices,
        minimum_free_mib=min_free_mib,
        maximum_temperature_c=max_temperature_c,
    )


def certificate_check(
    path: Path,
    min_hours: float,
    *,
    command: Sequence[str] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        return result("certificate", "error", f"certificate is missing or unsafe: {path}")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        return result(
            "certificate",
            "error",
            f"certificate mode is {mode:04o}, expected 0600: {path}",
            path=str(path.resolve()),
            mode=f"{mode:04o}",
        )
    query = command or (
        "openssl",
        "x509",
        "-in",
        str(path),
        "-noout",
        "-enddate",
    )
    try:
        completed = subprocess.run(
            list(query),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return result("certificate", "error", f"certificate check failed: {exc}")
    if completed.returncode:
        message = completed.stderr.strip() or completed.stdout.strip()
        return result("certificate", "error", f"certificate check failed: {message}")
    line = next(
        (item.strip() for item in completed.stdout.splitlines() if item.startswith("notAfter=")),
        None,
    )
    if line is None:
        return result("certificate", "error", "certificate expiry was not reported")
    try:
        normalized = " ".join(line.partition("=")[2].split())
        expiry = dt.datetime.strptime(
            normalized, "%b %d %H:%M:%S %Y %Z"
        ).replace(tzinfo=dt.timezone.utc)
    except ValueError as exc:
        return result("certificate", "error", f"cannot parse certificate expiry: {exc}")
    remaining_hours = (expiry.timestamp() - (time.time() if now is None else now)) / 3600.0
    status = "ok" if remaining_hours >= min_hours else "error"
    return result(
        "certificate",
        status,
        f"certificate has {remaining_hours:.1f} hours remaining",
        path=str(path.resolve()),
        expires_utc=expiry.isoformat(),
        remaining_hours=round(remaining_hours, 3),
        minimum_remaining_hours=min_hours,
        mode="0600",
    )


def read_tail(path: Path, limit: int) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - limit))
        return stream.read().decode("utf-8", "replace")


def log_check(path: Path, tail_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        return result("log", "error", f"log file is missing: {path}")
    try:
        text = read_tail(path, tail_bytes)
    except OSError as exc:
        return result("log", "error", f"cannot read log file: {exc}")
    matches: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in LOG_PATTERNS):
            matches.append(line[-500:])
    matches = matches[-20:]
    if matches:
        return result(
            "log",
            "error",
            f"found {len(matches)} concerning line(s) in the log tail",
            matches=matches,
            tail_bytes=tail_bytes,
        )
    return result("log", "ok", "no concerning pattern in the log tail", tail_bytes=tail_bytes)


def collect(arguments: argparse.Namespace) -> dict[str, Any]:
    checks = [process_check(arguments.expect_running)]
    checks.extend(
        disk_checks((arguments.run_dir, arguments.staging_dir), arguments.min_free_gib)
    )
    checks.append(
        heartbeat_check(
            arguments.run_dir,
            arguments.staging_dir,
            arguments.log,
            arguments.stale_minutes,
        )
    )
    checks.append(gpu_check(arguments.min_gpu_free_mib, arguments.max_gpu_temperature_c))
    checks.append(
        certificate_check(arguments.certificate, arguments.min_certificate_hours)
    )
    checks.append(log_check(arguments.log, arguments.log_tail_bytes))
    return {
        "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ok": not any(item["status"] == "error" for item in checks),
        "checks": checks,
    }


def parser() -> argparse.ArgumentParser:
    result_parser = argparse.ArgumentParser(description=__doc__)
    result_parser.add_argument("--run-dir", type=Path, required=True)
    result_parser.add_argument("--staging-dir", type=Path, required=True)
    result_parser.add_argument("--log", type=Path, required=True)
    result_parser.add_argument("--expect-running", action="store_true")
    result_parser.add_argument("--min-free-gib", type=float, default=100.0)
    result_parser.add_argument("--stale-minutes", type=float, default=120.0)
    result_parser.add_argument("--min-gpu-free-mib", type=float, default=2500.0)
    result_parser.add_argument("--max-gpu-temperature-c", type=float, default=85.0)
    result_parser.add_argument(
        "--certificate", type=Path, default=Path("~/.ssl/cadcproxy.pem").expanduser()
    )
    result_parser.add_argument("--min-certificate-hours", type=float, default=72.0)
    result_parser.add_argument("--log-tail-bytes", type=int, default=1024 * 1024)
    result_parser.add_argument("--json", action="store_true")
    return result_parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if any(
        value < 0
        for value in (
            arguments.min_free_gib,
            arguments.stale_minutes,
            arguments.min_gpu_free_mib,
            arguments.max_gpu_temperature_c,
            arguments.min_certificate_hours,
        )
    ) or arguments.log_tail_bytes <= 0:
        parser().error("thresholds must be nonnegative and log-tail-bytes must be positive")
    report = collect(arguments)
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for check in report["checks"]:
            print(f"[{check['status'].upper()}] {check['check']}: {check['message']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
