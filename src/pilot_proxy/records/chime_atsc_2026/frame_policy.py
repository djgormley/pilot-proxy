#!/usr/bin/env python3
"""Replay frozen coarse frame policies on timestamp-matched capture correlations."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

NFFT = 16384
DT = 2.56e-6
PILOT = dict(zip(range(14, 37), [844, 829, 813, 798, 783, 767, 752, 736,
    721, 706, 690, 675, 660, 644, 629, 614, 598, 583, 568, 552, 537, 521, 506]))
CLASSES = [(0, 1), (0, 8), (0, 32), (0, 64), (0, 128), (0, 255),
           (1, 0), (1, 32), (2, 0), (3, 0)]
POLICIES = ('cal_q0.1', 'cal_q0.5', 'cal_q0.9', 'keep_all')


class Refusal(ValueError):
    """Input cannot support this comparison."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(arrays):
    h = hashlib.sha256()
    for name, a in sorted(arrays.items()):
        a = np.ascontiguousarray(a)
        h.update(name.encode() + a.dtype.str.encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def unique_ids(ids):
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(set(map(int, ids))) != ids.size:
        raise Refusal('frame identities must be unique integers')
    return ids


def align_ids(visibility_ids, detector_ids):
    """Return detector rows for each visibility identity; never use row position."""
    visibility_ids, detector_ids = unique_ids(visibility_ids), unique_ids(detector_ids)
    lookup = {int(t): i for i, t in enumerate(detector_ids)}
    missing = [int(t) for t in visibility_ids if int(t) not in lookup]
    if missing:
        raise Refusal(f'{len(missing)} visibility frames lack matching detector identities')
    return np.array([lookup[int(t)] for t in visibility_ids], dtype=int)


def exact_keep(target, reference, valid, target_norm, reference_norm, ratio):
    """Keep Q <= eta using integer arithmetic, including the equality boundary."""
    target, reference, valid = map(np.asarray, (target, reference, valid))
    if target.shape != reference.shape or target.shape != valid.shape:
        raise Refusal('power and validity shapes differ')
    if target.dtype.kind not in 'iu' or reference.dtype.kind not in 'iu':
        raise Refusal('detector powers must be integers')
    if target_norm <= 0 or reference_norm <= 0:
        raise Refusal('invalid weight norms')
    num, den = map(int, ratio)
    if num < 0 or den <= 0:
        raise Refusal('invalid threshold ratio')
    return np.array([bool(v) and int(r) > 0 and
                     int(t) * int(reference_norm) * den <= int(r) * int(target_norm) * num
                     for t, r, v in zip(target, reference, valid)], dtype=bool)


def complex_stats(values, fpga):
    """Descriptive moments; the sample covariance is not the covariance of a mean."""
    values = np.asarray(values, dtype=np.complex128)
    fpga = unique_ids(np.asarray(fpga))
    if values.shape != fpga.shape or not np.isfinite(values).all():
        raise Refusal('invalid complex samples')
    n = values.size
    if n == 0:
        return {k: None for k in ('mean_real', 'mean_imag', 'mean_abs', 'var_real',
            'var_imag', 'cov_real_imag', 'cross_frame_real_product', 'lag1_cov_real', 'lag1_cov_imag')} | {'lag1_pairs': 0}
    mean = values.mean()
    out = dict(mean_real=float(mean.real), mean_imag=float(mean.imag), mean_abs=float(abs(mean)),
               var_real=None, var_imag=None, cov_real_imag=None, cross_frame_real_product=None,
               lag1_cov_real=None, lag1_cov_imag=None, lag1_pairs=0)
    if n > 1:
        cov = np.cov(np.array([values.real, values.imag]), ddof=1)
        out.update(var_real=float(cov[0, 0]), var_imag=float(cov[1, 1]), cov_real_imag=float(cov[0, 1]),
                   cross_frame_real_product=float((abs(values.sum())**2 - np.vdot(values, values).real) / (n * (n - 1))))
        order = np.argsort(fpga)
        centered = values[order] - mean
        pairs = np.diff(fpga[order]) == NFFT
        out['lag1_pairs'] = int(pairs.sum())
        if pairs.any():
            lag = np.mean(centered[1:][pairs] * centered[:-1][pairs].conj())
            out.update(lag1_cov_real=float(lag.real), lag1_cov_imag=float(lag.imag))
    return out


def allocation_bins(channel):
    low, high = 470 + 6 * (channel - 14), 476 + 6 * (channel - 14)
    return {fid: max(0., min(high, 800 - fid * .390625 + .1953125)
                     - max(low, 800 - fid * .390625 - .1953125))
            for fid in range(1024)
            if 800 - fid * .390625 + .1953125 > low and 800 - fid * .390625 - .1953125 < high}


def threshold_rows(path, channel):
    d = json.loads(path.read_text())
    if d.get('channel') != channel:
        raise Refusal('threshold channel mismatch')
    rows = {r['policy']: r for r in d['policies']}
    if set(rows) != set(POLICIES):
        raise Refusal('incomplete policy ladder')
    for name, r in rows.items():
        if name != 'keep_all':
            if r.get('policy_error') or r.get('eta_integer_ratio') is None:
                raise Refusal(f'{name}: frozen threshold unavailable')
            if tuple(float(r['eta']).as_integer_ratio()) != tuple(r['eta_integer_ratio']):
                raise Refusal('threshold ratio differs from frozen eta')
    return rows


def read_detector(run_dir, channel, archive_dir, provenance, *, expected_bank=None):
    config_path, manifest_path = run_dir / 'run_config.json', run_dir / 'input_manifest.json'
    config, manifest = json.loads(config_path.read_text()), json.loads(manifest_path.read_text())
    if config.get('frame_size_samples') != NFFT or config.get('absolute_time_used') is not False:
        raise Refusal('unsupported detector timing contract')
    if sha(manifest_path) != config['input_manifest_sha256']:
        raise Refusal('input manifest digest mismatch')
    datasets = [d for d in manifest['datasets'] if d['physical_channel'] == channel]
    if len(datasets) != 1 or len(datasets[0]['segments']) != 1:
        raise Refusal('missing pilot or unsupported segmented input')
    dataset = datasets[0]
    segment = dataset['segments'][0]
    raw = Path(segment['path'])
    with h5py.File(raw, 'r') as h:
        required = ('time0_fpga_count', 'time0_ctime', 'delta_time', 'freq_id', 'event_id')
        if any(k not in h.attrs for k in required):
            raise Refusal('raw input lacks timing/frequency metadata')
        attrs = {k: h.attrs[k].item() for k in required}
        shape = h[dataset['dataset_path']].shape
        input_map_digest = array_digest({'input_map': h['index_map/input'][:]})
    if shape != tuple(segment['shape']) or attrs['freq_id'] != PILOT[channel]:
        raise Refusal('raw input shape or frequency differs from manifest')
    if attrs['delta_time'] != DT or dataset['num_input_streams'] != 2048:
        raise Refusal('unsupported sample timing or input geometry')
    product_path = run_dir / 'chime_detector_outputs.npz'
    names = ('physical_channel', 'frame_index', 'p_target_u64', 'p_ref_sum_u64', 'valid',
             'target_norm_sq', 'reference_norm_sum_sq', 'pilot_frequency_hz', 'chime_frequency_hz')
    with np.load(product_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in names}
    channels = arrays['physical_channel']
    ix = np.flatnonzero(channels == channel)
    if ix.size != 1:
        raise Refusal('missing or duplicate detector channel')
    j = int(ix[0])
    indices = unique_ids(arrays['frame_index'])
    if not np.array_equal(indices, np.arange(indices.size)) or (indices.size * NFFT > shape[0]):
        raise Refusal('detector frame indices exceed manifest input')
    nt, nr = int(arrays['target_norm_sq'][j]), int(arrays['reference_norm_sum_sq'][j])
    bank_fields = ('physical_channel', 'pilot_frequency_hz', 'target_norm_sq',
                   'reference_norm_sum_sq', 'weight_bank_sha256', 'nfft')
    if expected_bank is None:
        archive_path = archive_dir / f'{PILOT[channel]}.npz'
        with np.load(archive_path, allow_pickle=False) as z:
            archive = {k: z[k] for k in bank_fields}
        bank = {k: a.item() for k, a in archive.items()}
        bank_source = dict(archive_path=str(archive_path),
                           archive_consumed_fields_sha256=array_digest(archive))
    else:
        bank = {k: expected_bank[k] for k in bank_fields}
        bank_source = dict(expected_bank_contract=bank)
    if (bank['physical_channel'] != channel or bank['nfft'] != NFFT
        or bank['pilot_frequency_hz'] != float(arrays['pilot_frequency_hz'][j])
        or nt != bank['target_norm_sq'] or nr != bank['reference_norm_sum_sq']
        or bank['weight_bank_sha256'] != config['weights_sha256']):
        raise Refusal('capture bank differs from expected bank contract')
    if float(arrays['chime_frequency_hz'][j]) != (800 - PILOT[channel] * .390625) * 1e6:
        raise Refusal('detector coarse frequency mismatch')
    entry = dict(run=str(run_dir), channel=channel, config_sha256=sha(config_path),
        input_manifest_sha256=sha(manifest_path), detector_sha256=sha(product_path),
        raw_path=str(raw), raw_header=attrs, raw_shape=list(shape), input_map_sha256=input_map_digest,
        raw_payload_rehashed=False, **bank_source)
    provenance.append(entry)
    return dict(fpga=attrs['time0_fpga_count'] + indices * NFFT, epoch=attrs['event_id'],
                target=arrays['p_target_u64'][:, j], reference=arrays['p_ref_sum_u64'][:, j],
                valid=arrays['valid'][:, j].astype(bool), nt=nt, nr=nr,
                pilot_frequency_hz=float(arrays['pilot_frequency_hz'][j]), source=entry)


def read_visibility(path, channel, provenance, *, expected_epoch, detector=None):
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in ('meta', 'frame_fpga0', 'keys', 'count', 'stacks', 'autos')}
    m = json.loads(str(a['meta']))
    ids = unique_ids(a['frame_fpga0'])
    fid = int(path.stem)
    if m.get('file') != f'baseband_{expected_epoch}_{fid}.h5':
        raise Refusal('visibility source event differs from capture directory')
    if not isinstance(m.get('time0_ctime'), (int, float)) or not np.isfinite(m['time0_ctime']):
        raise Refusal('visibility lacks a finite UTC origin')
    if detector is not None:
        raw = detector['source']['raw_header']
        if int(raw['event_id']) != int(expected_epoch):
            raise Refusal('detector event differs from capture directory')
        utc_offset = m['time0_ctime'] - raw['time0_ctime']
        fpga_offset = (m['time0_fpga'] - raw['time0_fpga_count']) * DT
        # Allow UTC rounding below one 2.56-microsecond sample.
        if not np.isfinite(utc_offset) or abs(utc_offset - fpga_offset) > 1e-6:
            raise Refusal('visibility and detector UTC/FPGA origins disagree')
    if (m['nfft'] != NFFT or m['delta_time'] != DT or m['freq_id'] != fid
        or abs(m['freq_mhz'] - (800 - fid * .390625)) > 1e-9 or fid not in allocation_bins(channel)):
        raise Refusal('visibility timing, frequency, or allocation metadata mismatch')
    if a['stacks'].shape != (ids.size, a['keys'].shape[0]) or a['autos'].shape != (ids.size, 2048):
        raise Refusal('visibility shape mismatch')
    if (m['n_frames'] != ids.size or not np.array_equal(ids, m['grid_fpga'] + np.arange(ids.size) * NFFT)
        or m['grid_fpga'] != m['time0_fpga'] + m['j0']):
        raise Refusal('visibility frame identities differ from reducer metadata')
    provenance.append(dict(path=str(path), consumed_arrays_sha256=array_digest(a),
                           file_bytes=path.stat().st_size, full_file_rehashed=False))
    indices = []
    labels = []
    for ew, ns in CLASSES:
        for pol in (0, 1):
            ix = np.flatnonzero(np.all(a['keys'] == [ew, ns, pol, pol], axis=1))
            if ix.size != 1:
                raise Refusal('missing or duplicate requested baseline class')
            indices.append(int(ix[0])); labels.append((ew, ns, pol))
    if not np.isfinite(a['autos']).all() or np.any(a['autos'] < 0):
        raise Refusal('invalid total input powers')
    live = a['autos'].mean(axis=0) > .5
    if not live.any():
        raise Refusal('no live inputs for diagnostic normalization')
    return dict(ids=ids, values=a['stacks'][:, indices].astype(np.complex128) / NFFT,
                total_power=a['autos'][:, live].mean(axis=1), labels=labels,
                multiplicity=a['count'][indices], n_live=int(live.sum()), meta=m)


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def run(root, output, channels):
    output.mkdir(parents=True, exist_ok=False)
    capture = root / 'output/channel-ruling-execution-2026-09-14/rebuild/author_actions/capture-runbook/reduce'
    coarse = root / 'results/archive_author_eras_2026-09-23/coarse'
    archive = root / 'products/chime_pilots_rebuild_20260829/products/_per_pilot'
    decisions, moments, coverage, refusals, provenance, threshold_provenance = [], [], [], [], [], []
    summaries = {}
    for channel in channels:
        try:
            path = coarse / f'ch{channel}.json'
            thresholds = threshold_rows(path, channel)
            threshold_provenance.append(dict(channel=channel, path=str(path), sha256=sha(path),
                policies=[{k: r.get(k) for k in ('policy', 'eta', 'eta_integer_ratio')} for r in thresholds.values()]))
        except (Refusal, OSError, KeyError) as exc:
            thresholds = None
            refusals.append(dict(channel=channel, epoch='', freq_id='', reason=str(exc)))
        for product_dir in sorted((root / 'datasets').glob('pilot_reduce_*')):
            epoch = product_dir.name.removeprefix('pilot_reduce_')
            try:
                detector = read_detector(capture / f'kernel_{epoch}_k230', channel, archive, provenance)
                if detector['epoch'] != int(epoch):
                    raise Refusal('raw event identity differs from capture directory')
                masks = {'keep_all': np.ones(detector['fpga'].size, bool)}
                if thresholds:
                    for name in POLICIES[:-1]:
                        masks[name] = exact_keep(detector['target'], detector['reference'], detector['valid'],
                                                detector['nt'], detector['nr'], thresholds[name]['eta_integer_ratio'])
                for i, fpga in enumerate(detector['fpga']):
                    row = dict(channel=channel, epoch=epoch, fpga0=int(fpga), detector_valid=bool(detector['valid'][i]),
                        q=float(int(detector['target'][i]) * detector['nr'] /
                                (int(detector['reference'][i]) * detector['nt'])) if detector['reference'][i] else None,
                        bank='nominal_archive', pilot_frequency_hz=detector['pilot_frequency_hz'])
                    row.update({name: bool(masks[name][i]) if name in masks else None for name in POLICIES})
                    decisions.append(row)
                for name, mask in masks.items():
                    s = summaries.setdefault((channel, name), dict(frames=0, kept=0, epochs=0, mixed_epochs=0))
                    s['frames'] += mask.size; s['kept'] += int(mask.sum()); s['epochs'] += 1
                    s['mixed_epochs'] += int(0 < mask.sum() < mask.size)
            except (Refusal, OSError, KeyError) as exc:
                detector = None
                refusals.append(dict(channel=channel, epoch=epoch, freq_id=PILOT[channel], reason=str(exc)))
            expected = allocation_bins(channel)
            present = [fid for fid in expected if (product_dir / f'{fid}.npz').exists()]
            validated = []
            for fid in present:
                try:
                    v = read_visibility(product_dir / f'{fid}.npz', channel, provenance,
                                        expected_epoch=epoch, detector=detector)
                    bin_moments = []
                    selections = {'keep_all': np.ones(v['ids'].size, bool)}
                    if detector and thresholds:
                        ix = align_ids(v['ids'], detector['fpga'])
                        selections.update({name: mask[ix] for name, mask in masks.items() if name != 'keep_all'})
                    for name, mask in selections.items():
                        n = int(mask.sum())
                        power = float(v['total_power'][mask].mean()) if n else None
                        for j, (ew, ns, pol) in enumerate(v['labels']):
                            st = complex_stats(v['values'][mask, j], v['ids'][mask])
                            row = dict(channel=channel, epoch=epoch, freq_id=fid, ew=ew, ns=ns, pol=pol, policy=name,
                                available_frames=mask.size, retained_frames=n, nominal_retained_samples=n * NFFT,
                                retained_exposure_s=n * NFFT * DT, available_contiguous_exposure_s=mask.size * NFFT * DT,
                                redundant_product_multiplicity=int(v['multiplicity'][j]), joint_valid_samples=None,
                                selected_live_total_power=power, live_inputs=v['n_live'],
                                normalized_mean_abs=st['mean_abs'] / power if power else None)
                            row.update(st)
                            bin_moments.append(row)
                    moments.extend(bin_moments)
                    validated.append(fid)
                except (Refusal, OSError, KeyError) as exc:
                    refusals.append(dict(channel=channel, epoch=epoch, freq_id=fid, reason=str(exc)))
            coverage.append(dict(channel=channel, epoch=epoch, expected_bins=len(expected),
                present_bins=len(present), available_bins=len(validated),
                available_bandwidth_mhz=sum(expected[fid] for fid in validated),
                absent_freq_ids=';'.join(str(fid) for fid in expected if fid not in present),
                refused_freq_ids=';'.join(str(fid) for fid in present if fid not in validated),
                missing_freq_ids=';'.join(str(fid) for fid in expected if fid not in validated),
                pilot_policy_available=bool(detector and thresholds and validated),
                complete_allocation_coverage=len(validated) == len(expected)))
        print(f'channel {channel}: {len(moments)} moment rows', flush=True)
    policy_summary = [dict(channel=c, policy=p, **s, retained_fraction=s['kept'] / s['frames'],
        nominal_uniform_information_time_multiplier=s['frames'] / s['kept'] if s['kept'] else None,
        observing_time_validated=False) for (c, p), s in sorted(summaries.items())]
    for filename, rows in [('frame_decisions.csv', decisions), ('complex_moments.csv', moments),
                           ('coverage.csv', coverage), ('refusals.csv', refusals), ('policy_summary.csv', policy_summary)]:
        csv_write(output / filename, rows)
    receipt = dict(schema='capture_frame_policy_replay_v1', channels=channels, frame_samples=NFFT,
        sample_seconds=DT, endpoint='short recorded captures; no observed 10-second product',
        source_sha256=sha(__file__), thresholds=threshold_provenance, inputs=provenance,
        corrected_ch33_bank='not combined; nominal replay only, invalid as a universal channel-33 detector',
        mean_units='quantized voltage-product units per nominal sample, averaged over redundant products',
        covariance_units='squared mean units; descriptive sample covariance within each retained capture',
        normalization='mean total power over inputs with all-frame mean power > 0.5; not thermal-only power',
        uncertainty_coverage_validated=False, scientific_rulings=False, oracle_bound=False,
        limitations=['No individual-product packet validity, gain/delay calibration, or fringestopping.',
            'No arbitrary per-input remasking after redundant stacking.',
            'Archive calibration thresholds are frozen; present captures are development/diagnostic data.',
            'Sample covariance is not a temporal covariance model, confidence bound, or Fisher residual.',
            'No downstream suppression or phasor-angle attenuation credit.',
            'Raw HDF5 payloads and unused large NPZ members were not rehashed.'],
        outputs={p.name: sha(p) for p in output.glob('*.csv')})
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--channels', type=int, nargs='+', default=list(range(14, 37)))
    args = parser.parse_args()
    if not set(args.channels).issubset(PILOT):
        parser.error('channels must be in 14..36')
    run(args.root, args.output, args.channels)
