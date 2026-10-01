"""Regression checks for frame selection and the limits of aggregate diagnostics."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np

from pilot_proxy.records.chime_atsc_2026.frame_policy import (CLASSES, DT, NFFT, POLICIES, Refusal, align_ids, allocation_bins, complex_stats,
                             exact_keep, threshold_rows, read_detector, read_visibility, run, sha)


class FramePolicyTests(unittest.TestCase):
    def test_alignment_uses_id_not_position(self):
        np.testing.assert_array_equal(align_ids(np.array([100, 200]), np.array([200, 100])), [1, 0])

    def test_missing_and_duplicate_id_refuse(self):
        for a, b in [([1, 2], [1]), ([1, 1], [1, 2]), ([1], [1, 1])]:
            with self.assertRaises(Refusal):
                align_ids(np.array(a), np.array(b))

    def test_exact_threshold_boundary_and_uint64_scale(self):
        t = np.array([2**63, 2**63 + 1, 0, 1], dtype=np.uint64)
        r = np.array([2**63, 2**63, 0, 1], dtype=np.uint64)
        np.testing.assert_array_equal(exact_keep(t, r, [True, True, True, False], 7, 7, (1, 1)),
                                      [True, False, False, False])

    def test_quiet_frames_beat_every_identical_dump(self):
        data = np.array([1, 1, 10, 10], dtype=complex)
        ids = np.arange(4, dtype=np.int64) * NFFT
        whole = complex_stats(data, ids)
        quiet = complex_stats(data[:2], ids[:2])
        self.assertAlmostEqual(whole['mean_real'], 5.5)
        self.assertAlmostEqual(quiet['mean_real'], 1.)
        self.assertLess(quiet['cross_frame_real_product'], whole['cross_frame_real_product'])

    def test_constant_phase_is_not_zero_residual_after_mean_removal(self):
        v = np.array([1, 2, 3, 4]) * np.exp(.4j)
        st = complex_stats(v, np.arange(4, dtype=np.int64) * NFFT)
        self.assertGreater(st['var_real'] + st['var_imag'], 0)
        self.assertAlmostEqual(np.angle(v[0] * v[-1].conjugate()), 0)

    def test_lag_one_uses_actual_spacing(self):
        st = complex_stats(np.array([1, 2, 3], complex), np.array([0, NFFT, 3*NFFT]))
        self.assertEqual(st['lag1_pairs'], 1)

    def test_allocation_edges_sum_to_six_mhz(self):
        for ch in range(14, 37):
            self.assertAlmostEqual(sum(allocation_bins(ch).values()), 6.)
        self.assertGreater(len(allocation_bins(30)), 1)

    def test_empty_policy_has_no_fake_zero_residual(self):
        self.assertIsNone(complex_stats(np.array([], complex), np.array([], dtype=int))['mean_abs'])

    def test_missing_frozen_threshold_refuses(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'thresholds.json'
            p.write_text(json.dumps(dict(channel=29, policies=[dict(policy='keep_all')])))
            with self.assertRaises(Refusal):
                threshold_rows(p, 29)

    def test_missing_pilot_refuses_before_any_row_alignment(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            manifest = d / 'input_manifest.json'
            manifest.write_text(json.dumps(dict(datasets=[])))
            (d / 'run_config.json').write_text(json.dumps(dict(frame_size_samples=NFFT,
                absolute_time_used=False, input_manifest_sha256=sha(manifest))))
            with self.assertRaisesRegex(Refusal, 'missing pilot'):
                read_detector(d, 29, d, [])


class DetectorBankTests(unittest.TestCase):
    def test_explicit_bank_checks_all_fields_and_preserves_archive_default(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw = directory / 'raw.h5'
            with h5py.File(raw, 'w') as h:
                h.attrs.update(time0_fpga_count=100, time0_ctime=1., delta_time=DT,
                               freq_id=552, event_id=20260917040230)
                h.create_dataset('baseband', shape=(2*NFFT, 2048), dtype='uint8')
                h.create_dataset('index_map/input', data=np.arange(2048))
            manifest = directory / 'input_manifest.json'
            manifest.write_text(json.dumps(dict(datasets=[dict(physical_channel=33,
                dataset_path='baseband', num_input_streams=2048,
                segments=[dict(path=str(raw), shape=[2*NFFT, 2048])])])))
            (directory / 'run_config.json').write_text(json.dumps(dict(frame_size_samples=NFFT,
                absolute_time_used=False, input_manifest_sha256=sha(manifest), weights_sha256='corrected')))
            np.savez(directory / 'chime_detector_outputs.npz', physical_channel=[33],
                frame_index=np.arange(2), p_target_u64=np.ones((2, 1), np.uint64),
                p_ref_sum_u64=np.ones((2, 1), np.uint64), valid=np.ones((2, 1), bool),
                target_norm_sq=[7], reference_norm_sum_sq=[11], pilot_frequency_hz=[584309440.],
                chime_frequency_hz=[(800 - 552*.390625)*1e6])
            bank = dict(physical_channel=33, nfft=NFFT, pilot_frequency_hz=584309440.,
                        target_norm_sq=7, reference_norm_sum_sq=11, weight_bank_sha256='corrected')
            np.savez(directory / '552.npz', **(bank | dict(weight_bank_sha256='nominal')))
            with self.assertRaisesRegex(Refusal, 'bank contract'):
                read_detector(directory, 33, directory, [])
            provenance = []
            result = read_detector(directory, 33, directory, provenance, expected_bank=bank)
            self.assertEqual(result['nt'], 7)
            self.assertEqual(provenance[0]['expected_bank_contract'], bank)
            self.assertNotIn('archive_path', provenance[0])
            for field in bank:
                wrong = bank | {field: 'other' if field == 'weight_bank_sha256' else bank[field] + 1}
                with self.subTest(field=field), self.assertRaisesRegex(Refusal, 'bank contract'):
                    read_detector(directory, 33, directory, [], expected_bank=wrong)
            np.savez(directory / '552.npz', **bank)
            default = read_detector(directory, 33, directory, [])
            self.assertIn('archive_path', default['source'])
            self.assertNotIn('expected_bank_contract', default['source'])


class VisibilityValidationTests(unittest.TestCase):
    epoch = '20260917040230'
    origin = 155358203125
    utc = 1789617750.0

    def detector(self):
        return dict(epoch=int(self.epoch), fpga=self.origin + np.arange(2, dtype=np.int64) * NFFT,
                    target=np.ones(2, np.uint64), reference=np.ones(2, np.uint64),
                    valid=np.ones(2, bool), nt=1, nr=1, pilot_frequency_hz=596309441.,
                    source=dict(raw_header=dict(event_id=int(self.epoch),
                        time0_fpga_count=self.origin, time0_ctime=self.utc)))

    def visibility_file(self, directory, **overrides):
        path = directory / '521.npz'
        meta = dict(file=f'baseband_{self.epoch}_521.h5', time0_ctime=self.utc,
                    time0_fpga=self.origin, grid_fpga=self.origin, j0=0,
                    nfft=NFFT, delta_time=DT, freq_id=521, freq_mhz=800 - 521 * .390625,
                    n_frames=2)
        meta.update(overrides)
        keys = np.array([[ew, ns, pol, pol] for ew, ns in CLASSES for pol in (0, 1)])
        np.savez(path, meta=json.dumps(meta),
                 frame_fpga0=meta['grid_fpga'] + np.arange(2, dtype=np.int64) * NFFT,
                 keys=keys, count=np.ones(len(keys), int),
                 stacks=np.ones((2, len(keys)), complex) * NFFT,
                 autos=np.ones((2, 2048)))
        return path

    def test_source_event_is_required_even_without_pilot(self):
        with TemporaryDirectory() as tmp:
            path = self.visibility_file(Path(tmp), file='baseband_20200101000000_521.h5')
            with self.assertRaisesRegex(Refusal, 'source event'):
                read_visibility(path, 35, [], expected_epoch=self.epoch)

    def test_matching_fpga_ids_cannot_hide_a_utc_mismatch(self):
        with TemporaryDirectory() as tmp:
            path = self.visibility_file(Path(tmp), time0_ctime=self.utc + 1)
            with self.assertRaisesRegex(Refusal, 'UTC/FPGA'):
                read_visibility(path, 35, [], expected_epoch=self.epoch, detector=self.detector())

    def test_nonfinite_utc_is_refused_without_pilot(self):
        with TemporaryDirectory() as tmp:
            path = self.visibility_file(Path(tmp), time0_ctime=float('nan'))
            with self.assertRaisesRegex(Refusal, 'finite UTC'):
                read_visibility(path, 35, [], expected_epoch=self.epoch)

    def test_consistent_shifted_origin_is_allowed(self):
        with TemporaryDirectory() as tmp:
            path = self.visibility_file(Path(tmp), time0_ctime=self.utc + NFFT * DT,
                                        time0_fpga=self.origin + NFFT, grid_fpga=self.origin + NFFT)
            result = read_visibility(path, 35, [], expected_epoch=self.epoch, detector=self.detector())
            np.testing.assert_array_equal(result['ids'], self.origin + np.arange(1, 3) * NFFT)

    def replay(self, failure):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            products = root / 'datasets' / f'pilot_reduce_{self.epoch}'
            products.mkdir(parents=True)
            self.visibility_file(products)
            coarse = root / 'results/archive_author_eras_2026-09-23/coarse'
            coarse.mkdir(parents=True)
            (coarse / 'ch35.json').write_text('{}')
            thresholds = {p: dict(policy=p, eta=1., eta_integer_ratio=[1, 1]) for p in POLICIES}
            visibility = read_visibility(products / '521.npz', 35, [], expected_epoch=self.epoch)
            if failure == 'partial':
                visibility['values'][0, 1] = np.nan
            elif failure == 'join':
                visibility['ids'] = visibility['ids'] + NFFT
            with patch('pilot_proxy.records.chime_atsc_2026.frame_policy.threshold_rows', return_value=thresholds), \
                 patch('pilot_proxy.records.chime_atsc_2026.frame_policy.read_detector', return_value=self.detector()), \
                 patch('pilot_proxy.records.chime_atsc_2026.frame_policy.allocation_bins', return_value={521: 6.}), \
                 patch('pilot_proxy.records.chime_atsc_2026.frame_policy.read_visibility', return_value=visibility):
                run(root, root / 'release', [35])
            def rows(name):
                with (root / 'release' / name).open() as f:
                    return list(csv.DictReader(f))
            return rows('coverage.csv')[0], rows('complex_moments.csv'), rows('refusals.csv')

    def test_failed_join_does_not_count_as_coverage(self):
        coverage, moments, refusals = self.replay('join')
        self.assertEqual(coverage['present_bins'], '1')
        self.assertEqual(coverage['available_bins'], '0')
        self.assertEqual(coverage['available_bandwidth_mhz'], '0')
        self.assertEqual(coverage['refused_freq_ids'], '521')
        self.assertEqual(coverage['missing_freq_ids'], '521')
        self.assertEqual(coverage['complete_allocation_coverage'], 'False')
        self.assertEqual(coverage['pilot_policy_available'], 'False')
        self.assertFalse(moments)
        self.assertIn('matching detector identities', refusals[0]['reason'])

    def test_refused_bin_does_not_leave_partial_moments(self):
        coverage, moments, refusals = self.replay('partial')
        self.assertEqual(coverage['available_bins'], '0')
        self.assertFalse(moments)
        self.assertIn('invalid complex samples', refusals[0]['reason'])

    def test_validated_bin_contributes_coverage_and_all_moments(self):
        coverage, moments, refusals = self.replay(None)
        self.assertEqual(coverage['available_bins'], '1')
        self.assertEqual(coverage['available_bandwidth_mhz'], '6.0')
        self.assertEqual(coverage['complete_allocation_coverage'], 'True')
        self.assertEqual(coverage['pilot_policy_available'], 'True')
        self.assertEqual(coverage['refused_freq_ids'], '')
        self.assertEqual(len(moments), 4 * 2 * len(CLASSES))
        self.assertFalse(refusals)


if __name__ == '__main__':
    unittest.main()
