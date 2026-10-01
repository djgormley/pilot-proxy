# coding=utf-8
from __future__ import annotations

import pytest

from pilot_proxy.result_schema import (
    COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO,
    RESULT_SCHEMA_TOKEN,
    fixed_point_contract,
    result_layout,
    result_schema_object,
)

FRAME_SIZE_SAMPLES = 16_384
DETECTOR_WINDOW_SAMPLES = 128
NUM_INPUT_STREAMS = 2
WINDOWS_PER_STREAM = 128
DETECTOR_ROWS = 256
DTV_BANDWIDTH_HZ = 6_000_000.0
BIN_ENBW_HZ = 3_051.7578125
PILOT_BELOW_DATA_DB = 11.3
PILOT_CAPTURE_EFFICIENCY = 1.0
THRESHOLD = {
    "threshold_data_shelf_snr_db": -26.0,
    "threshold_pilot_excess_db": -4.364,
    "threshold_coarse_power_ratio": 1.366,
    "threshold_half_num": 123,
    "threshold_half_den": 180,
}


def test_result_layout_uses_all_rows_summed_convention() -> None:
    layout = result_layout(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
    )

    assert layout["windows_per_stream"] == WINDOWS_PER_STREAM
    assert layout["detector_rows_per_frame"] == DETECTOR_ROWS
    assert layout["combine_mode"] == COMBINE_MODE_ALL_ROWS_SUMMED_BEFORE_RATIO


def test_result_schema_object_contains_fixed_contract_and_threshold() -> None:
    schema = result_schema_object(
        frame_size_samples=FRAME_SIZE_SAMPLES,
        num_input_streams=NUM_INPUT_STREAMS,
        detector_window_samples=DETECTOR_WINDOW_SAMPLES,
        dtv_bandwidth_hz=DTV_BANDWIDTH_HZ,
        bin_enbw_hz=BIN_ENBW_HZ,
        pilot_below_data_db=PILOT_BELOW_DATA_DB,
        pilot_capture_efficiency=PILOT_CAPTURE_EFFICIENCY,
        threshold=THRESHOLD,
    )

    assert schema["schema_version"] == RESULT_SCHEMA_TOKEN
    assert schema["layout"]["num_input_streams"] == NUM_INPUT_STREAMS
    assert schema["calibration"]["pilot_below_data_db"] == PILOT_BELOW_DATA_DB
    assert schema["threshold"]["threshold_half_den"] == THRESHOLD[
        "threshold_half_den"
    ]
    assert schema["fixed_point_contract"]["detector_window_samples"] == (
        DETECTOR_WINDOW_SAMPLES
    )
    assert schema["fixed_point_contract"]["power_accumulator"] == "uint64"


@pytest.mark.parametrize("window", [64, 128])
def test_fixed_point_contract_describes_selected_supported_window(window) -> None:
    contract = fixed_point_contract(detector_window_samples=window)

    assert "K=128" in contract["detector_window_support_reason"]
    assert f"selects K={window}" in contract["detector_window_support_reason"]
    assert "k128_lock_reason" not in contract
    assert contract["supported_detector_window_samples"] == [64, 128]
    assert contract["packed_complex_bits"] == 8
    assert contract["sample_bits_per_component"] == 4
