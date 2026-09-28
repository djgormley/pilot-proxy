from datetime import datetime,timezone
import io
import zipfile

import numpy as np
import pytest

from pilot_proxy.characterization import histogram_frames as m



def t(year,month):return datetime(year,month,15,tzinfo=timezone.utc).timestamp()


def test_membership_includes_sparse_interior_month_and_untimed_acquisition_fallback():
    eras=[{"era":1,"first_month":"2020-01","last_month":"2020-03","months":["2020-01","2020-03"]},
          {"era":2,"first_month":"2020-06","last_month":"2020-07"}]
    time=np.array([t(2020,1),t(2020,2),t(2020,4),np.nan,t(2020,7)])
    unit=np.array([t(2020,1),t(2020,2),t(2020,4),t(2020,6),t(2020,7)])
    frame_month,member_month,ids,current=m.memberships(time,unit,eras,2)
    np.testing.assert_array_equal(ids,[1,1,-1,2,2])
    np.testing.assert_array_equal(current,[False,False,False,True,True])
    assert frame_month[3]==-1 and member_month[3]==2020*12+5
    eligible=np.isfinite(time)&np.isfinite(np.array([1,0,1,1,1]))
    assert eligible[1] and not eligible[3]


def test_overlapping_saved_eras_refused():
    with pytest.raises(ValueError,match="overlap"):
        m.memberships(np.array([t(2020,2)]),np.array([t(2020,2)]),
            [{"era":1,"first_month":"2020-01","last_month":"2020-03"},{"era":2,"first_month":"2020-02","last_month":"2020-04"}],2)


def fixture_product(fine):
    return {"frame_index":np.arange(4),"frame_unit_index":np.array([0,0,1,1]),"frame_in_unit":np.array([0,1,0,1]),
        "source_event_keys":np.array(["a","b"]),"unit_order":np.array(["a","b"]),
        "valid":np.array([1,1,0,1],dtype=np.uint8),"reject_mask":np.array([0,0,0,1],dtype=np.uint8),
        "p_target_u64":np.array([1,2,0,3],dtype=np.uint64),"p_ref_sum_u64":np.array([2,2,0,2],dtype=np.uint64),
        "baseband_power_linear":np.array([1,128,0,2.]),"coarse_power_ratio":np.array([1,2,np.nan,3]),"fine_power_u64":fine}


@pytest.mark.parametrize("fine_value",[0,2**64-1])
def test_header_only_gate_matches_original_without_decoding_fine_values(tmp_path,fine_value):
    fine=np.full((4,3,256),fine_value,dtype=np.uint64)
    data=fixture_product(fine);path=tmp_path/"product.npz";np.savez_compressed(path,**data)
    expected=m.evaluate_frame_health(data)
    with np.load(path,allow_pickle=False) as archive:
        proxy=m.HeaderFineArchive(archive,path)
        actual=m.evaluate_frame_health(proxy)
        np.testing.assert_array_equal(actual.include,expected.include)
        assert actual.reason_counts==expected.reason_counts
        assert "fine_power_u64" not in proxy.read_fields
        assert proxy.fine_metadata.strides==(0,0,0) and not proxy.fine_metadata.flags.writeable
        with pytest.raises(RuntimeError):proxy["psd_frame_db_i16"]


def test_header_payload_length_and_unsigned_requirement(tmp_path):
    path=tmp_path/"truncated.npz";blob=io.BytesIO();np.save(blob,np.ones((4,3,256),np.uint64))
    with zipfile.ZipFile(path,"w") as archive:archive.writestr("fine_power_u64.npy",blob.getvalue()[:-8])
    with pytest.raises(ValueError,match="byte length"):m.member_header(path,"fine_power_u64")
    signed=tmp_path/"signed.npz";np.savez_compressed(signed,fine_power_u64=np.ones((4,3,256),np.int64))
    with np.load(signed) as archive:
        with pytest.raises(ValueError,match="unsigned"):m.HeaderFineArchive(archive,signed)
