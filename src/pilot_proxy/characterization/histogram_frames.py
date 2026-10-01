"""Export small coarse-frame fields against a characterization release's era and health records.

No era refitting; no PSD/fine payload decoding; no hardware access. Histograms
retain every finite Q on timed health-selected frames, including zero if present.

``--control-product`` exports one control band instead: a band the project
profile's frequency plan gives role ``control``, read at its declared
``target_freq_id``. It has no ledger, no dated era, no threshold, no tolerance
and no verdict; its archive is one period (``CONTROL_PERIOD_RULE``).

    pilot-proxy characterize histogram-frames --products DIR --release DIR --output DIR
    pilot-proxy characterize histogram-frames --control-product FILE [--profile DIR] --output DIR

``--release`` is the archive characterization whose ledger and saved eras the
frames are keyed to (``ledger/channels/ch*_fid*.json``,
``channels/chNN/eras.json``, ``products-preflight.json``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pilot_proxy import archive_health, archived_product_keys
from pilot_proxy.archive_health import evaluate_frame_health
from pilot_proxy.config.project import default_project
from pilot_proxy.products import reader
from pilot_proxy.products.reader import HEALTH_GATE_SCHEMA, NFFT, Product

from . import blocks, eras
from .blocks import month_index, month_label

# the code this export runs, recorded with its sha256 in the extraction plan
SOURCES=(Path(__file__).resolve(),Path(reader.__file__).resolve(),Path(blocks.__file__).resolve(),
         Path(archive_health.__file__).resolve(),Path(archived_product_keys.__file__).resolve())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def write_json(path,value):
    with Path(path).open("x") as stream:stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def member_header(path,name):
    """Read only the NPY header and verify advertised payload size, without decoding it."""
    with zipfile.ZipFile(path) as archive:
        info=archive.getinfo(name+".npy")
        with archive.open(info) as stream:
            version=np.lib.format.read_magic(stream)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
            else:raise ValueError(f"Unsupported NPY header version:{version}")
            header_bytes=stream.tell()
    if dtype.hasobject or header_bytes+math.prod(shape)*dtype.itemsize!=info.file_size:
        raise ValueError("fine member header does not match complete uncompressed byte length")
    return {"shape":list(shape),"dtype":dtype.str,"fortran_order":fortran,
            "header_bytes":header_bytes,"uncompressed_member_bytes":info.file_size,
            "compressed_member_bytes":info.compress_size}


class HeaderFineArchive(Mapping):
    """Cache small fields and expose ONLY fine metadata through a read-only zero-stride view.

    Current exact-uint health logic reads fine ndim/shape/dtype only. All uint
    bit patterns are finite/nonnegative. No returned fine value is a measurement
    or exported array; PSD and alternative fine-value reads are forbidden.
    """
    def __init__(self,archive,path):
        self.archive=archive;self.files=archive.files;self.cache={};self.read_fields=[]
        self.fine_header=member_header(path,"fine_power_u64")
        shape=tuple(self.fine_header["shape"]);dtype=np.dtype(self.fine_header["dtype"])
        if len(shape)!=3 or shape[0]<=0 or shape[2]<=0 or dtype.kind!="u":
            raise ValueError("fine unsigned-integer metadata required for header-only gate")
        self.fine_metadata=np.broadcast_to(np.zeros((),dtype=dtype),shape)
    def __getitem__(self,key):
        if key=="fine_power_u64":return self.fine_metadata
        if key.startswith("psd_") or key.startswith("fine_power") or key.startswith("integrated_spectrum"):
            raise RuntimeError(f"large/spectral member forbidden in coarse export:{key}")
        if key not in self.cache:
            self.cache[key]=self.archive[key];self.read_fields.append(key)
        return self.cache[key]
    def __iter__(self):return iter(self.files)
    def __len__(self):return len(self.files)
    def __contains__(self,key):return key in self.files
    def close(self):self.archive.close()


def month_number(label):
    stamp=datetime.strptime(label,"%Y-%m")
    return stamp.year*12+stamp.month-1


def memberships(frame_time,unit_time,eras,current):
    timed_month=month_index(frame_time)
    member_month=timed_month.copy();missing=member_month<0
    member_month[missing]=month_index(unit_time[missing])
    ids=np.full(len(frame_time),-1,dtype=np.int32)
    for era in eras:
        inside=(member_month>=month_number(era["first_month"]))&(member_month<=month_number(era["last_month"]))
        if np.any(inside&(ids!=-1)):raise ValueError("saved era intervals overlap")
        ids[inside]=int(era["era"])
    return timed_month,member_month,ids,ids==int(current)


def assert_count(name,observed,expected):
    if int(observed)!=int(expected):raise ValueError(f"corrected ledger mismatch:{name}: {observed} != {expected}")


def process_channel(product_path,ledger_path,eras_path,output,preflight):
    ledger=json.loads(ledger_path.read_text());era_record=json.loads(eras_path.read_text())
    sections=ledger["sections"];source_product=sections["product"];source_era=sections["era"]
    initial_stat=product_path.stat();digest=sha(product_path)
    if digest!=ledger["product_sha256"] or digest!=preflight["sha256"]:
        raise ValueError(f"product differs from corrected ledger/preflight:{product_path}")
    if initial_stat.st_size!=preflight["bytes"]:raise ValueError("product byte count differs from preflight")
    channel=int(ledger["channel"]);fid=int(ledger["freq_id"])
    if era_record["channel"]!=channel or era_record["freq_id"]!=fid or era_record["current_era"]!=source_era["current_era"]:
        raise ValueError("saved era identities differ from ledger")
    if era_record["config_digest"]!=source_era["config_digest"]:raise ValueError("saved era configuration differs")
    with Product(product_path,require_health=True) as product:
        proxy=HeaderFineArchive(product.archive,product_path);product._z=proxy
        view=product.view
        if view.physical_channel!=channel or view.freq_id!=fid:raise ValueError("product identity differs")
        if int(product.scalar("nfft"))!=NFFT:raise ValueError("unexpected frame geometry")
        # Mandatory original gate; exact uint fine metadata suffices for its two structural checks.
        health=evaluate_frame_health(proxy)
        valid=np.asarray(view.valid,dtype=bool);health_include=np.asarray(health.include,dtype=bool)
        selected=valid&health_include;Q=np.asarray(view.statistic,dtype=np.float64)
        frame_time=product.frame_time;unit_time=product.unit_time
        finite_time=np.isfinite(frame_time);finite_Q=np.isfinite(Q)
        hist=selected&finite_time&finite_Q
        timed_month,member_month,era_id,current=memberships(frame_time,unit_time,era_record["eras"],era_record["current_era"])
        unit_index=product.frame_unit_index
        unit_event_id=product.unit_event_id
        if unit_event_id.dtype.kind not in "iu":raise ValueError("acquisition IDs must preserve their integer type")
        counts={"total":len(Q),"valid":int(valid.sum()),"health_include":int(health_include.sum()),
            "selected":int(selected.sum()),"finite_time_selected":int((selected&finite_time).sum()),
            "histogram_eligible":int(hist.sum()),"nonpositive_histogram":int((hist&(Q<=0)).sum()),
            "current_selected":int((selected&current).sum()),"current_histogram_eligible":int((hist&current).sum()),
            "current_untimed_selected":int((selected&current&~finite_time).sum()),
            "unassigned_selected":int((selected&(era_id<0)).sum()),
            "unassigned_histogram_eligible":int((hist&(era_id<0)).sum()),
            "untimed_selected":int((selected&~finite_time).sum()),"nonfinite_Q_selected":int((selected&~finite_Q).sum())}
        for key,field in (("total","n_frames"),("valid","n_valid"),("selected","n_selected"),("untimed_selected","n_without_time")):
            assert_count(key,counts[key],source_product[field])
        for key,field in (("total","frames"),("valid","valid"),("selected","health_selected")):
            assert_count(key,counts[key],preflight[field])
        assert_count("current_selected",counts["current_selected"],source_era["current_frames"])
        assert_count("era frames_selected",counts["selected"],era_record["frames_selected"])
        assert_count("era frames_without_time",counts["untimed_selected"],era_record["frames_without_time"])
        if source_product["health_schema"]!=HEALTH_GATE_SCHEMA or preflight["health_schema"]!=HEALTH_GATE_SCHEMA:
            raise ValueError("corrected population used a different health gate")
        reasons={key:int(value) for key,value in health.reason_counts.items()}
        expected_reasons={item.split(":")[0]:int(item.split(":")[1]) for item in source_product.get("health_reasons","").split(";") if item}
        if reasons!=expected_reasons:raise ValueError("health reasons differ from corrected ledger")
        definitions=[]
        for record in era_record["eras"]:
            member=era_id==int(record["era"])
            n_selected=int((member&selected).sum());n_untimed=int((member&selected&~finite_time).sum())
            assert_count(f"era{record['era']} selected",n_selected,record["frames"])
            assert_count(f"era{record['era']} untimed",n_untimed,record["frames_without_time"])
            assert_count(f"era{record['era']} units",np.unique(unit_index[member&selected]).size,record["units"])
            if int(record["era"])==int(era_record["current_era"]):
                if record["first_month"]!=source_era["current_first_month"] or record["last_month"]!=source_era["current_last_month"]:
                    raise ValueError("current era boundaries differ from ledger")
            definitions.append({**record,"is_current":int(record["era"])==int(era_record["current_era"]),
                "exported_count_selected":n_selected,"exported_count_histogram_eligible":int((member&hist).sum()),
                "exported_count_untimed_selected":n_untimed})
        arrays={"Q":Q,"valid":valid,"health_include":health_include,"selected":selected,
            "finite_time":finite_time,"finite_Q":finite_Q,"positive_Q":finite_Q&(Q>0),
            "histogram_eligible":hist,"era_id":era_id,"is_current":current,"current_histogram_eligible":hist&current,
            "frame_index":product.frame_column("frame_index"),"frame_unit_index":unit_index,
            "frame_in_unit":product.frame_in_unit,"frame_time":frame_time,"unit_time":unit_time,
            "frame_month":timed_month,"membership_month":member_month,
            "acquisition_id":unit_event_id[unit_index],"unit_event_id":unit_event_id,
            "unit_keys":np.asarray(proxy["unit_keys"]),"source_event_keys":np.asarray(proxy["source_event_keys"]),
            "unit_time0_ctime":product.unit_time0,"unit_delta_time":product.unit_delta_time}
        geometry={name:int(product.scalar(name)) for name in ("nfft","detector_window_samples","num_input_streams","target_norm_sq","reference_norm_sum_sq")}
        geometry.update(mu0=product.mu0,sample_rate_hz=float(product.scalar("sample_rate_hz")),
                        pilot_frequency_hz=float(product.scalar("pilot_frequency_hz")),chime_frequency_hz=float(product.scalar("chime_frequency_hz")))
        final_stat=product_path.stat()
        if (initial_stat.st_size,initial_stat.st_mtime_ns)!=(final_stat.st_size,final_stat.st_mtime_ns):raise ValueError("product changed during extraction")
        output_path=output/f"ch{channel:02d}.npz"
        with output_path.open("xb") as stream:np.savez_compressed(stream,**arrays)
        metadata={"schema":"coarse-histogram-frame-export-v1","channel":channel,"freq_id":fid,
            "product":{"path":str(product_path),"sha256":digest,"bytes":initial_stat.st_size},
            "source_records":{"ledger":{"path":str(ledger_path),"sha256":sha(ledger_path)},"eras":{"path":str(eras_path),"sha256":sha(eras_path)}},
            "current_era":int(era_record["current_era"]),"era_definitions":definitions,"counts":counts,
            "era_config_digest":era_record["config_digest"],"era_config":era_record["config"],
            "era_scope":"Corrected retrospective full-archive assignments reused unchanged; this does not make them causal or independent of histogram data",
            "membership_rule":"Inclusive first_month..last_month using frame UTC month, acquisition-start month when frame time is nonfinite. Unassigned transition/outside months retain era_id=-1. Health status does not change temporal labels.",
            "histogram_rule":"selected & finite_time & finite_Q; Q<=0 is retained and separately flagged, no statistical trimming",
            "timestamp_rule":"unit_time0_ctime[frame_unit_index]+frame_in_unit*nfft*unit_delta_time[frame_unit_index]; no fabricated frame interval",
            "acquisition_rule":"acquisition_id is stored unit_event_id mapped by frame_unit_index; original unit_keys/source_event_keys also retained to disambiguate units",
            "health":{"schema":HEALTH_GATE_SCHEMA,"reason_counts":reasons,
                "member_header_optimization":"Original v1 gate invoked. Its unsigned fine terms branch inspects only shape/dtype; header and full member byte length validated, then a read-only zero-stride metadata view is supplied. No fine numerical values or PSD are read or exported.",
                "fine_member_header":proxy.fine_header},
            "geometry":geometry,"decoded_npz_members":proxy.read_fields,"export_sha256":sha(output_path)}
        write_json(output/f"ch{channel:02d}.json",metadata)
    return metadata


CONTROL_PERIOD_RULE=("A control band has no emitter, so no era is dated: its archive is one period, era 1, from the first "
    "to the last membership month of its health-selected frames, with evidence 'archive start'.")


def control_band(project,channel,freq_id,pilot_hz):
    """The profile's control band for a product; ValueError for any other band.

    The frequency plan decides, not a channel list here: the band must have role control, the product must be
    read at the band's declared target_freq_id, and the product's nominal marker must lie inside the band."""
    try:band=project.frequency_plan.band(str(channel))
    except KeyError as error:raise ValueError(f"channel {channel} is not a band of the frequency plan") from error
    if band.role!="control":raise ValueError(f"channel {channel} has role {band.role!r} in the frequency plan, not control")
    target=int(project.target_freq_id(band))
    if int(freq_id)!=target:raise ValueError(f"control band {channel} is read at freq_id {target}, the product is at {freq_id}")
    if not band.contains_hz(pilot_hz):raise ValueError(f"marker position {pilot_hz} Hz lies outside band {channel}")
    return band


def control_period(member_month,selected,finite_time,unit_index):
    """The one period of a control band, as a saved-era record (CONTROL_PERIOD_RULE)."""
    months=member_month[selected&(member_month>=0)]
    if not months.size:raise ValueError("control product has no health-selected frame with a time")
    member=selected&(member_month>=0)
    populated=[month_label(int(value)) for value in np.unique(months)]
    return {"era":1,"first_month":populated[0],"last_month":populated[-1],"evidence":"archive start","state":"control",
        "months":populated,"populated_months":len(populated),"frames":int(member.sum()),
        "frames_without_time":int((member&~finite_time).sum()),"units":int(np.unique(unit_index[member]).size)}


def process_control(product_path,project,output):
    """Export a control band's frames with the screened fields and rules, but no ledger and one period."""
    initial_stat=product_path.stat();digest=sha(product_path)
    with Product(product_path,require_health=True) as product:
        proxy=HeaderFineArchive(product.archive,product_path);product._z=proxy
        view=product.view
        channel=int(view.physical_channel);fid=int(view.freq_id)
        band=control_band(project,channel,fid,float(product.scalar("pilot_frequency_hz")))
        if int(product.scalar("nfft"))!=NFFT:raise ValueError("unexpected frame geometry")
        health=evaluate_frame_health(proxy)
        valid=np.asarray(view.valid,dtype=bool);health_include=np.asarray(health.include,dtype=bool)
        selected=valid&health_include;Q=np.asarray(view.statistic,dtype=np.float64)
        frame_time=product.frame_time;unit_time=product.unit_time
        finite_time=np.isfinite(frame_time);finite_Q=np.isfinite(Q)
        hist=selected&finite_time&finite_Q
        unit_index=product.frame_unit_index
        member_month=memberships(frame_time,unit_time,[],1)[1]
        period=control_period(member_month,selected,finite_time,unit_index)
        timed_month,member_month,era_id,current=memberships(frame_time,unit_time,[period],period["era"])
        unit_event_id=product.unit_event_id
        if unit_event_id.dtype.kind not in "iu":raise ValueError("acquisition IDs must preserve their integer type")
        counts={"total":len(Q),"valid":int(valid.sum()),"health_include":int(health_include.sum()),
            "selected":int(selected.sum()),"finite_time_selected":int((selected&finite_time).sum()),
            "histogram_eligible":int(hist.sum()),"nonpositive_histogram":int((hist&(Q<=0)).sum()),
            "current_selected":int((selected&current).sum()),"current_histogram_eligible":int((hist&current).sum()),
            "current_untimed_selected":int((selected&current&~finite_time).sum()),
            "unassigned_selected":int((selected&(era_id<0)).sum()),
            "unassigned_histogram_eligible":int((hist&(era_id<0)).sum()),
            "untimed_selected":int((selected&~finite_time).sum()),"nonfinite_Q_selected":int((selected&~finite_Q).sum())}
        assert_count("period frames",counts["current_selected"],period["frames"])
        reasons={key:int(value) for key,value in health.reason_counts.items()}
        definitions=[{**period,"is_current":True,"exported_count_selected":counts["current_selected"],
            "exported_count_histogram_eligible":counts["current_histogram_eligible"],
            "exported_count_untimed_selected":counts["current_untimed_selected"]}]
        arrays={"Q":Q,"valid":valid,"health_include":health_include,"selected":selected,
            "finite_time":finite_time,"finite_Q":finite_Q,"positive_Q":finite_Q&(Q>0),
            "histogram_eligible":hist,"era_id":era_id,"is_current":current,"current_histogram_eligible":hist&current,
            "frame_index":product.frame_column("frame_index"),"frame_unit_index":unit_index,
            "frame_in_unit":product.frame_in_unit,"frame_time":frame_time,"unit_time":unit_time,
            "frame_month":timed_month,"membership_month":member_month,
            "acquisition_id":unit_event_id[unit_index],"unit_event_id":unit_event_id,
            "unit_keys":np.asarray(proxy["unit_keys"]),"source_event_keys":np.asarray(proxy["source_event_keys"]),
            "unit_time0_ctime":product.unit_time0,"unit_delta_time":product.unit_delta_time}
        geometry={name:int(product.scalar(name)) for name in ("nfft","detector_window_samples","num_input_streams","target_norm_sq","reference_norm_sum_sq")}
        geometry.update(mu0=product.mu0,sample_rate_hz=float(product.scalar("sample_rate_hz")),
                        pilot_frequency_hz=float(product.scalar("pilot_frequency_hz")),chime_frequency_hz=float(product.scalar("chime_frequency_hz")))
        final_stat=product_path.stat()
        if (initial_stat.st_size,initial_stat.st_mtime_ns)!=(final_stat.st_size,final_stat.st_mtime_ns):raise ValueError("product changed during extraction")
        output_path=output/f"ch{channel:02d}.npz"
        with output_path.open("xb") as stream:np.savez_compressed(stream,**arrays)
        plan_file=project.files["frequency_plan"]
        metadata={"schema":"coarse-histogram-frame-export-v1","role":"control","channel":channel,"freq_id":fid,
            "band":{"label":band.label,"low_mhz":band.low_mhz,"high_mhz":band.high_mhz,"role":band.role,"target_freq_id":band.target_freq_id},
            "product":{"path":str(product_path),"sha256":digest,"bytes":initial_stat.st_size},
            "source_records":{"frequency_plan":{"path":str(plan_file),"sha256":sha(plan_file)}},
            "current_era":period["era"],"era_definitions":definitions,"counts":counts,
            "era_scope":CONTROL_PERIOD_RULE,
            "membership_rule":"Inclusive first_month..last_month using frame UTC month, acquisition-start month when frame time is nonfinite. A frame with neither time retains era_id=-1. Health status does not change temporal labels.",
            "histogram_rule":"selected & finite_time & finite_Q; Q<=0 is retained and separately flagged, no statistical trimming",
            "timestamp_rule":"unit_time0_ctime[frame_unit_index]+frame_in_unit*nfft*unit_delta_time[frame_unit_index]; no fabricated frame interval",
            "acquisition_rule":"acquisition_id is stored unit_event_id mapped by frame_unit_index; original unit_keys/source_event_keys also retained to disambiguate units",
            "health":{"schema":HEALTH_GATE_SCHEMA,"reason_counts":reasons,
                "member_header_optimization":"Original v1 gate invoked. Its unsigned fine terms branch inspects only shape/dtype; header and full member byte length validated, then a read-only zero-stride metadata view is supplied. No fine numerical values or PSD are read or exported.",
                "fine_member_header":proxy.fine_header},
            "geometry":geometry,"decoded_npz_members":proxy.read_fields,"export_sha256":sha(output_path)}
        write_json(output/f"ch{channel:02d}.json",metadata)
    return metadata


def run_control(product,output,profile=None):
    """Export one control band into a new frames directory that the histogram comparison reads."""
    from pilot_proxy.config.project import default_project_dir, load_project
    product=Path(product).resolve();output=Path(output).resolve()
    project=load_project(Path(profile).resolve() if profile else default_project_dir())
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("use a new empty frames directory")
    source_paths=list(SOURCES)
    inputs={str(path):sha(path) for path in source_paths if path.exists()}
    inputs[str(project.files["frequency_plan"])]=sha(project.files["frequency_plan"])
    plan={"schema":"coarse-histogram-export-plan-v1","role":"control","created_utc":datetime.now(timezone.utc).isoformat(),"control_product":str(product),
        "profile":str(project.directory),"profile_sha256":project.file_sha256(),"inputs":inputs,"numpy_version":np.__version__,"python":sys.version,
        "scope":"Read-only extraction of a control band's small per-frame coordinates; no ledger, no era dating, no threshold, no tolerance, no verdict"}
    write_json(output/"extraction-plan.json",plan)
    record=process_control(product,project,output)
    print(json.dumps({"channel":record["channel"],"role":"control","counts":record["counts"]}),flush=True)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError(f"source record changed during extraction:{path}")
    manifest={"schema":"coarse-histogram-frames-manifest-v1","role":"control","completed_utc":datetime.now(timezone.utc).isoformat(),"passed":True,
        "extraction_plan_sha256":sha(output/"extraction-plan.json"),"totals":dict(record["counts"]),
        "channels":[{"channel":record["channel"],"freq_id":record["freq_id"],"role":"control","npz":f"ch{record['channel']:02d}.npz","metadata":f"ch{record['channel']:02d}.json","counts":record["counts"]}],
        "files":{path.name:sha(path) for path in output.iterdir() if path.is_file()},
        "era_denominator_note":"The period count includes acquisition-month assignments of untimed frames. Histogram eligibility requires actual finite frame time; both counts are retained.",
        "fine_and_psd_payloads_decoded":False}
    write_json(output/"manifest.json",manifest)
    return manifest


def run(products,release,output):
    products=Path(products).resolve();release=Path(release).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("use a new empty frames directory")
    preflight_path=release/"products-preflight.json"
    preflight={item["filename"]:item for item in json.loads(preflight_path.read_text())["products"]}
    ledger_paths=sorted((release/"ledger/channels").glob("ch*_fid*.json"))
    screened=len(default_project().frequency_plan.bands("screened"))
    if len(ledger_paths)!=screened or len(preflight)!=screened:raise ValueError(f"corrected cohort must contain {screened} bands")
    source_paths=list(SOURCES[:3])+[Path(eras.__file__).resolve()]+list(SOURCES[3:])
    inputs={str(path):sha(path) for path in source_paths+[preflight_path,release/"ledger/run.json"]}
    for path in ledger_paths:
        ledger=json.loads(path.read_text());era_path=release/f"channels/ch{ledger['channel']}/eras.json"
        inputs[str(path)]=sha(path);inputs[str(era_path)]=sha(era_path)
    plan={"schema":"coarse-histogram-export-plan-v1","created_utc":datetime.now(timezone.utc).isoformat(),"products":str(products),"corrected_release":str(release),"inputs":inputs,"numpy_version":np.__version__,"python":sys.version,"scope":"Read-only extraction of small per-frame coordinates with exact corrected product/era/health denominators; no refitting"}
    write_json(output/"extraction-plan.json",plan)
    records=[]
    for path in ledger_paths:
        ledger=json.loads(path.read_text());product_path=products/ledger["product"];era_path=release/f"channels/ch{ledger['channel']}/eras.json"
        result=process_channel(product_path,path,era_path,output,preflight[ledger["product"]]);records.append(result)
        print(json.dumps({"channel":result["channel"],"counts":result["counts"]}),flush=True)
    for path,digest in inputs.items():
        if sha(path)!=digest:raise ValueError(f"source/release record changed during extraction:{path}")
    totals={key:sum(record["counts"][key] for record in records) for key in records[0]["counts"]}
    manifest={"schema":"coarse-histogram-frames-manifest-v1","completed_utc":datetime.now(timezone.utc).isoformat(),"passed":True,
        "extraction_plan_sha256":sha(output/"extraction-plan.json"),"totals":totals,"channels":[{"channel":record["channel"],"freq_id":record["freq_id"],"npz":f"ch{record['channel']:02d}.npz","metadata":f"ch{record['channel']:02d}.json","counts":record["counts"]} for record in records],
        "files":{path.name:sha(path) for path in output.iterdir() if path.is_file()},
        "era_denominator_note":"Saved era counts include acquisition-month assignments of untimed frames. Histogram eligibility requires actual finite frame time; both counts are retained.",
        "fine_and_psd_payloads_decoded":False}
    write_json(output/"manifest.json",manifest)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(prog="pilot-proxy characterize histogram-frames",description=__doc__)
    parser.add_argument("--products",type=Path);parser.add_argument("--release",type=Path,help="the archive characterization (ledger and saved eras) the frames are keyed to");parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--control-product",type=Path,help="export this control band's product alone, checked against the project profile")
    parser.add_argument("--profile",type=Path,help="project profile directory for --control-product (default: the repository's)")
    args=parser.parse_args(argv)
    if args.control_product:
        if args.products or args.release:parser.error("--control-product takes no --products or --release")
        run_control(args.control_product,args.output,args.profile)
    elif args.products and args.release and not args.profile:run(args.products,args.release,args.output)
    else:parser.error("give --products and --release, or --control-product")
    return 0


__all__ = ["CONTROL_PERIOD_RULE", "HeaderFineArchive", "control_band", "control_period", "main", "member_header",
           "memberships", "process_channel", "process_control", "run", "run_control"]
