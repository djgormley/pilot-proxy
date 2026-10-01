// Finite noise/TX-zero/tone reference capture; derived from preserved smoke v5.
// No physical power calibration or assumption of IID receiver noise.
#include <lime/LimeSuite.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <thread>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void on_signal(int) { interrupted = 1; }
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
void check(int code, const std::string& message) {
    if (code != 0) { const char* detail=LMS_GetLastErrorMessage(); throw std::runtime_error(message + ": " + (detail ? detail : "unknown")); }
}
std::string quoted(const std::string& value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c < 32) result += '?';
        else result += c;
    }
    return result + '"';
}
std::uint64_t serial_number(std::string s) {
    if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
    require(!s.empty() && s.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos, "Invalid hexadecimal serial");
    return std::stoull(s, nullptr, 16);
}
struct Output {
    int fd;
    explicit Output(const std::string& path) : fd(::open(path.c_str(), O_WRONLY|O_CREAT|O_EXCL, 0644)) {
        require(fd >= 0, "Output must be a new file: " + path);
    }
    ~Output() { if (fd >= 0) { ::fsync(fd); ::close(fd); } }
    void bytes(const void* ptr, std::size_t n) {
        const char* p = static_cast<const char*>(ptr);
        while (n) { auto k = ::write(fd, p, n); require(k > 0, "Capture write failed"); p += k; n -= k; }
    }
    void text(const std::string& s) { bytes(s.data(), s.size()); }
};
struct Options {
    std::string serial, tx_file, rx_file, status_file, startup_file, chunks_file, tx_status_file, accepted_file, schedule_file, mode;
    double frequency = 0, rate = 2000000, rx_bandwidth = 1500000, tx_bandwidth = 5000000;
    unsigned rx_gain = 30, tx_gain = 0, expected_tx_gain = 0;
    std::uint64_t tx_samples = 0, record_samples = 0;
};
Options parse(int argc, char** argv) {
    require(argc > 1 && argc % 2 == 1, "Expected name-value options");
    std::map<std::string, std::string> values;
    for (int i = 1; i < argc; i += 2) require(values.emplace(argv[i], argv[i+1]).second, "Duplicate argument");
    auto get = [&](const char* key) { auto it = values.find(key); require(it != values.end(), std::string("Missing ")+key); return it->second; };
    Options o;
    o.serial = get("--serial"); serial_number(o.serial);
    o.mode=get("--mode"); require(o.mode=="noise" || o.mode=="txzero" || o.mode=="tone","Unknown reference mode");
    o.record_samples=std::stoull(get("--record-samples")); o.accepted_file=get("--accepted-file");o.schedule_file=get("--schedule-file");
    o.frequency = std::stod(get("--frequency-hz")); o.rate = std::stod(get("--sample-rate-hz"));
    o.rx_bandwidth = std::stod(get("--rx-bandwidth-hz")); o.tx_bandwidth = std::stod(get("--tx-bandwidth-hz"));
    const auto rx = std::stoul(get("--native-rx-gain-db")); require(rx <= 50, "RX gain exceeds smoke limit"); o.rx_gain = rx;
    const auto tx=std::stoul(get("--native-tx-gain-db")); require(tx<=50,"TX gain exceeds smoke limit"); o.tx_gain=tx;
    const auto expected_tx=std::stoul(get("--expected-tx-gain-readback-db"));
    require(expected_tx<=50,"Expected TX readback outside smoke range"); o.expected_tx_gain=expected_tx;
    o.tx_samples = std::stoull(get("--tx-samples"));
    o.tx_file = get("--tx-file"); o.rx_file = get("--rx-file"); o.status_file = get("--status-file");
    o.startup_file=get("--startup-file"); o.chunks_file=get("--chunks-file"); o.tx_status_file=get("--tx-status-file");
    require(values.size() == 19, "Unknown argument");
    require(std::isfinite(o.frequency) && o.frequency >= 30000000 && o.frequency <= 1900000000., "Frequency outside Mini BAND2 smoke range");
    require(std::isfinite(o.rate) && o.rate >= 1000000 && o.rate <= 4000000, "Rate outside smoke limits");
    require(std::isfinite(o.rx_bandwidth) && o.rx_bandwidth >= 1400100 && o.rx_bandwidth < o.rate, "Invalid RX bandwidth");
    require(std::isfinite(o.tx_bandwidth) && o.tx_bandwidth >= 5000000 && o.tx_bandwidth <= 40000000, "Invalid TX bandwidth");
    require(o.frequency==500000000 && o.rate==2000000 && o.rx_gain==30,"Reference geometry is fixed at 500 MHz, 2 MHz, native RX30");
    require(o.record_samples>=40000 && o.record_samples<=8000000,"Record must be 20 ms through 4 s");
    require(o.mode=="noise" ? o.tx_samples==0 : o.tx_samples==o.record_samples+600000,"Finite payload duration differs from record plus fixed guards");
    return o;
}
struct Hardware {
    lms_device_t* device = nullptr;
    lms_stream_t rx{}, tx{};
    bool rx_setup=false, tx_setup=false, rx_started=false, tx_started=false;
    int stop_tx=-999, antenna_off=-999, gain_zero=-999, disable_tx=-999, off_readback=-999;
    int stop_rx=-999, disable_rx=-999, close_result=-999, destroy_tx=-999, destroy_rx=-999;
    void silence() noexcept {
        if (!device) return;
        // Remove the RF path before waiting on stream shutdown.
        antenna_off=LMS_SetAntenna(device, LMS_CH_TX, 0, LMS_PATH_NONE);
        gain_zero=LMS_SetGaindB(device, LMS_CH_TX, 0, 0);
        disable_tx=LMS_EnableChannel(device, LMS_CH_TX, 0, false);
        off_readback=LMS_GetAntenna(device, LMS_CH_TX, 0);
        if (tx_started) { stop_tx=LMS_StopStream(&tx); tx_started=false; }
    }
    void close() noexcept {
        if (!device) return;
        silence();
        if (rx_started) { stop_rx=LMS_StopStream(&rx); rx_started=false; }
        if (tx_setup) { destroy_tx=LMS_DestroyStream(device,&tx); tx_setup=false; }
        if (rx_setup) { destroy_rx=LMS_DestroyStream(device,&rx); rx_setup=false; }
        disable_rx=LMS_EnableChannel(device,LMS_CH_RX,0,false);
        close_result=LMS_Close(device); device=nullptr;
    }
    ~Hardware() { close(); }
};
struct Counters { std::uint64_t underrun=0, overrun=0, dropped=0; };
void setup(Hardware& h, bool tx) {
    auto& stream = tx ? h.tx : h.rx;
    stream.isTx=tx; stream.channel=0; stream.fifoSize=1048576;
    stream.throughputVsLatency=.5F; stream.dataFmt=lms_stream_t::LMS_FMT_F32;
    stream.linkFmt=lms_stream_t::LMS_LINK_FMT_I12;
    check(LMS_SetupStream(h.device,&stream), "Setup stream");
    (tx ? h.tx_setup : h.rx_setup)=true;
}
int run(const Options& o) {
    // All output paths are exclusively claimed before opening a device.
    Output rx_output(o.rx_file), status_output(o.status_file), startup_output(o.startup_file), chunks_output(o.chunks_file), tx_status_output(o.tx_status_file), accepted_output(o.accepted_file), schedule_output(o.schedule_file);
    std::ifstream input(o.tx_file, std::ios::binary|std::ios::ate);
    require(input.good() && input.tellg()==static_cast<std::streamoff>(o.tx_samples*8), "TX file size differs");
    input.seekg(0); std::vector<std::complex<float>> waveform(o.tx_samples);
    if(!waveform.empty()) {input.read(reinterpret_cast<char*>(waveform.data()), waveform.size()*8); require(input.good(), "TX read failed");}
    for (auto v: waveform) require(std::isfinite(v.real()) && std::isfinite(v.imag()) && std::abs(v.real())<=.01F && std::abs(v.imag())<=.01F, "TX finite component limit 0.01 exceeded");
    const auto tail=static_cast<std::size_t>(o.rate*.05);
    if(o.mode!="noise") {
        require(waveform.size()>2*tail, "Waveform too short for zero guards");
        for(std::size_t i=0;i<tail;++i) require(waveform[i]==std::complex<float>(0,0),"First 50 ms must be zero");
        for(std::size_t i=waveform.size()-tail;i<waveform.size();++i) require(waveform[i]==std::complex<float>(0,0),"Final 50 ms must be zero");
    }
    if(o.mode=="txzero") for(auto v:waveform) require(v==std::complex<float>(0,0),"TX-zero payload must contain only zeros");
    require(std::string(LMS_GetLibraryVersion()).rfind("23.11.",0)==0, "Worker requires LimeSuite 23.11 ABI");
    Hardware h;
    std::atomic<bool> stop{false}, ready{false}, rx_failed{false};
    std::atomic<std::uint64_t> next_rx{0}, finish_at{std::numeric_limits<std::uint64_t>::max()}, accepted_start{std::numeric_limits<std::uint64_t>::max()};
    std::thread receiver;
    std::string error, rx_error, descriptor;
    std::uint64_t first_rx=0, captured=0, ready_at=0, tx_start=0, sent=0, accepted_samples=0, accepted_first=0, accepted_next=0, schedule_base=0;
    double rx_rate=0, tx_rate=0, rf_rate=0, rx_frequency=0, tx_frequency=0, rx_lpf=0, tx_lpf=0;
    unsigned actual_rx_gain=0, actual_tx_gain=0;
    double peak=0, max_chunk_rms=0;
    Counters rx_counts, tx_counts, qualified_counts;
    std::uint64_t startup_samples=0, startup_resets=0, qualification_overlap_offset=0, qualification_samples=0;
    bool tx_attempted=false, device_opened=false, tx_started_before_qualification=false, payload_submission_attempted=false, tx_final_status_saved=false;
    Counters tx_before_submission, tx_after_submission;
    std::map<std::string,uint16_t> disabled_before, disabled_after;
    auto verify_disabled=[&](std::map<std::string,uint16_t>& evidence) {
        require(LMS_GetAntenna(h.device,LMS_CH_TX,0)==LMS_PATH_NONE,"Noise TX antenna path is not NONE");
        for(const auto& item:std::vector<std::pair<LMS7Parameter,uint16_t>>{{LMS7_TXEN_A,0},{LMS7_EN_TXTSP,0},{LMS7_EN_G_TRF,0},{LMS7_PD_TXPAD_TRF,1}}) {
            uint16_t value=999;check(LMS_ReadParam(h.device,item.first,&value),"TX-disabled register readback");
            evidence[item.first.name]=value;require(value==item.second,"TX-disabled register mismatch: "+std::string(item.first.name));
        }
    };
    std::uint64_t tx_status_queries=0;
    const auto deadline=Clock::now()+std::chrono::seconds(20);
    auto alive=[&]() { require(!interrupted && !stop.load() && Clock::now()<deadline, "Interrupted or smoke deadline reached"); };
    auto sample_tx_status=[&](const std::string& phase, bool require_active=true) {
        lms_stream_status_t delta{}; const int ret=LMS_GetStreamStatus(&h.tx,&delta);
        if(ret==0) {
            tx_counts.underrun+=delta.underrun;tx_counts.overrun+=delta.overrun;tx_counts.dropped+=delta.droppedPackets;
            auto& portion=payload_submission_attempted?tx_after_submission:tx_before_submission;
            portion.underrun+=delta.underrun;portion.overrun+=delta.overrun;portion.dropped+=delta.droppedPackets;
        }
        std::ostringstream row;
        row<<"{\"query_index\":"<<tx_status_queries++<<",\"phase\":"<<quoted(phase)<<",\"status_return\":"<<ret
           <<",\"payload_submission_attempted\":"<<(payload_submission_attempted?"true":"false")<<",\"samples_accepted_so_far\":"<<sent
           <<",\"active\":"<<(delta.active?"true":"false")<<",\"host_fifo_empty_wait_delta\":"<<delta.underrun
           <<",\"overrun_delta\":"<<delta.overrun<<",\"dropped_delta\":"<<delta.droppedPackets
           <<",\"fifo_filled_samples\":"<<delta.fifoFilledCount<<",\"reported_last_tx_timestamp\":"<<delta.timestamp<<"}\n";
        tx_status_output.text(row.str());
        require(ret==0,"TX stream status failed");
        require(!require_active || delta.active,"TX stream inactive");
        require(delta.overrun==0 && delta.droppedPackets==0,"TX stream overrun/drop");
        // Legacy LimeSuite underrun counts empty HOST FIFO waits, including
        // deliberate empty startup and waits after the finite burst was queued.
        // Preserve them; they do not alone establish a missing RF sample.
    };
    try {
        const int n=LMS_GetDeviceList(nullptr); require(n>0, "No LimeSDR found");
        std::vector<lms_info_str_t> devices(n); require(LMS_GetDeviceList(devices.data())==n, "Device inventory changed");
        const char* selected=nullptr;
        for (auto& d:devices) {
            std::string text(d); auto at=text.find("serial="); if (at==std::string::npos) continue;
            auto s=text.substr(at+7); s=s.substr(0,s.find(','));
            if (serial_number(s)==serial_number(o.serial)) { require(selected==nullptr, "Serial is ambiguous"); selected=d; descriptor=text; }
        }
        require(selected!=nullptr, "Requested serial not found");
        check(LMS_Open(&h.device,selected,nullptr), "Open"); device_opened=true;
        check(LMS_SetAntenna(h.device,LMS_CH_TX,0,LMS_PATH_NONE),"Pre-init TX antenna off");
        check(LMS_SetGaindB(h.device,LMS_CH_TX,0,0),"Pre-init TX gain minimum");
        check(LMS_EnableChannel(h.device,LMS_CH_TX,0,false),"Pre-init TX disabled");
        check(LMS_Init(h.device), "Init"); alive();
        check(LMS_SetAntenna(h.device,LMS_CH_TX,0,LMS_PATH_NONE),"Initial TX antenna off");
        check(LMS_SetGaindB(h.device,LMS_CH_TX,0,0),"Initial TX gain minimum");
        check(LMS_EnableChannel(h.device,LMS_CH_TX,0,false),"Initial TX disabled");
        check(LMS_EnableChannel(h.device,LMS_CH_RX,0,true),"Enable RX");
        if(o.mode!="noise") check(LMS_EnableChannel(h.device,LMS_CH_TX,0,true),"Enable disconnected TX configuration");
        check(LMS_SetSampleRate(h.device,o.rate,0),"Sample rate");
        for (bool direction:{false,true}) {
            if(direction && o.mode=="noise") continue;
            alive(); check(LMS_SetLOFrequency(h.device,direction,0,o.frequency),"Tune LO");
            check(LMS_SetLPFBW(h.device,direction,0,direction?o.tx_bandwidth:o.rx_bandwidth),"LPF");
            check(LMS_SetGaindB(h.device,direction,0,direction?o.tx_gain:o.rx_gain),"Gain");
        }
        check(LMS_SetAntenna(h.device,LMS_CH_RX,0,LMS_PATH_LNAW),"RX antenna");
        check(LMS_SetAntenna(h.device,LMS_CH_TX,0,LMS_PATH_NONE),"TX antenna remains off");
        // No explicit LMS_Calibrate: this is transport smoke, not RF calibration.
        check(LMS_GetSampleRate(h.device,false,0,&rx_rate,&rf_rate),"RX rate readback");
        if(o.mode!="noise") check(LMS_GetSampleRate(h.device,true,0,&tx_rate,&rf_rate),"TX rate readback");
        check(LMS_GetLOFrequency(h.device,false,0,&rx_frequency),"RX LO readback");
        if(o.mode!="noise") check(LMS_GetLOFrequency(h.device,true,0,&tx_frequency),"TX LO readback");
        check(LMS_GetLPFBW(h.device,false,0,&rx_lpf),"RX LPF readback");
        if(o.mode!="noise") check(LMS_GetLPFBW(h.device,true,0,&tx_lpf),"TX LPF readback");
        check(LMS_GetGaindB(h.device,false,0,&actual_rx_gain),"RX gain readback");
        if(o.mode!="noise") check(LMS_GetGaindB(h.device,true,0,&actual_tx_gain),"TX gain readback");
        require(std::abs(rx_rate-o.rate)<=1 && (o.mode=="noise" || std::abs(tx_rate-o.rate)<=1), "Rate readback differs");
        require(std::abs(rx_frequency-o.frequency)<=100 && (o.mode=="noise" || std::abs(tx_frequency-o.frequency)<=100), "LO readback differs");
        require(std::abs(rx_lpf-o.rx_bandwidth)<=.05*o.rx_bandwidth && (o.mode=="noise" || std::abs(tx_lpf-o.tx_bandwidth)<=.05*o.tx_bandwidth), "LPF readback differs");
        require((o.mode=="noise" || actual_tx_gain==o.expected_tx_gain) && actual_rx_gain==o.rx_gain, "Gain readback differs");
        setup(h,false); if(o.mode!="noise") setup(h,true);
        // Complete path selection and stream starts before RX qualification.
        // No tone samples are submitted until qualified RX is ready.
        if(o.mode!="noise") {
            tx_attempted=true;
            check(LMS_SetAntenna(h.device,LMS_CH_TX,0,LMS_PATH_TX2),"TX antenna active before RX qualification");
            require(LMS_GetAntenna(h.device,LMS_CH_TX,0)==LMS_PATH_TX2,"TX path readback differs");
            check(LMS_StartStream(&h.tx),"Start empty TX before RX qualification");h.tx_started=true;tx_started_before_qualification=true;
        } else {
            check(LMS_SetAntenna(h.device,LMS_CH_TX,0,LMS_PATH_NONE),"Noise TX path NONE");
            check(LMS_EnableChannel(h.device,LMS_CH_TX,0,false),"Noise TX channel disabled");
        }
        check(LMS_StartStream(&h.rx),"Start RX"); h.rx_started=true;
        if(o.mode=="noise") verify_disabled(disabled_before);
        const auto startup_begin=Clock::now();
        receiver=std::thread([&]() {
            try {
                std::vector<std::complex<float>> buffer(16384), candidate;
                bool have_expected=false; std::uint64_t expected=0, candidate_first=0, candidate_offset=0, chunk_index=0;
                const auto settle_samples=static_cast<std::uint64_t>(o.rate*.25);
                const auto clean_samples=static_cast<std::uint64_t>(o.rate*.1);
                const auto max_startup_samples=static_cast<std::uint64_t>(o.rate*2);
                while (!stop.load()) {
                    alive(); const bool was_ready=ready.load();
                    if (!was_ready) require(Clock::now()-startup_begin<std::chrono::seconds(2) && startup_samples<max_startup_samples,"RX startup qualification budget exhausted");
                    lms_stream_meta_t meta{};
                    const int count=LMS_RecvStream(&h.rx,buffer.data(),buffer.size(),&meta,100);
                    const bool valid_count=count>0 && count<=static_cast<int>(buffer.size());
                    const auto file_offset=was_ready?captured:startup_samples;
                    // Preserve every returned valid-length block before validation.
                    if (valid_count) {
                        (was_ready?rx_output:startup_output).bytes(buffer.data(),count*8);
                        if(was_ready) captured+=count; else startup_samples+=count;
                    }
                    std::uint64_t accepted_in_chunk=0, accepted_offset=accepted_samples;
                    if(was_ready && valid_count) {
                        const auto end=finish_at.load();
                        if(end!=std::numeric_limits<std::uint64_t>::max()) {
                            const auto begin=accepted_start.load();
                            const auto lo=std::max(meta.timestamp,begin), hi=std::min(meta.timestamp+static_cast<std::uint64_t>(count),end);
                            if(lo<hi) {
                                require((accepted_samples==0 && lo==begin) || (accepted_samples>0 && lo==accepted_next),"Accepted interval timestamp gap");
                                if(accepted_samples==0) accepted_first=lo;
                                accepted_in_chunk=hi-lo;accepted_output.bytes(buffer.data()+(lo-meta.timestamp),accepted_in_chunk*8);
                                accepted_samples+=accepted_in_chunk;accepted_next=hi;
                            }
                        }
                    }
                    double chunk_peak=0, energy=0; bool finite=true;
                    if(valid_count) for(int i=0;i<count;++i) {
                        const auto v=buffer[i]; finite=finite && std::isfinite(v.real()) && std::isfinite(v.imag());
                        if(std::isfinite(v.real()) && std::isfinite(v.imag())) {
                            chunk_peak=std::max(chunk_peak,static_cast<double>(std::max(std::abs(v.real()),std::abs(v.imag()))));
                            energy+=std::norm(static_cast<std::complex<double>>(v));
                        }
                    }
                    const double rms=valid_count?std::sqrt(energy/count):0;
                    peak=std::max(peak,chunk_peak); max_chunk_rms=std::max(max_chunk_rms,rms);
                    lms_stream_status_t delta{}; const int status_return=LMS_GetStreamStatus(&h.rx,&delta);
                    if(status_return==0) {
                        rx_counts.underrun+=delta.underrun; rx_counts.overrun+=delta.overrun; rx_counts.dropped+=delta.droppedPackets;
                        if(was_ready) {qualified_counts.underrun+=delta.underrun;qualified_counts.overrun+=delta.overrun;qualified_counts.dropped+=delta.droppedPackets;}
                    }
                    const bool gap=valid_count && have_expected && meta.timestamp!=expected;
                    const bool clean=valid_count && !gap && status_return==0 && delta.active && delta.underrun==0 && delta.overrun==0 && delta.droppedPackets==0;
                    const double startup_elapsed=std::chrono::duration<double>(Clock::now()-startup_begin).count();
                    const bool startup_expired=!was_ready && (startup_elapsed>=2 || startup_samples>max_startup_samples);
                    std::string action=was_ready?"qualified_append":"startup_settling";
                    if(!was_ready) {
                        const bool settled=file_offset>=settle_samples;
                        if(!clean || !settled) {
                            if(!candidate.empty() || !clean) ++startup_resets;
                            candidate.clear(); action=clean?"startup_settling":"startup_reset";
                        } else {
                            if(candidate.empty()) {candidate_first=meta.timestamp;candidate_offset=file_offset;}
                            candidate.insert(candidate.end(),buffer.begin(),buffer.begin()+count);
                            action="qualification_candidate";
                        }
                    }
                    if(startup_expired) action="startup_budget_exhausted";
                    std::ostringstream row;
                    row<<std::setprecision(17)<<"{\"chunk_index\":"<<chunk_index++<<",\"phase\":"<<quoted(was_ready?"qualified":"startup")
                       <<",\"file\":"<<quoted(was_ready?"rx.cfile":"startup.cfile")<<",\"file_sample_offset\":"<<file_offset
                       <<",\"accepted_file_sample_offset\":"<<accepted_offset<<",\"accepted_samples_from_chunk\":"<<accepted_in_chunk
                       <<",\"received_count\":"<<count<<",\"timestamp\":"<<meta.timestamp<<",\"expected_timestamp_known\":"<<(have_expected?"true":"false")
                       <<",\"expected_timestamp\":"<<expected<<",\"timestamp_gap\":"<<(gap?"true":"false")<<",\"status_return\":"<<status_return
                       <<",\"stream_active\":"<<(delta.active?"true":"false")<<",\"startup_elapsed_seconds\":"<<startup_elapsed
                       <<",\"underrun_delta\":"<<delta.underrun<<",\"overrun_delta\":"<<delta.overrun<<",\"dropped_delta\":"<<delta.droppedPackets
                       <<",\"finite\":"<<(finite?"true":"false")<<",\"peak_component\":"<<chunk_peak<<",\"rms\":"<<rms
                       <<",\"candidate_samples\":"<<candidate.size()<<",\"action\":"<<quoted(action)<<"}\n";
                    chunks_output.text(row.str());
                    require(finite,"RX nonfinite"); require(chunk_peak<.98 && rms<.35,"RX overload guard");
                    require(!startup_expired,"RX startup qualification budget exhausted");
                    if(was_ready) {
                        require(valid_count,"Qualified RX returned no/invalid samples");
                        require(!gap,"Qualified RX timestamp discontinuity");
                        require(status_return==0,"Qualified RX stream status failed");
                        require(clean,"Qualified RX stream underrun/overrun/drop");
                    }
                    if(valid_count) {expected=meta.timestamp+count;have_expected=true;} else have_expected=false;
                    if(!was_ready && candidate.size()>=clean_samples) {
                        // Candidate is also retained verbatim in startup.cfile;
                        // this explicit overlap seeds a contiguous analyzed file.
                        first_rx=candidate_first;qualification_overlap_offset=candidate_offset;qualification_samples=candidate.size();
                        rx_output.bytes(candidate.data(),candidate.size()*8);captured=candidate.size();
                        ready_at=expected;next_rx.store(expected);ready.store(true);candidate.clear();
                    } else if(was_ready) next_rx.store(expected);
                    if(was_ready && expected>=finish_at.load()) break;
                }
            } catch (const std::exception& e) { rx_error=e.what(); rx_failed.store(true); stop.store(true); }
        });
        while (!ready.load()) { alive(); std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
        alive();
        schedule_base=next_rx.load();
        tx_start=o.mode=="noise" ? 0 : schedule_base+static_cast<std::uint64_t>(o.rate*.2);
        const auto begin=o.mode=="noise" ? schedule_base+static_cast<std::uint64_t>(o.rate*.2) : tx_start+static_cast<std::uint64_t>(o.rate*.15);
        std::ostringstream scheduled;
        scheduled<<"{\"schema\":\"lime-reference-schedule-v1\",\"mode\":"<<quoted(o.mode)
                 <<",\"schedule_base_timestamp\":"<<schedule_base<<",\"tx_start_timestamp\":"<<tx_start
                 <<",\"accepted_first_timestamp\":"<<begin<<",\"accepted_stop_timestamp_exclusive\":"<<begin+o.record_samples
                 <<",\"record_samples\":"<<o.record_samples<<"}\n";
        schedule_output.text(scheduled.str());require(::fsync(schedule_output.fd)==0,"Schedule fsync failed");
        accepted_start.store(begin);finish_at.store(begin+o.record_samples);
        if(o.mode!="noise") {
            sample_tx_status("before_first_submission");
            while(sent<o.tx_samples) {
                alive();const auto size=std::min<std::uint64_t>(16384,o.tx_samples-sent);
                lms_stream_meta_t meta{};meta.timestamp=tx_start+sent;meta.waitForTimestamp=true;
                meta.flushPartialPacket=sent+size==o.tx_samples;payload_submission_attempted=true;
                const int count=LMS_SendStream(&h.tx,waveform.data()+sent,size,&meta,100);
                require(count>0 && count<=static_cast<int>(size),"TX did not accept bounded samples");sent+=count;
                sample_tx_status("after_submission_call");
            }
        }
        receiver.join();require(!rx_failed.load(),rx_error);alive();
        if(o.mode!="noise") {sample_tx_status("after_capture");tx_final_status_saved=true;}
        else verify_disabled(disabled_after);
        require(sent==o.tx_samples,"Full finite TX waveform was not accepted");
        require(accepted_samples==o.record_samples && accepted_first==begin && accepted_next==begin+o.record_samples,"Exact accepted interval incomplete");
    } catch (const std::exception& e) { error=e.what(); }
    // Silence immediately on failure; receiver I/O has a bounded 100 ms timeout.
    stop.store(true); h.silence(); if (receiver.joinable()) receiver.join();
    if (rx_failed.load() && (error.empty() || error=="Interrupted or smoke deadline reached")) error=rx_error;
    if(h.tx_setup && tx_started_before_qualification && !tx_final_status_saved) {
        try {sample_tx_status("failure_cleanup_after_stop",false);}
        catch(const std::exception& e) {if(error.empty())error=e.what();}
    }
    if (error.empty() && interrupted) error="Interrupted";
    h.close();
    if (error.empty() && (h.antenna_off || h.gain_zero || h.disable_tx || h.off_readback!=LMS_PATH_NONE || (tx_started_before_qualification && (h.stop_tx || h.destroy_tx)) || h.stop_rx || h.destroy_rx || h.disable_rx || h.close_result)) error="TX cleanup not confirmed by API/readback";
    std::ostringstream disabled_json;disabled_json<<"{\"before\":{";bool comma=false;
    for(const auto& item:disabled_before){if(comma)disabled_json<<",";disabled_json<<quoted(item.first)<<":"<<item.second;comma=true;}
    disabled_json<<"},\"after\":{";comma=false;for(const auto& item:disabled_after){if(comma)disabled_json<<",";disabled_json<<quoted(item.first)<<":"<<item.second;comma=true;}disabled_json<<"}}";
    std::ostringstream report;
    report<<std::setprecision(17)<<"{\n\"schema\":\"lime-reference-capture-v1\",\n\"success\":"<<(error.empty()?"true":"false")
          <<",\n\"mode\":"<<quoted(o.mode)<<",\n\"record_samples_requested\":"<<o.record_samples
          <<",\n\"schedule_base_timestamp\":"<<schedule_base<<",\n\"accepted_samples\":"<<accepted_samples<<",\n\"accepted_first_timestamp\":"<<accepted_first<<",\n\"accepted_stop_timestamp_exclusive\":"<<accepted_next
          <<",\n\"accepted_interval_passed\":"<<(error.empty()?"true":"false")<<",\n\"tx_disabled_register_readbacks\":"<<disabled_json.str()
          <<",\n\"tx_chain_active_for_payload\":"<<(o.mode=="noise"?"false":"true")
          <<",\n\"error\":"<<quoted(error)<<",\n\"rx_error\":"<<quoted(rx_error)<<",\n\"scope\":\"fixed-window noise/TX-zero/tone reference; no power calibration or IID guarantee\""
          <<",\n\"library_version\":"<<quoted(LMS_GetLibraryVersion())<<",\n\"device_descriptor\":"<<quoted(descriptor)
          <<",\n\"initialization_includes_internal_tx_gain_calibration\":true,\n\"explicit_rf_calibration_run\":false,\n\"explicit_rf_calibration_meaning\":\"No explicit LMS_Calibrate call; initialization gain calibration and LPF tuning run internally\",\n\"lpf_configuration_includes_internal_filter_tuning\":true,\n\"rx_ready_timestamp\":"<<ready_at<<",\n\"tx_start_timestamp\":"<<tx_start
          <<",\n\"tx_started_before_rx_qualification\":"<<(tx_started_before_qualification?"true":"false")<<",\n\"payload_submission_attempted\":"<<(payload_submission_attempted?"true":"false")
          <<",\n\"device_opened\":"<<(device_opened?"true":"false")<<",\n\"tx_attempted\":"<<(tx_attempted?"true":"false")<<",\n\"tx_samples_sent\":"<<sent
          <<",\n\"first_rx_timestamp\":"<<first_rx<<",\n\"next_rx_timestamp\":"<<next_rx.load()<<",\n\"captured_samples\":"<<captured
          <<",\n\"rx_peak_component\":"<<peak<<",\n\"rx_max_chunk_rms\":"<<max_chunk_rms
          <<",\n\"rx_rate_hz\":"<<rx_rate<<",\n\"tx_rate_hz\":"<<tx_rate<<",\n\"rx_frequency_hz\":"<<rx_frequency<<",\n\"tx_frequency_hz\":"<<tx_frequency
          <<",\n\"rx_lpf_hz\":"<<rx_lpf<<",\n\"tx_lpf_hz\":"<<tx_lpf<<",\n\"native_rx_gain_db\":"<<actual_rx_gain<<",\n\"native_tx_gain_db\":"<<(o.mode=="noise"?"null":std::to_string(actual_tx_gain))<<",\n\"requested_native_tx_gain_db\":"<<o.tx_gain<<",\n\"expected_native_tx_gain_readback_db\":"<<o.expected_tx_gain
          <<",\n\"startup_samples\":"<<startup_samples<<",\n\"startup_resets\":"<<startup_resets<<",\n\"qualification_overlap_startup_sample_offset\":"<<qualification_overlap_offset<<",\n\"qualification_samples\":"<<qualification_samples
          <<",\n\"rx_qualified\":"<<(ready.load()?"true":"false")
          <<",\n\"startup_rx_underrun\":"<<(rx_counts.underrun-qualified_counts.underrun)<<",\n\"startup_rx_overrun\":"<<(rx_counts.overrun-qualified_counts.overrun)<<",\n\"startup_rx_dropped\":"<<(rx_counts.dropped-qualified_counts.dropped)
          <<",\n\"qualified_rx_underrun\":"<<qualified_counts.underrun<<",\n\"qualified_rx_overrun\":"<<qualified_counts.overrun<<",\n\"qualified_rx_dropped\":"<<qualified_counts.dropped
          <<",\n\"rx_underrun\":"<<rx_counts.underrun<<",\n\"rx_overrun\":"<<rx_counts.overrun<<",\n\"rx_dropped\":"<<rx_counts.dropped
          <<",\n\"tx_host_empty_waits_before_submission\":"<<tx_before_submission.underrun<<",\n\"tx_host_empty_waits_after_submission_attempt\":"<<tx_after_submission.underrun
          <<",\n\"tx_underrun_interpretation\":\"Host FIFO empty-wait count; not alone proof of missing RF samples. Phase deltas retained in tx-status.jsonl.\""
          <<",\n\"tx_underrun\":"<<tx_counts.underrun<<",\n\"tx_overrun\":"<<tx_counts.overrun<<",\n\"tx_dropped\":"<<tx_counts.dropped
          <<",\n\"cleanup\":{\"antenna_off_return\":"<<h.antenna_off<<",\"gain_zero_return\":"<<h.gain_zero<<",\"disable_tx_return\":"<<h.disable_tx
          <<",\"antenna_off_readback\":"<<h.off_readback<<",\"stop_tx_return\":"<<h.stop_tx<<",\"stop_rx_return\":"<<h.stop_rx<<",\"destroy_tx_return\":"<<h.destroy_tx<<",\"destroy_rx_return\":"<<h.destroy_rx<<",\"disable_rx_return\":"<<h.disable_rx<<",\"close_return\":"<<h.close_result<<"}\n}\n";
    status_output.text(report.str()); std::cout<<report.str(); return error.empty()?0:1;
}
int main(int argc,char** argv) {
    std::signal(SIGINT,on_signal); std::signal(SIGTERM,on_signal);
    try { return run(parse(argc,argv)); }
    catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; }
}
