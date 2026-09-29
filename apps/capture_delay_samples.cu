// Bounded, selected-channel voltage capture using the production ibverbs input.
// No GPU processing, disk writes in the receive loop, or FPGA configuration.
#include "spatial/libibverbs.hpp"
#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <thread>

#ifdef HAVE_IBVERBS
namespace {
constexpr size_t packet_bytes = 2622; // CustomHeader + 20 uint16 scales + 64*20 complex int8
constexpr size_t ring_slots = 2048;
constexpr size_t slot_bytes = 2688;
using json = nlohmann::json;
struct SharedWindow {
  std::array<std::atomic<int64_t>,4> first{};
  std::atomic<int64_t> start{0};
  int64_t samples;
  explicit SharedWindow(int64_t n) : samples(n) {}
};

class SampleState final : public ProcessorStateBase {
public:
  SampleState(int board, int64_t offset, std::vector<int> channels, SharedWindow &window)
      : board_(board), offset_(offset), channels_(std::move(channels)), window_(window),
        ring_(ring_slots*slot_bytes),
        retained_(channels_.size()*(window.samples/64+16)*packet_bytes) {
    for(int i=0;i<int(ring_slots);++i) free_.push_back(i);
  }
  void *get_current_write_pointer() override { return ring_.data(); }
  void *get_next_write_pointer() override { return ring_.data(); }
  void add_received_packet_metadata(int, const sockaddr_in &) override {
    throw std::runtime_error("sample capture requires zero-copy input");
  }
  int reserve_write_batch(int n,void **p,int *idx) override {
    int count=0;
    while(count<n && !free_.empty()) {
      idx[count]=free_.back();free_.pop_back();
      p[count]=ring_.data()+idx[count]*slot_bytes;++count;
    }
    return count;
  }
  int reserve_write_batch_strided(int, uint64_t &,int n,void **p,int *idx) override {
    return reserve_write_batch(n,p,idx);
  }
  void commit_write_batch(int n,const int *idx,const int *length,const sockaddr_in *) override {
    for(int i=0;i<n;++i) {
      ingest(ring_.data()+idx[i]*slot_bytes,length[i]);
      free_.push_back(idx[i]); // Payload is consumed synchronously before repost.
    }
  }
  void abandon_write_batch(int,const int *) override {}
  void *packet_ring_base() const override { return const_cast<uint8_t*>(ring_.data()); }
  size_t packet_ring_bytes() const override { return ring_.size(); }
  size_t packet_ring_slot_stride() const override { return slot_bytes; }
  size_t packet_ring_slot_offset() const override { return 0; }
  size_t packet_ring_slot_count() const override { return ring_slots; }
  size_t slot_data_capacity() const override { return slot_bytes; }
  void release_buffer(int) override {}
  void set_pipeline(GPUPipeline *) override {}
  void process_all_available_packets() override {}
  void handle_buffer_completion(bool=false) override {}
  void save(const std::filesystem::path &dir,json &entry) {
    auto name="fpga"+std::to_string(board_)+".packets";
    std::ofstream out(dir/name,std::ios::binary);
    out.write(reinterpret_cast<const char*>(retained_.data()),used_);
    out.close();if(!out) throw std::runtime_error("failed writing "+name);
    entry={{"file",name},{"seed_delay",offset_},{"received",packets_received.load()},
      {"retained",used_/packet_bytes},{"malformed",malformed_},
      {"capacity_rejections",overflow_},{"reached_window_end",reached_end.load()}};
  }
  std::atomic<bool> reached_end{false};
private:
  void ingest(const uint8_t *p,int length) {
    if(length!=int(packet_bytes)) {++malformed_;return;}
    CustomHeader hdr;std::memcpy(&hdr,p,sizeof(hdr));
    if(hdr.sample_count>uint64_t(INT64_MAX)) {++malformed_;return;}
    const int64_t aligned=int64_t(hdr.sample_count)-offset_;
    if(aligned<=0) {++malformed_;return;}
    if(window_.first[board_].load(std::memory_order_relaxed)==0)
      window_.first[board_].store(aligned,std::memory_order_release);
    int64_t start=window_.start.load(std::memory_order_acquire);
    if(start==0) {
      int64_t latest=0;
      for(auto &v:window_.first) {
        auto value=v.load(std::memory_order_acquire);if(value==0)return;
        latest=std::max(latest,value);
      }
      int64_t expected=0;
      window_.start.compare_exchange_strong(expected,latest+65536);
      start=window_.start.load(std::memory_order_acquire);
    }
    // Drain beyond the last retained sample to tolerate packet reordering.
    if(aligned>=start+window_.samples+4096) reached_end.store(true);
    if(aligned+64<=start || aligned>=start+window_.samples)return;
    if(std::find(channels_.begin(),channels_.end(),int(hdr.freq_channel))==channels_.end())return;
    if(used_+packet_bytes>retained_.size()){++overflow_;return;}
    std::memcpy(retained_.data()+used_,p,packet_bytes);used_+=packet_bytes;
  }
  int board_;int64_t offset_;std::vector<int> channels_;SharedWindow &window_;
  std::vector<uint8_t> ring_,retained_;std::vector<int> free_;
  size_t used_=0;uint64_t malformed_=0,overflow_=0;
};
std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> result;std::stringstream in(s);std::string word;
  while(std::getline(in,word,','))result.push_back(word);return result;
}
}
int main(int argc,char **argv) {
  argparse::ArgumentParser args("capture_delay_samples");
  args.add_argument("--interfaces").required();
  args.add_argument("--seed-delays").required();
  args.add_argument("--channels").required().help("1--40 coarse selectors retained in RAM; no FPGA retuning");
  args.add_argument("--output").required().help("New directory; never overwritten");
  args.add_argument("--samples").default_value(524288).scan<'i',int>();
  args.add_argument("--timeout").default_value(5.0).scan<'g',double>();
  args.add_argument("--port").default_value(36001).scan<'i',int>();
  try {
    args.parse_args(argc,argv);
    auto names=split(args.get<std::string>("--interfaces"));
    std::vector<int> channels;
    for(auto &v:split(args.get<std::string>("--channels")))channels.push_back(std::stoi(v));
    std::sort(channels.begin(),channels.end());
    if(names.size()!=4 || channels.empty() || channels.size()>40 || channels.front()<0 || channels.back()>65535 ||
       std::adjacent_find(channels.begin(),channels.end())!=channels.end())
      throw std::runtime_error("require four interfaces in FPGA order and 1--40 distinct valid channels");
    int samples=args.get<int>("--samples");double timeout=args.get<double>("--timeout");
    if(samples<4096 || samples>524288 || samples%64 || !(timeout>0 && timeout<=60))
      throw std::runtime_error("samples must be a multiple of 64 in [4096,524288]; timeout in (0,60]");
    std::filesystem::path output=args.get<std::string>("--output");
    if(std::filesystem::exists(output))throw std::runtime_error("output already exists");
    json seed;std::ifstream(args.get<std::string>("--seed-delays"))>>seed;
    SharedWindow window(samples);
    std::array<std::unique_ptr<SampleState>,4> states;
    std::array<std::unique_ptr<LibibverbsPacketCapture>,4> inputs;
    for(int b=0;b<4;++b) {
      const int64_t offset=seed.at(std::to_string(b)).get<int64_t>();
      if(offset < -1000000000000LL || offset > 1000000000000LL)
        throw std::runtime_error("seed offset outside supported range");
      states[b]=std::make_unique<SampleState>(b,offset,channels,window);
      inputs[b]=std::make_unique<LibibverbsPacketCapture>(names[b],args.get<int>("--port"),0,b,1);
      inputs[b]->arm_zero_copy(*states[b]);
    }
    // Preallocate and register all buffers before any receive threads start.
    std::array<std::thread,4> threads;
    for(int b=0;b<4;++b)threads[b]=std::thread([&,b]{inputs[b]->get_packets(*states[b]);});
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(timeout);
    bool complete=false;
    while(std::chrono::steady_clock::now()<deadline) {
      complete=true;for(auto &s:states)complete&=s->reached_end.load();
      if(complete)break;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    for(auto &s:states)s->running.store(false);
    for(auto &t:threads)t.join();
    std::filesystem::create_directories(output);
    json manifest={{"format","lambda-delay-packets-v1"},{"packet_bytes",packet_bytes},
      {"start_sample",window.start.load()},{"samples",samples},{"channels",channels},
      {"complete",complete},{"sample_seconds",1.08e-6},
      {"note","Seed offsets choose overlap only; never apply a fitted result automatically."}};
    for(int b=0;b<4;++b)states[b]->save(output,manifest["fpgas"][std::to_string(b)]);
    std::ofstream(output/"manifest.json")<<manifest.dump(2)<<'\n';
    std::cout<<manifest.dump(2)<<'\n';return complete?0:2;
  } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
#else
int main(){std::cerr<<"Requires libibverbs at build time\n";return 1;}
#endif
