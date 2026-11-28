#pragma once
#include <ATen/ATen.h>
#include <map>
#include <set>
#include <string>
#include <optional>

#include "builders.h"
#include "shader.h"

struct DeviceContext;

struct TuneConfig
{
    std::vector<uint32_t> values;
    uint32_t operator[](size_t i) const { return values[i]; }
    bool empty() const { return values.empty(); }
};

struct TuneParam
{
    std::string name;
    std::vector<uint32_t> options;
    size_t size() const { return options.size(); }
};

struct TuneSpace
{
    std::vector<TuneParam> params;
    std::function<bool(const TuneConfig&)> valid;
    std::optional<size_t> kernel_param;
    std::vector<TuneConfig> configs() const;
};

struct TuneDispatch
{
    ShaderKey key{};
    uint32_t subgroupSize = 0;
    SpecializationBuilder spd;
    PushConstantBuilder pcs;
    std::vector<at::Tensor> read;
    std::vector<at::Tensor> write;
    uint32_t groupX = 1, groupY = 1, groupZ = 1;
    double extra_ns = 0; // work the config adds outside this dispatch, added to its time
};

// One way of running an op: its kernels at the op's dtype, a composite of other ops, or another dtype through casts.
// `run` does the whole op (dispatches and any casts); the tuner times it on the wall clock around a flush.
struct TunePath
{
    const char* name;
    std::function<void()> run;
};

class Autotuner
{
public:
    explicit Autotuner(DeviceContext* device);

    const TuneConfig& select(
        ShaderKey shader,
        uint64_t tuned_id,
        const TuneSpace& space,
        const std::function<TuneDispatch(const TuneConfig&)>& build
    );

    // The index of the fastest of `paths` for this op and shape, timed once and stored with the kernel configs
    // (keyed on `shader`'s binary, so a changed shader re-times it). One path is returned without timing. The caller
    // runs the chosen path itself; the timing runs every path, so paths must be safe to repeat.
    size_t choosePath(ShaderKey shader, uint64_t tuned_id, const std::vector<TunePath>& paths);

    // Forgets every tuned result of this process, so each is timed again on its next use; `persistent` also empties
    // the tuning file, otherwise the file keeps its lines and new timings are appended (a later line wins).
    void clear(bool persistent);
    // While retuning, the first use of each key in the block is timed again whatever is cached, and stored.
    void beginRetune();
    void endRetune();
    size_t entries();
    std::string cacheFile();

private:
    DeviceContext* device = nullptr;
    std::mutex mutex_;
    std::map<std::pair<uint32_t, uint64_t>, TuneConfig> results_; // (shader binary, tuned_id)
    std::map<std::pair<uint64_t, uint64_t>, TuneConfig> stored_;  // (binary hash, tuned_id)
    std::string path_;
    bool loaded_ = false;

    int retune_depth_ = 0;                             // nested retune blocks
    std::set<std::pair<uint32_t, uint64_t>> retuned_;  // keys timed again in the current retune block

    // the cached result of `key`, or null when it has to be timed (nothing cached, or retuning and not yet retimed)
    const TuneConfig* cached(const std::pair<uint32_t, uint64_t>& key, uint64_t hash, const std::function<bool(const TuneConfig&)>& usable);
    const TuneConfig& remember(const std::pair<uint32_t, uint64_t>& key, uint64_t hash, TuneConfig config);

    double budget_seconds_; // a search settles for its best so far after this long; 0 for no limit
    bool log_;
    size_t threads_;        // compiling pipelines; GPU timing stays serial

    void load();
    void save(uint64_t hash, uint64_t tuned_id, const TuneConfig& config);

    TuneConfig fastest(const TuneSpace& space, const std::function<TuneDispatch(const TuneConfig&)>& build);
    void compile(const std::vector<TuneConfig>& configs, const std::function<TuneDispatch(const TuneConfig&)>& build);
};
