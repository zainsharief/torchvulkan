#pragma once
#include <ATen/ATen.h>
#include <map>

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
    std::vector<TuneConfig> configs() const;
};

struct TuneDispatch
{
    ShaderKey key{};
    uint32_t subgroupSize = 0;
    SpecializationBuilder spec;
    PushConstantBuilder push;
    std::vector<at::Tensor> read;
    std::vector<at::Tensor> write;
    uint32_t groupX = 1, groupY = 1, groupZ = 1;
};

class Autotuner
{
public:
    explicit Autotuner(DeviceContext* device) : device(device) {}

    const TuneConfig& select(
        ShaderKey shader,
        uint64_t tuned_id,
        const TuneSpace& space,
        const std::function<TuneDispatch(const TuneConfig&)>& build
    );

private:
    DeviceContext* device = nullptr;
    std::mutex mutex_;
    std::map<std::pair<uint32_t, uint64_t>, TuneConfig> results_; // (shader binary, tuned_id)
    std::map<std::pair<uint64_t, uint64_t>, TuneConfig> stored_;  // (binary hash, tuned_id)
    std::string path_;
    bool loaded_ = false;

    void load();
    void save(uint64_t hash, uint64_t tuned_id, const TuneConfig& config);

    TuneConfig fastest(const TuneSpace& space, const std::function<TuneDispatch(const TuneConfig&)>& build);
};
