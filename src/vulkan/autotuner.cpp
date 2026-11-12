#include "autotuner.h"
#include "device_context.h"
#include "generated/shader_registry.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace {

// spaces up to this size are timed in full
constexpr size_t EXHAUSTIVE_LIMIT = 32;
constexpr size_t SEARCH_SAMPLES = 16;

// a config's time is the median of several spans, so a burst of other GPU work can't make a slow config look fast
constexpr size_t TIMING_SPANS = 5;

bool in_space(const TuneSpace& space, const TuneConfig& config)
{
    if (config.values.size() != space.params.size()) return false;
    for (size_t i = 0; i < config.values.size(); i++) {
        const std::vector<uint32_t>& options = space.params[i].options;
        if (std::find(options.begin(), options.end(), config.values[i]) == options.end()) return false;
    }
    return !space.valid || space.valid(config);
}

}

std::vector<TuneConfig> TuneSpace::configs() const
{
    std::vector<TuneConfig> result;
    if (params.empty()) return result;

    for (const TuneParam& p : params) {
        if (p.options.empty()) return result;
    }

    std::vector<size_t> index(params.size(), 0);
    TuneConfig config;
    config.values.resize(params.size());

    while (true)
    {
        for (size_t i = 0; i < params.size(); i++) config.values[i] = params[i].options[index[i]];
        if (!valid || valid(config)) result.push_back(config);

        size_t i = params.size();
        while (i > 0 && ++index[i - 1] == params[i - 1].options.size()) index[--i] = 0;
        if (i == 0) return result;
    }
}

const TuneConfig& Autotuner::select(
    ShaderKey shader,
    uint64_t tuned_id,
    const TuneSpace& space,
    const std::function<TuneDispatch(const TuneConfig&)>& build)
{
    const uint32_t binary = device->shader_manager->resolveBinary(shader);
    const std::pair<uint32_t, uint64_t> key{binary, tuned_id};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = results_.find(key);
        if (it != results_.end()) return it->second;
    }

    const uint64_t hash = torchvulkan::SHADER_BINARIES[binary].hash;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_) load();
        auto stored = stored_.find({hash, tuned_id});
        if (stored != stored_.end() && in_space(space, stored->second)) return results_.emplace(key, stored->second).first->second;
    }

    TuneConfig best = fastest(space, build);

    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = results_.emplace(key, std::move(best));
    if (inserted && !it->second.empty()) save(hash, tuned_id, it->second);
    return it->second;
}

void Autotuner::load()
{
    // assume active mutex
    loaded_ = true;
    path_ = device->cache.cachePath("tuning", "txt");
    if (path_.empty()) return;

    // one line per result: binary hash, tuned id, then the config's values; a later line wins
    std::ifstream file(path_);
    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream fields(line);
        uint64_t hash = 0, tuned_id = 0;
        if (!(fields >> std::hex >> hash >> tuned_id)) continue;
        TuneConfig config;
        for (uint32_t value; fields >> std::dec >> value;) config.values.push_back(value);
        stored_[{hash, tuned_id}] = std::move(config);
    }
}

void Autotuner::save(uint64_t hash, uint64_t tuned_id, const TuneConfig& config)
{
    // assume active mutex
    if (path_.empty()) return;

    std::ostringstream line;
    line << std::hex << hash << ' ' << tuned_id << std::dec;
    for (uint32_t value : config.values) line << ' ' << value;
    line << '\n';

    // appended in one write, so processes tuning at the same time interleave whole lines
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec);
    std::ofstream(path_, std::ios::app) << line.str() << std::flush;
}

TuneConfig Autotuner::fastest(const TuneSpace& space, const std::function<TuneDispatch(const TuneConfig&)>& build)
{
    const std::vector<TuneConfig> configs = space.configs();
    if (configs.empty()) return {};
    if (configs.size() == 1 || !device->shader_manager->supportsTimestamps()) return configs.front();

    TuneConfig best;
    double best_time = std::numeric_limits<double>::infinity();
    std::set<std::vector<uint32_t>> timed;
    std::exception_ptr first_error;

    // times a config once; returns whether it is the fastest so far
    auto consider = [&](const TuneConfig& c) 
    {
        if (!timed.insert(c.values).second) return false;

        double time = std::numeric_limits<double>::infinity();
        
        try {
            TuneDispatch d = build(c);
            std::vector<double> spans(TIMING_SPANS);
            for (double& span : spans)
            {
                span = device->shader_manager->getTimestamp(
                    d.key,
                    d.subgroupSize,
                    d.spec.build(), d.push.build(),
                    d.read, d.write,
                    d.groupX, d.groupY, d.groupZ
                );
            }
            std::nth_element(spans.begin(), spans.begin() + spans.size() / 2, spans.end());
            time = spans[spans.size() / 2];
        } catch (const c10::Error&) {
            // a config the driver cannot build is skipped
            if (!first_error) first_error = std::current_exception();
        }
        
        if (time >= best_time) return false;
        best_time = time;
        best = c;
        return true;
    };

    if (configs.size() <= EXHAUSTIVE_LIMIT) 
    {
        for (const TuneConfig& c : configs) consider(c);
        if (best.empty() && first_error) std::rethrow_exception(first_error);
        return best;
    } 

    // coordinate descent from the best of a spread of samples: try every option of one param at a time
    // and keep the fastest, until a whole pass finds nothing faster
    for (size_t i = 0; i < SEARCH_SAMPLES; i++) consider(configs[i * configs.size() / SEARCH_SAMPLES]);

    bool improved = !best.empty();
    while (improved)
    {
        improved = false;
        for (size_t i = 0; i < space.params.size(); i++)
        {
            for (uint32_t option : space.params[i].options)
            {
                TuneConfig candidate = best;
                candidate.values[i] = option;
                if (in_space(space, candidate) && consider(candidate)) improved = true;
            }
        }
    }

    if (best.empty() && first_error) std::rethrow_exception(first_error);
    return best;
}
