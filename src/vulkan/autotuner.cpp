#include "autotuner.h"
#include "device_context.h"
#include "generated/shader_registry.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>

namespace {

// spaces up to this size try every config; larger ones descend from a spread of samples
constexpr size_t EXHAUSTIVE_LIMIT = 32;
constexpr size_t SEARCH_SAMPLES = 16;
static_assert((SEARCH_SAMPLES & (SEARCH_SAMPLES - 1)) == 0, "samples are taken in bit-reversed order");

// a full timing is the median of up to MAX_TIMING_SPANS spans, sized to about FULL_TIMING_TARGET_NS of GPU time
constexpr size_t MAX_TIMING_SPANS = 5;
constexpr double FULL_TIMING_TARGET_NS = 250e6;

// only configs probing within this factor of the best full timing are timed in full
constexpr double FULL_TIMING_FACTOR = 4.0;
// a kernel still over HOPELESS_FACTOR after this many configs is abandoned; also the compile batch size. Looser than
// FULL_TIMING_FACTOR: an emulated coop kernel's configs can differ 10x, so its first samples can all be poor
constexpr size_t HOPELESS_AFTER = 4;
constexpr double HOPELESS_FACTOR = 8.0;
// a kernel whose first config probes this far over the best is skipped
constexpr double KERNEL_DROP_FACTOR = 20.0;
// enough workgroups to fill a large GPU, few enough that an emulated kernel can't run for seconds
constexpr uint64_t PROBE_MAX_WORKGROUPS = 512;

// cuts the grid to at most PROBE_MAX_WORKGROUPS (keeping z, then y, whole) and returns the fraction kept;
double shrink_to_probe(TuneDispatch& d)
{
    const uint64_t full = uint64_t(d.groupX) * d.groupY * d.groupZ;
    if (full <= PROBE_MAX_WORKGROUPS) return 1.0;
    d.groupZ = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(d.groupZ, PROBE_MAX_WORKGROUPS)));
    d.groupY = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(d.groupY, PROBE_MAX_WORKGROUPS / d.groupZ)));
    d.groupX = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(d.groupX, PROBE_MAX_WORKGROUPS / (uint64_t(d.groupY) * d.groupZ))));
    return static_cast<double>(uint64_t(d.groupX) * d.groupY * d.groupZ) / full;
}

double env_number(const char* name, double fallback)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    char* end = nullptr;
    double parsed = std::strtod(value, &end);
    return end != value ? parsed : fallback;
}

std::string describe(const TuneSpace& space, const TuneConfig& config)
{
    std::ostringstream text;
    for (size_t i = 0; i < config.values.size() && i < space.params.size(); i++) {
        text << (i ? " " : "") << space.params[i].name << '=' << config.values[i];
    }
    return text.str();
}

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

Autotuner::Autotuner(DeviceContext* device)
    : device(device),
      budget_seconds_(env_number("TORCHVULKAN_TUNE_BUDGET_S", 20.0)),
      log_(env_number("TORCHVULKAN_TUNE_LOG", 0) != 0),
      threads_(std::clamp<unsigned>(std::thread::hardware_concurrency() / 2, 1u, 8u))
{
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

void Autotuner::compile(const std::vector<TuneConfig>& configs, const std::function<TuneDispatch(const TuneConfig&)>& build)
{
    if (configs.size() <= 1 || threads_ <= 1) return; // compiled on first use

    struct Pipeline { ShaderKey key; uint32_t subgroup; SpecializationBuilder spd; };
    std::vector<Pipeline> pipelines;
    std::set<std::tuple<uint32_t, uint32_t, std::string>> seen;

    for (const TuneConfig& c : configs)
    {
        try {
            TuneDispatch d = build(c);
            if (!seen.emplace(device->shader_manager->resolveBinary(d.key), d.subgroupSize, std::string(d.spd.build().key)).second) continue;
            pipelines.push_back({d.key, d.subgroupSize, d.spd});
        } catch (const c10::Error&) {
            // reported when timed
        }
    }

    std::atomic<size_t> next{0};
    auto work = [&]() {
        for (size_t i; (i = next++) < pipelines.size();) {
            try {
                device->shader_manager->prepare(pipelines[i].key, pipelines[i].subgroup, pipelines[i].spd.build());
            } catch (...) {
                // fails again when timed, and is skipped there
            }
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 1; t < std::min(pipelines.size(), threads_); t++) pool.emplace_back(work);
    work();
    for (std::thread& t : pool) t.join();
}

TuneConfig Autotuner::fastest(const TuneSpace& space, const std::function<TuneDispatch(const TuneConfig&)>& build)
{
    const std::vector<TuneConfig> configs = space.configs();
    if (configs.empty()) return {};
    if (configs.size() == 1 || !device->shader_manager->supportsTimestamps()) return configs.front();

    const auto start = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };

    // every config timed so far (probe or full), and the best full timing over all kernels
    struct Timing { double time; bool full; };
    std::map<std::vector<uint32_t>, Timing> timed;
    double best_time = std::numeric_limits<double>::infinity();
    auto over_budget = [&]() { return budget_seconds_ > 0 && !std::isinf(best_time) && elapsed() > budget_seconds_; };

    std::exception_ptr first_error;

    auto span_of = [&](const TuneDispatch& d, uint32_t runs) {
        return device->shader_manager->getTimestamp(
            d.key, d.subgroupSize, d.spd.build(), d.pcs.build(), d.read, d.write, d.groupX, d.groupY, d.groupZ, runs
        );
    };

    auto time_of = [&](const TuneConfig& c, bool probe) {
        double time = std::numeric_limits<double>::infinity();
        auto known = timed.find(c.values);
        if (known != timed.end() && (known->second.full || known->second.time > FULL_TIMING_FACTOR * best_time)) return known->second.time;
        try {
            TuneDispatch d = build(c);
            if (known != timed.end()) {
                time = known->second.time;
            } else {
                TuneDispatch part = d;
                double share = shrink_to_probe(part);
                span_of(part, 1);
                time = span_of(part, 2) / share + d.extra_ns;
                timed[c.values] = {time, false};
            }
            if (!probe && time <= FULL_TIMING_FACTOR * best_time) {
                // short kernels get 3 runs per span against noise; long ones fewer spans
                const double estimate = std::clamp(time - d.extra_ns, 1.0, FULL_TIMING_TARGET_NS);
                const uint32_t runs = estimate * 3 * MAX_TIMING_SPANS <= FULL_TIMING_TARGET_NS ? 3 : 1;
                const size_t span_count = std::clamp<size_t>(static_cast<size_t>(FULL_TIMING_TARGET_NS / (estimate * runs)), 1, MAX_TIMING_SPANS);
                std::vector<double> spans(span_count);
                for (double& span : spans) span = span_of(d, runs);
                std::nth_element(spans.begin(), spans.begin() + spans.size() / 2, spans.end());
                time = spans[spans.size() / 2] + d.extra_ns;
                timed[c.values] = {time, true};
                best_time = std::min(best_time, time);
            }
        } catch (const c10::Error&) {
            // a config the driver cannot build is skipped
            if (!first_error) first_error = std::current_exception();
            timed[c.values] = {time, true};
        }
        if (log_) std::cerr << "torchvulkan tune: " << describe(space, c) << (probe ? " probe " : " ") << time / 1e6 << " ms\n";
        return time;
    };

    auto compile_new = [&](const std::vector<TuneConfig>& batch) {
        std::vector<TuneConfig> todo;
        for (const TuneConfig& c : batch) if (!timed.count(c.values)) todo.push_back(c);
        if (todo.empty()) return;
        const double t0 = elapsed();
        compile(todo, build);
        if (log_) std::cerr << "torchvulkan tune: compiled " << todo.size() << " pipelines in " << elapsed() - t0 << " s\n";
    };

    auto search = [&](const TuneSpace& part) {
        const std::vector<TuneConfig> own = part.configs();
        TuneConfig best;
        double best_own = std::numeric_limits<double>::infinity();
        std::set<std::vector<uint32_t>> considered;

        // times a config once; returns whether it is this kernel's fastest so far
        auto consider = [&](const TuneConfig& c)
        {
            if (!considered.insert(c.values).second) return false;
            double time = time_of(c, false);
            if (time >= best_own) return false;
            best_own = time;
            best = c;
            return true;
        };
        auto hopeless = [&]() { return considered.size() >= HOPELESS_AFTER && best_own > HOPELESS_FACTOR * best_time; };

        // samples in bit-reversed order, so each batch spreads over the whole space rather than covering one stretch of
        // it (configs() varies the last param fastest, so a stretch mostly differs in the last params)
        std::vector<TuneConfig> samples;
        if (own.size() <= EXHAUSTIVE_LIMIT) samples = own;
        else for (size_t k = 0; k < SEARCH_SAMPLES; k++)
        {
            size_t reversed = 0;
            for (size_t b = 1; b < SEARCH_SAMPLES; b <<= 1) reversed = (reversed << 1) | ((k & b) ? 1 : 0);
            samples.push_back(own[reversed * own.size() / SEARCH_SAMPLES]);
        }

        // in batches, so a hopeless kernel stops before compiling everything
        for (size_t i = 0; i < samples.size(); i += HOPELESS_AFTER)
        {
            std::vector<TuneConfig> batch(samples.begin() + i, samples.begin() + std::min(i + HOPELESS_AFTER, samples.size()));
            compile_new(batch);
            for (const TuneConfig& c : batch) consider(c);
            if (hopeless() || over_budget()) return std::pair{best, best_own};
        }
        if (own.size() <= EXHAUSTIVE_LIMIT) return std::pair{best, best_own};

        // coordinate descent: try every option of one param at a time, keep the fastest, until a pass finds nothing faster
        bool improved = !best.empty();
        while (improved && !over_budget())
        {
            improved = false;
            for (size_t i = 0; i < part.params.size() && !over_budget(); i++)
            {
                std::vector<TuneConfig> candidates;
                for (uint32_t option : part.params[i].options)
                {
                    TuneConfig candidate = best;
                    candidate.values[i] = option;
                    if (!considered.count(candidate.values) && in_space(part, candidate)) candidates.push_back(candidate);
                }
                compile_new(candidates);
                for (const TuneConfig& c : candidates) if (consider(c)) improved = true;
            }
        }
        return std::pair{best, best_own};
    };

    // kernels' params don't connect, so a descent can't cross kernels: search each separately
    struct Part { TuneSpace space; uint32_t kernel; double probe; };
    std::vector<Part> parts;
    if (!space.kernel_param) parts.push_back({space, 0, 0.0});
    else {
        for (uint32_t kernel : space.params[*space.kernel_param].options) {
            TuneSpace part = space;
            part.params[*space.kernel_param].options = {kernel};
            if (!part.configs().empty()) parts.push_back({part, kernel, 0.0});
        }
    }

    // probe each kernel's first config and search the fastest kernel first, so it sets the bar for the rest
    if (parts.size() > 1) {
        std::vector<TuneConfig> probes;
        for (const Part& p : parts) probes.push_back(p.space.configs().front());
        compile_new(probes);
        for (size_t i = 0; i < parts.size(); i++) parts[i].probe = time_of(probes[i], true);
        std::stable_sort(parts.begin(), parts.end(), [](const Part& a, const Part& b) { return a.probe < b.probe; });
    }

    TuneConfig best;
    double best_overall = std::numeric_limits<double>::infinity();
    for (const Part& part : parts)
    {
        if (&part != &parts.front() && part.probe > KERNEL_DROP_FACTOR * best_time) {
            if (log_) std::cerr << "torchvulkan tune: dropping kernel " << part.kernel << " (probe " << part.probe / best_time << "x the best)\n";
            continue;
        }
        auto [config, time] = search(part.space);
        if (time < best_overall) {
            best = config;
            best_overall = time;
        }
    }

    if (log_) std::cerr << "torchvulkan tune: chose " << describe(space, best) << " " << best_overall / 1e6 << " ms after " << elapsed() << " s, "
                              << timed.size() << " configs\n";
    if (best.empty() && first_error) std::rethrow_exception(first_error);
    return best;
}
