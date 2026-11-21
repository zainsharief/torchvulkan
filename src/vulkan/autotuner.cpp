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

// spaces up to this size are timed in full
constexpr size_t EXHAUSTIVE_LIMIT = 32;
constexpr size_t SEARCH_SAMPLES = 16;

// a config's time is the median of several spans, so a burst of other GPU work can't make a slow config look fast
constexpr size_t TIMING_SPANS = 5;
// GPU time one config's full timing aims for
constexpr double TIMING_TARGET_NS = 250e6;

// a config this much slower than the best so far gets one dispatch, not the full timing
constexpr double SLOW_FACTOR = 4.0;
// a kernel whose first this many configs are all that slow is dropped (a cooperative-matrix kernel the
// driver emulates, say)
constexpr size_t HOPELESS_CONFIGS = 4;
// a kernel whose probe is this much slower than the best of another kernel isn't searched at all
constexpr double DROP_FACTOR = 20.0;
// probes dispatch at most this many workgroups and scale the time up: enough to fill a large GPU, few
// enough that a kernel the device only emulates can't run for seconds (or trip a driver watchdog)
constexpr uint64_t PROBE_GROUPS = 512;

// cuts the grid to about PROBE_GROUPS workgroups (keeping z, then y, whole where it can); returns the share kept.
// Tuned kernels' workgroups don't wait on each other, so a partial grid runs like the full one.
double shrink_to_probe(TuneDispatch& d)
{
    const uint64_t full = uint64_t(d.groupX) * d.groupY * d.groupZ;
    if (full <= PROBE_GROUPS) return 1.0;
    d.groupZ = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(d.groupZ, PROBE_GROUPS)));
    d.groupY = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(d.groupY, PROBE_GROUPS / d.groupZ)));
    d.groupX = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(d.groupX, PROBE_GROUPS / (uint64_t(d.groupY) * d.groupZ))));
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

// seconds one search may take before it settles for the best config so far; 0 for no limit
double tune_budget_seconds()
{
    static const double budget = env_number("TORCHVULKAN_TUNE_BUDGET_S", 20.0);
    return budget;
}

// threads compiling candidate pipelines; the GPU timing itself stays serial
size_t tune_threads()
{
    static const size_t threads = static_cast<size_t>(std::max(1.0,
        env_number("TORCHVULKAN_TUNE_THREADS", std::clamp<unsigned>(std::thread::hardware_concurrency() / 2, 1u, 8u))));
    return threads;
}

bool tune_log()
{
    static const bool enabled = env_number("TORCHVULKAN_TUNE_LOG", 0) != 0;
    return enabled;
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
    if (configs.size() <= 1 || tune_threads() <= 1) return; // compiled on first use

    // only the pipeline is needed: keep each dispatch's key and constants, not the dispatch (it can hold
    // large temporaries)
    struct Pipeline { ShaderKey key; uint32_t subgroup; SpecializationBuilder spd; };
    std::vector<Pipeline> pipelines;
    // configs that differ only outside the pipeline (push constants, grid) share one: build it once
    std::set<std::tuple<uint32_t, uint32_t, std::string>> seen;
    for (const TuneConfig& c : configs)
    {
        try {
            TuneDispatch d = build(c);
            if (!seen.emplace(device->shader_manager->resolveBinary(d.key), d.subgroupSize, std::string(d.spd.build().key)).second) continue;
            pipelines.push_back({d.key, d.subgroupSize, d.spd});
        } catch (const c10::Error&) {
            // timing reports it
        }
    }

    std::atomic<size_t> next{0};
    auto work = [&]() {
        for (size_t i; (i = next++) < pipelines.size();) {
            try {
                device->shader_manager->prepare(pipelines[i].key, pipelines[i].subgroup, pipelines[i].spd.build());
            } catch (...) {
                // a config the driver cannot build fails again when timed, and is skipped there
            }
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 1; t < std::min(pipelines.size(), tune_threads()); t++) pool.emplace_back(work);
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
    const double budget = tune_budget_seconds();

    // every config timed so far, as a probe or in full, and the best full timing over all kernels
    struct Timing { double time; bool full; };
    std::map<std::vector<uint32_t>, Timing> timed;
    double best_time = std::numeric_limits<double>::infinity();
    auto over_budget = [&]() { return budget > 0 && !std::isinf(best_time) && elapsed() > budget; };

    std::exception_ptr first_error;
    auto span_of = [&](const TuneDispatch& d, uint32_t runs) {
        return device->shader_manager->getTimestamp(
            d.key, d.subgroupSize, d.spd.build(), d.pcs.build(), d.read, d.write, d.groupX, d.groupY, d.groupZ, runs
        );
    };
    // a config is timed in full only while it could still win
    auto time_of = [&](const TuneConfig& c, bool probe) {
        double time = std::numeric_limits<double>::infinity();
        auto known = timed.find(c.values);
        if (known != timed.end() && (known->second.full || known->second.time > SLOW_FACTOR * best_time)) return known->second.time;
        try {
            TuneDispatch d = build(c);
            if (known != timed.end()) {
                time = known->second.time;
            } else {
                // a first look on part of the grid, scaled up
                TuneDispatch part = d;
                double share = shrink_to_probe(part);
                // a warm-up dispatch absorbs any compile the driver defers to first use; then two, because
                // AMD's timestamps around a single emulated cooperative-matrix dispatch can read microseconds
                // for milliseconds of work
                span_of(part, 1);
                time = span_of(part, 2) / share + d.extra_ns;
                timed[c.values] = {time, false};
            }
            if (!probe && time <= SLOW_FACTOR * best_time) {
                // short kernels get several spans of several runs, against noise; a long one is measured
                // well enough by a dispatch or two, and timing it fifteen times only slows the search
                const double estimate = std::clamp(time - d.extra_ns, 1.0, TIMING_TARGET_NS); // ns, from the probe
                const uint32_t runs = estimate * 3 * TIMING_SPANS <= TIMING_TARGET_NS ? 3 : 1;
                const size_t span_count = std::clamp<size_t>(static_cast<size_t>(TIMING_TARGET_NS / (estimate * runs)), 1, TIMING_SPANS);
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
        if (tune_log()) std::cerr << "torchvulkan tune: " << describe(space, c) << (probe ? " probe " : " ") << time / 1e6 << " ms\n";
        return time;
    };
    auto compile_new = [&](const std::vector<TuneConfig>& batch) {
        std::vector<TuneConfig> todo;
        for (const TuneConfig& c : batch) if (!timed.count(c.values)) todo.push_back(c);
        if (todo.empty()) return;
        const double t0 = elapsed();
        compile(todo, build);
        if (tune_log()) std::cerr << "torchvulkan tune: compiled " << todo.size() << " pipelines in " << elapsed() - t0 << " s\n";
    };

    // searches one kernel: a spread of samples (all of a small kernel), then coordinate descent from the best
    auto search = [&](const TuneSpace& part) {
        const std::vector<TuneConfig> own = part.configs();
        TuneConfig best;
        double best_own = std::numeric_limits<double>::infinity();
        std::set<std::vector<uint32_t>> considered;

        // times a config once; returns whether it is the fastest so far
        auto consider = [&](const TuneConfig& c)
        {
            if (!considered.insert(c.values).second) return false;
            double time = time_of(c, false);
            if (time >= best_own) return false;
            best_own = time;
            best = c;
            return true;
        };
        // a kernel whose first configs are all far slower than the best so far can't win
        auto hopeless = [&]() { return considered.size() >= HOPELESS_CONFIGS && best_own > SLOW_FACTOR * best_time; };

        std::vector<TuneConfig> samples;
        if (own.size() <= EXHAUSTIVE_LIMIT) samples = own;
        else for (size_t i = 0; i < SEARCH_SAMPLES; i++) samples.push_back(own[i * own.size() / SEARCH_SAMPLES]);

        // compiled a few at a time, so a hopeless kernel stops before compiling all of them
        for (size_t i = 0; i < samples.size(); i += HOPELESS_CONFIGS)
        {
            std::vector<TuneConfig> batch(samples.begin() + i, samples.begin() + std::min(i + HOPELESS_CONFIGS, samples.size()));
            compile_new(batch);
            for (const TuneConfig& c : batch) consider(c);
            if (hopeless() || over_budget()) return std::pair{best, best_own};
        }
        if (own.size() <= EXHAUSTIVE_LIMIT) return std::pair{best, best_own};

        // coordinate descent from the best of a spread of samples: try every option of one param at a time
        // and keep the fastest, until a whole pass finds nothing faster
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

    // the kernels' params don't connect, so one descent could never leave the kernel it starts in
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

    // a probe of each kernel, so the fastest kernel is searched first and sets the bar the others are
    // measured against
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
        // a kernel the device only emulates is dropped on its probe instead of being searched
        if (&part != &parts.front() && part.probe > DROP_FACTOR * best_time) {
            if (tune_log()) std::cerr << "torchvulkan tune: dropping kernel " << part.kernel << " (probe " << part.probe / best_time << "x the best)\n";
            continue;
        }
        auto [config, time] = search(part.space);
        if (time < best_overall) {
            best = config;
            best_overall = time;
        }
    }

    if (tune_log()) std::cerr << "torchvulkan tune: chose " << describe(space, best) << " " << best_overall / 1e6 << " ms after " << elapsed() << " s, "
                              << timed.size() << " configs\n";
    if (best.empty() && first_error) std::rethrow_exception(first_error);
    return best;
}
