#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SDPBackend.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <tuple>
#include <vector>
#include "vulkan/vulkan_context.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"

namespace {

enum class AttentionMode : uint32_t {
    FORWARD = 0,
    BACKWARD_KV = 1,
    BACKWARD_Q = 2
};

const uint32_t ATTENTION_COOP_PAD = 8; // must match PAD in attention_coop.slang

struct Attention {
    at::Tensor q;
    at::Tensor k;
    at::Tensor v;
    at::Tensor bias;
    int64_t B = 0;
    int64_t Hq = 0;
    int64_t Hk = 0;
    int64_t L = 0;
    int64_t S = 0;
    int64_t D = 0;
    int64_t Dv = 0;
    int64_t group = 1;
    bool causal = false;
    double scale = 1.0;
    at::ScalarType dtype = at::kFloat;
    at::ScalarType accumulate = at::kFloat;
};

Attention prepare_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& bias,
    bool causal,
    std::optional<double> scale)
{
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
        "torchvulkan [ERROR]: fused attention takes 4D query, key and value, got ", q.sizes(), ", ", k.sizes(), " and ", v.sizes(), "."
    );
    Attention a;
    a.q = q;
    a.k = k;
    a.v = v;
    a.B = q.size(0);
    a.Hq = q.size(1);
    a.Hk = k.size(1);
    a.L = q.size(2);
    a.S = k.size(2);
    a.D = q.size(3);
    a.Dv = v.size(3);
    TORCH_CHECK(k.size(0) == a.B && v.size(0) == a.B && v.size(1) == a.Hk && v.size(2) == a.S && k.size(3) == a.D,
        "torchvulkan [ERROR]: attention shapes don't match: query ", q.sizes(), ", key ", k.sizes(), ", value ", v.sizes(), "."
    );
    TORCH_CHECK(a.Hk > 0 && a.Hq % a.Hk == 0,
        "torchvulkan [ERROR]: the number of query heads (", a.Hq, ") must be a multiple of the key/value heads (", a.Hk, ")."
    );
    a.group = a.Hq / a.Hk;
    a.causal = causal;
    a.scale = scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(a.D));
    a.dtype = q.scalar_type();
    a.accumulate = reduce_accumulate_dtype(a.dtype);
    if (bias.defined()) a.bias = bias.scalar_type() == a.dtype ? bias : bias.to(a.dtype);
    return a;
}

double neg_inf() { return -std::numeric_limits<double>::infinity(); }
double pos_inf() { return std::numeric_limits<double>::infinity(); }

int64_t chunk_rows(const Attention& a)
{
    DeviceContext* device = get_device_context(a.q);
    int64_t row_elements = a.B * a.Hq * std::max({a.S, a.D, a.Dv});
    // scores, probabilities and two temporaries
    int64_t row_bytes = row_elements * static_cast<int64_t>(c10::elementSize(a.accumulate)) * 4;
    int64_t budget = static_cast<int64_t>(std::max<size_t>(device->memoryFree() / 2, size_t(64) << 20));
    int64_t rows = std::max<int64_t>(1, budget / std::max<int64_t>(row_bytes, 1));
    rows = std::min<int64_t>(rows, std::numeric_limits<int32_t>::max() / std::max<int64_t>(row_elements, 1));
    return std::clamp<int64_t>(rows, 1, std::max<int64_t>(a.L, 1));
}

// [B, Hq, rows, X] as [B, Hk, group * rows, X]
at::Tensor by_kv_head(const at::Tensor& t, const Attention& a)
{
    return t.reshape({a.B, a.Hk, a.group * t.size(2), t.size(3)});
}

at::Tensor chunk_scores(const Attention& a, const at::Tensor& q, const at::Tensor& k, const at::Tensor& bias, int64_t r0, int64_t rows)
{
    at::Tensor scores = at::matmul(by_kv_head(q.narrow(2, r0, rows), a), k.transpose(-2, -1));
    scores = at::mul(scores.reshape({a.B, a.Hq, rows, a.S}), a.scale);
    if (bias.defined()) scores = at::add(scores, bias.narrow(2, r0, rows));
    if (a.causal) {
        at::TensorOptions index = scores.options().dtype(at::kLong);
        at::Tensor row = at::arange(r0, r0 + rows, index).unsqueeze(1);
        at::Tensor col = at::arange(0, a.S, index).unsqueeze(0);
        scores = at::where(at::gt(col, row), at::scalar_tensor(neg_inf(), scores.options()), scores);
    }
    return scores;
}

at::Tensor expanded_bias(const Attention& a)
{
    return a.bias.defined() ? a.bias.expand({a.B, a.Hq, a.L, a.S}) : at::Tensor();
}

std::tuple<at::Tensor, at::Tensor> chunked_forward(const Attention& a, const at::Tensor& out)
{
    at::Tensor q = a.q.to(a.accumulate);
    at::Tensor k = a.k.to(a.accumulate);
    at::Tensor v = a.v.to(a.accumulate);
    at::Tensor bias = expanded_bias(a);
    at::Tensor lse = at::empty({a.B, a.Hq, a.L}, q.options());
    at::Tensor zero = at::scalar_tensor(0.0, q.options());

    int64_t step = chunk_rows(a);
    for (int64_t r0 = 0; r0 < a.L; r0 += step) {
        int64_t rows = std::min(step, a.L - r0);
        at::Tensor scores = chunk_scores(a, q, k, bias, r0, rows);
        at::Tensor max = at::amax(scores, {-1}, true);
        // a row with every key masked attends to nothing
        at::Tensor safe_max = at::where(at::eq(max, neg_inf()), zero, max);
        at::Tensor p = at::exp(at::sub(scores, safe_max));
        at::Tensor sum = at::sum(p, {-1}, true);
        at::Tensor empty = at::eq(sum, 0.0);
        at::Tensor o = at::matmul(by_kv_head(p, a), v).reshape({a.B, a.Hq, rows, a.Dv});
        o = at::div(o, at::where(empty, at::scalar_tensor(1.0, q.options()), sum));
        out.narrow(2, r0, rows).copy_(o);
        lse.narrow(2, r0, rows).copy_(at::where(empty, at::scalar_tensor(pos_inf(), q.options()), at::add(safe_max, at::log(sum))).squeeze(-1));
    }
    return {out, lse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunked_backward(
    const Attention& a,
    const at::Tensor& grad_out,
    const at::Tensor& out,
    const at::Tensor& lse,
    bool want_bias)
{
    at::Tensor q = a.q.to(a.accumulate);
    at::Tensor k = a.k.to(a.accumulate);
    at::Tensor v = a.v.to(a.accumulate);
    at::Tensor dout = grad_out.to(a.accumulate);
    at::Tensor bias = expanded_bias(a);
    at::Tensor delta = at::sum(at::mul(dout, out.to(a.accumulate)), {-1}, true);
    at::Tensor log_sum = lse.to(a.accumulate).unsqueeze(-1);

    at::Tensor grad_q = at::empty({a.B, a.Hq, a.L, a.D}, q.options());
    at::Tensor grad_k = at::zeros({a.B, a.Hk, a.S, a.D}, q.options());
    at::Tensor grad_v = at::zeros({a.B, a.Hk, a.S, a.Dv}, q.options());
    at::Tensor grad_bias;
    bool bias_rows = false;
    if (want_bias) {
        grad_bias = at::zeros(a.bias.sizes(), q.options());
        bias_rows = a.bias.dim() >= 2 && a.bias.size(-2) == a.L && a.L != 1;
    }

    int64_t step = chunk_rows(a);
    for (int64_t r0 = 0; r0 < a.L; r0 += step) {
        int64_t rows = std::min(step, a.L - r0);
        at::Tensor scores = chunk_scores(a, q, k, bias, r0, rows);
        // masked keys and rows that attend to nothing (lse = +inf) get p = 0
        at::Tensor p = at::exp(at::sub(scores, log_sum.narrow(2, r0, rows)));
        at::Tensor dout_rows = by_kv_head(dout.narrow(2, r0, rows), a);

        grad_v.add_(at::matmul(by_kv_head(p, a).transpose(-2, -1), dout_rows));
        at::Tensor dp = at::matmul(dout_rows, v.transpose(-2, -1)).reshape({a.B, a.Hq, rows, a.S});
        at::Tensor ds = at::mul(p, at::sub(dp, delta.narrow(2, r0, rows)));
        at::Tensor dsg = by_kv_head(ds, a);

        grad_q.narrow(2, r0, rows).copy_(at::mul(at::matmul(dsg, k).reshape({a.B, a.Hq, rows, a.D}), a.scale));
        grad_k.add_(at::mul(at::matmul(dsg.transpose(-2, -1), by_kv_head(q.narrow(2, r0, rows), a)), a.scale));

        if (want_bias) {
            std::vector<int64_t> shape(a.bias.sizes().begin(), a.bias.sizes().end());
            if (bias_rows) {
                shape[shape.size() - 2] = rows;
                grad_bias.narrow(-2, r0, rows).copy_(at::sum_to(ds, shape));
            } else {
                grad_bias.add_(at::sum_to(ds, shape));
            }
        }
    }
    return {grad_q.to(a.dtype), grad_k.to(a.dtype), grad_v.to(a.dtype), want_bias ? grad_bias.to(a.bias.scalar_type()) : at::Tensor()};
}

struct AttentionTiling {
    bool coop;
    uint32_t wg;
    uint32_t br;
    uint32_t bc;
    uint32_t block;
    uint32_t subgroup;
    uint32_t splits = 1;
};

AttentionTiling attention_tiling(const TuneConfig& config)
{
    AttentionTiling t{config[0] == 1, config[1], config[2], config[3], config[4], config[5]};
    if (config.values.size() > 6) t.splits = config[6];
    return t;
}

// must match the shared offsets in attention_simd.slang
uint64_t simd_shared(AttentionMode mode, uint64_t br, uint64_t bc, uint64_t d, uint64_t dv)
{
    switch (mode) {
        case AttentionMode::FORWARD: return br * (d + 1) + bc * (std::max(d, dv) + 1) + br * (bc + 1) + br;
        case AttentionMode::BACKWARD_KV: return bc * (d + 1) + bc * (dv + 1) + br * (d + 1) + br * (dv + 1) + 2 * br * (bc + 1) + 2 * br;
        case AttentionMode::BACKWARD_Q: return br * (d + 1) + br * (dv + 1) + bc * (d + 1) + bc * (dv + 1) + br * (bc + 1) + 2 * br;
    }
    return 0;
}

// must match the shared arrays in attention_coop.slang
uint64_t coop_shared_t(uint64_t br, uint64_t bc, uint64_t dp, uint64_t dvp)
{
    return br * (dp + ATTENTION_COOP_PAD) + bc * (std::max(dp, dvp) + ATTENTION_COOP_PAD) + br * (bc + ATTENTION_COOP_PAD);
}

uint64_t coop_shared_a(uint64_t br, uint64_t bc, uint64_t dvp, uint64_t wg)
{
    return br * (std::max(bc, dvp) + ATTENTION_COOP_PAD) + wg;
}

uint64_t coop_backward_shared_t(uint64_t br, uint64_t bc, uint64_t dp, uint64_t dvp)
{
    return (bc + br) * (dp + ATTENTION_COOP_PAD) + (bc + br) * (dvp + ATTENTION_COOP_PAD) + 2 * br * (bc + ATTENTION_COOP_PAD);
}

uint64_t coop_backward_shared_a(uint64_t br, uint64_t bc, uint64_t block, uint64_t subgroups)
{
    return 2 * br * (bc + ATTENTION_COOP_PAD) + 2 * br + subgroups * block * block;
}

uint64_t coop_backward_frags(AttentionMode mode, uint64_t br, uint64_t bc, uint64_t dp, uint64_t dvp, uint64_t block, uint64_t subgroups)
{
    uint64_t blocks = mode == AttentionMode::BACKWARD_KV ? (bc / block) * (dp / block + dvp / block) : (br / block) * (dp / block);
    return ceil_div(blocks, subgroups);
}

int64_t split_keys(const Attention& a, const AttentionTiling& t)
{
    return round_up(ceil_div(a.S, t.splits), t.bc);
}

bool fused_shapes_fit(const Attention& a, const DeviceContext* device)
{
    const auto& limits = device->properties.limits;
    if (!fits_32bit_indexing({a.q, a.k, a.v, a.bias})) return false;
    // outputs and statistics are indexed from each batch's base
    uint64_t per_batch = static_cast<uint64_t>(std::max(a.Hq, a.Hk)) * std::max(a.L, a.S) * std::max(a.D, a.Dv);
    if (per_batch > std::numeric_limits<int32_t>::max()) return false;
    return a.Hq <= limits.maxComputeWorkGroupCount[1] && a.D > 0 && a.Dv > 0;
}

uint64_t attention_tune_key(const Attention& a)
{
    uint64_t key = static_cast<uint64_t>(a.D) | (static_cast<uint64_t>(a.Dv) << 10) | (log2_ceil(a.L) << 20) | (log2_ceil(a.S) << 26) | (log2_ceil(a.B * a.Hq) << 32);
    return key | (log2_ceil(a.group) << 38) | (uint64_t(a.bias.defined()) << 44) | (uint64_t(a.causal) << 45);
}

void push_forward_constants(
    PushConstantBuilder& pcs,
    const Attention& a,
    const at::Tensor& out,
    const at::Tensor& lse,
    int64_t b0,
    int64_t splits)
{
    at::Tensor bias = expanded_bias(a);
    pcs.push(get_batch_address(a.q, b0))
       .push(get_batch_address(a.k, b0))
       .push(get_batch_address(a.v, b0))
       .push(bias.defined() ? get_batch_address(bias, b0) : uint64_t(0))
       .push(get_batch_address(out, b0 * splits))
       .push(get_batch_address(lse, b0 * splits));
    for (const at::Tensor& t : {a.q, a.k, a.v, out}) {
        for (int i = 0; i < 3; i++) pcs.push(static_cast<uint32_t>(t.stride(i)));
    }
    for (int i = 0; i < 4; i++) pcs.push(bias.defined() ? static_cast<uint32_t>(bias.stride(i)) : uint32_t(0));
    pcs.push(static_cast<uint32_t>(a.L))
       .push(static_cast<uint32_t>(a.S))
       .push(static_cast<uint32_t>(a.Hq))
       .push(static_cast<float>(a.scale));
}

TuneDispatch attention_forward_dispatch(
    const Attention& a,
    const TuneConfig& config,
    const at::Tensor& out,
    const at::Tensor& lse,
    int64_t b0,
    int64_t batch)
{
    AttentionTiling t = attention_tiling(config);
    uint32_t D = static_cast<uint32_t>(a.D);
    uint32_t Dv = static_cast<uint32_t>(a.Dv);
    uint32_t keys = t.splits > 1 ? static_cast<uint32_t>(split_keys(a, t)) : 0;

    TuneDispatch d;
    if (t.coop) {
        uint32_t Dp = static_cast<uint32_t>(round_up(D, t.block));
        uint32_t Dvp = static_cast<uint32_t>(round_up(Dv, t.block));
        d.key = ShaderKey{torchvulkan::Kernel::ATTENTION_COOP, a.dtype};
        d.subgroupSize = t.subgroup;
        d.spd.push(t.wg)
             .push(t.subgroup)
             .push(t.br)
             .push(t.bc)
             .push(D)
             .push(Dv)
             .push(Dp)
             .push(Dvp)
             .push(static_cast<uint32_t>(a.group))
             .push(static_cast<uint32_t>(a.bias.defined()))
             .push(static_cast<uint32_t>(a.causal))
             .push(t.block)
             .push(static_cast<uint32_t>(coop_shared_t(t.br, t.bc, Dp, Dvp)))
             .push(static_cast<uint32_t>(coop_shared_a(t.br, t.bc, Dvp, t.wg)))
             .push(t.bc / (t.wg / t.br)) // S_COLS
             .push(Dvp / (t.wg / t.br)) // O_COLS
             .push(t.splits)
             .push(keys);
    } else {
        d.key = ShaderKey{torchvulkan::Kernel::ATTENTION_SIMD, a.dtype};
        d.subgroupSize = get_device_context(a.q)->default_subgroup_size;
        d.spd.push(static_cast<uint32_t>(AttentionMode::FORWARD))
             .push(t.wg)
             .push(t.br)
             .push(t.bc)
             .push(D)
             .push(Dv)
             .push(static_cast<uint32_t>(a.group))
             .push(static_cast<uint32_t>(a.bias.defined()))
             .push(static_cast<uint32_t>(a.causal))
             .push(static_cast<uint32_t>(simd_shared(AttentionMode::FORWARD, t.br, t.bc, D, Dv)))
             .push(static_cast<uint32_t>(ceil_div(t.br * Dv, t.wg))) // ACC_A
             .push(uint32_t(1)) // ACC_B
             .push(t.splits)
             .push(keys);
    }

    push_forward_constants(d.pcs, a, out, lse, b0, t.splits);

    d.read = {a.q, a.k, a.v};
    if (a.bias.defined()) d.read.push_back(a.bias);
    d.write = {out, lse};
    d.groupX = static_cast<uint32_t>(ceil_div(a.L, t.br));
    d.groupY = static_cast<uint32_t>(a.Hq);
    d.groupZ = static_cast<uint32_t>(batch * t.splits);
    return d;
}

// as [B, L, Hq, Dv] underneath when the query is [B, L, Hq, D], so merging the heads back is a view
at::Tensor attention_output(const Attention& a)
{
    if (a.q.stride(1) < a.q.stride(2)) return at::empty({a.B, a.L, a.Hq, a.Dv}, a.q.options()).transpose(1, 2);
    return at::empty({a.B, a.Hq, a.L, a.Dv}, a.q.options());
}

// out is the splits' outputs weighted by exp(lse_split - lse); a split that saw no unmasked key (lse +inf) drops out
void merge_splits(
    const Attention& a,
    int64_t splits,
    const at::Tensor& partial_out,
    const at::Tensor& partial_lse,
    const at::Tensor& out,
    const at::Tensor& lse)
{
    at::Tensor part = partial_lse.view({a.B, splits, a.Hq, a.L});
    at::Tensor attended = at::ne(part, pos_inf());
    at::Tensor split_lse = at::where(attended, part, at::scalar_tensor(neg_inf(), part.options()));
    at::Tensor total = at::logsumexp(split_lse, {1});
    at::Tensor weight = at::where(attended, at::exp(at::sub(split_lse, total.unsqueeze(1))), at::scalar_tensor(0.0, part.options()));
    out.copy_(at::sum(at::mul(partial_out.view({a.B, splits, a.Hq, a.L, a.Dv}), weight.unsqueeze(-1)), {1}));
    lse.copy_(at::where(at::eq(total, neg_inf()), at::scalar_tensor(pos_inf(), total.options()), total));
}

// The SIMD kernels stage whole query and key/value rows in shared memory, so past a 128-wide head their tiles shrink
// until they reread keys and values far more often than the matmul-based path: on an RX 6800 XT (RADV, 64 KiB) the
// fused forward is 2x slower at D=256 and 3x at 320-496, and the backward 20x at D=512. PyTorch likewise stops
// flash attention at 256 and runs larger heads on its math path. So without cooperative matrices those heads go
// chunked. With them it depends on the device (emulated coop loses to chunked, real coop often doesn't), so the
// tuner times the chunked path against the fused kernels it picked.
constexpr int64_t SIMD_MAX_FUSED_HEAD = 128;

bool large_head(const Attention& a) { return std::max(a.D, a.Dv) > SIMD_MAX_FUSED_HEAD; }

// tuned_id bit of the fused-or-chunked choice (attention_tune_key takes bits 0-45, the mode 46-47)
constexpr uint64_t PATH_CHOICE = uint64_t(1) << 48;

// Whether the tuner finds the chunked path faster than `fused`, the fused kernels with their tuned configs.
bool chunked_is_faster(const Attention& a, AttentionMode mode, ShaderKey shader, const std::function<void()>& fused, const std::function<void()>& chunked)
{
    DeviceContext* device = get_device_context(a.q);
    TuneSpace space;
    space.params = {{"chunked", {0, 1}}};
    const TuneConfig& config = device->tuner->select(
        shader,
        attention_tune_key(a) | (static_cast<uint64_t>(mode) << 46) | PATH_CHOICE,
        space,
        [&](const TuneConfig& c) {
            TuneDispatch d;
            d.dispatch = false;
            d.extra = c[0] == 1 ? chunked : fused;
            return d;
        }
    );
    return !config.empty() && config[0] == 1;
}

std::optional<std::tuple<at::Tensor, at::Tensor>> fused_forward(Attention a)
{
    DeviceContext* device = get_device_context(a.q);
    const auto& limits = device->properties.limits;
    if (!is_dtype_supported(device, a.dtype)) return std::nullopt;

    if (a.q.stride(3) != 1) a.q = a.q.contiguous();
    if (a.k.stride(3) != 1) a.k = a.k.contiguous();
    if (a.v.stride(3) != 1) a.v = a.v.contiguous();
    if (!fused_shapes_fit(a, device)) return std::nullopt;
    std::vector<uint32_t> blocks = coop_block_sizes(device, a.dtype, a.accumulate);
    if (blocks.empty() && large_head(a)) return std::nullopt;

    at::Tensor out = attention_output(a);
    at::Tensor lse = at::empty({a.B, a.Hq, a.L}, a.q.options().dtype(a.accumulate));

    std::vector<uint32_t> kernels = {0};
    if (blocks.empty()) {
        blocks = {0};
    } else {
        kernels.push_back(1);
    }

    uint64_t max_threads = std::min(limits.maxComputeWorkGroupInvocations, limits.maxComputeWorkGroupSize[0]);
    uint64_t max_l = uint64_t(1) << log2_ceil(a.L);
    size_t t_bytes = c10::elementSize(a.dtype);
    size_t a_bytes = c10::elementSize(a.accumulate);

    TuneSpace space;
    space.params = {
        {"kernel", kernels},
        {"wg", {64, 128, 256}},
        {"br", {8, 16, 32, 64}},
        {"bc", {16, 32, 64}},
        {"block", blocks},
        {"subgroup", subgroup_size_options(device)},
        {"splits", {1, 2, 4, 8, 16, 32}},
    };
    space.kernel_param = 0;
    space.valid = [&](const TuneConfig& config) {
        AttentionTiling t = attention_tiling(config);
        if (t.wg > max_threads || t.bc > t.wg || t.br > t.wg) return false;
        if (t.splits > 1 && ((t.splits - 1) * split_keys(a, t) >= a.S || t.splits > limits.maxComputeWorkGroupCount[2])) return false;
        if (t.br > std::max<uint64_t>(2 * max_l, 8) || ceil_div(a.L, t.br) > limits.maxComputeWorkGroupCount[0]) return false;
        if (!t.coop) {
            // the SIMD kernel takes only the first option of the cooperative-matrix params
            if (t.block != blocks.front() || t.subgroup != device->min_subgroup_size) return false;
            return a_bytes * simd_shared(AttentionMode::FORWARD, t.br, t.bc, a.D, a.Dv) <= limits.maxComputeSharedMemorySize;
        }
        uint32_t row_threads = t.wg / t.br;
        if (t.wg % t.subgroup != 0 || t.wg % t.br != 0 || t.br % t.block != 0 || t.bc % t.block != 0) return false;
        if (t.bc % row_threads != 0 || row_threads > t.block || !fits_workgroup_size(device, t.wg, t.subgroup)) return false;
        uint64_t Dp = round_up(a.D, t.block);
        uint64_t Dvp = round_up(a.Dv, t.block);
        return t_bytes * coop_shared_t(t.br, t.bc, Dp, Dvp) + a_bytes * coop_shared_a(t.br, t.bc, Dvp, t.wg) <= limits.maxComputeSharedMemorySize;
    };

    // split keys write each split's output and logsumexp in the accumulate dtype, merged after
    auto targets = [&](const AttentionTiling& t) {
        if (t.splits == 1) return std::make_tuple(out, lse);
        at::TensorOptions options = a.q.options().dtype(a.accumulate);
        return std::make_tuple(at::empty({a.B * t.splits, a.Hq, a.L, a.Dv}, options), at::empty({a.B * t.splits, a.Hq, a.L}, options));
    };

    const ShaderKey shader{kernels.size() > 1 ? torchvulkan::Kernel::ATTENTION_COOP : torchvulkan::Kernel::ATTENTION_SIMD, a.dtype};
    const TuneConfig& config = device->tuner->select(
        shader,
        attention_tune_key(a),
        space,
        [&](const TuneConfig& c) {
            AttentionTiling t = attention_tiling(c);
            auto [o, l] = targets(t);
            TuneDispatch d = attention_forward_dispatch(a, c, o, l, 0, std::min<int64_t>(a.B, limits.maxComputeWorkGroupCount[2] / t.splits));
            // charged for the merge, or splitting would look free
            if (t.splits > 1) {
                d.extra = [a, splits = t.splits, o = o, l = l, out, lse]() { merge_splits(a, splits, o, l, out, lse); };
                d.extra_key = t.splits;
            }
            return d;
        }
    );
    if (config.empty()) return std::nullopt;

    AttentionTiling tiling = attention_tiling(config);
    auto [partial_out, partial_lse] = targets(tiling);

    // the batch (times the splits) is the grid's z dimension, so it goes in chunks of the device's limit
    int64_t max_batch = limits.maxComputeWorkGroupCount[2] / tiling.splits;
    auto run = [&]() {
        for (int64_t b0 = 0; b0 < a.B; b0 += max_batch) {
            dispatch_tuned_shader(device, attention_forward_dispatch(a, config, partial_out, partial_lse, b0, std::min(max_batch, a.B - b0)));
        }
        if (tiling.splits > 1) merge_splits(a, tiling.splits, partial_out, partial_lse, out, lse);
    };
    if (large_head(a) && chunked_is_faster(a, AttentionMode::FORWARD, shader, run, [&]() { chunked_forward(a, attention_output(a)); })) {
        return std::nullopt;
    }
    run();
    return std::make_tuple(out, lse);
}

TuneDispatch attention_backward_dispatch(
    AttentionMode mode,
    const Attention& a,
    const TuneConfig& config,
    const at::Tensor& dout,
    const at::Tensor& lse,
    const at::Tensor& delta,
    const at::Tensor& grad_a,
    const at::Tensor& grad_b,
    int64_t b0,
    int64_t batch)
{
    AttentionTiling t = attention_tiling(config);
    uint32_t D = static_cast<uint32_t>(a.D);
    uint32_t Dv = static_cast<uint32_t>(a.Dv);
    at::Tensor bias = expanded_bias(a);

    TuneDispatch d;
    if (t.coop) {
        uint32_t subgroups = t.wg / t.subgroup;
        uint32_t Dp = static_cast<uint32_t>(round_up(D, t.block));
        uint32_t Dvp = static_cast<uint32_t>(round_up(Dv, t.block));
        d.key = ShaderKey{torchvulkan::Kernel::ATTENTION_COOP_BACKWARD, a.dtype};
        d.subgroupSize = t.subgroup;
        d.spd.push(t.wg)
             .push(t.subgroup)
             .push(t.br)
             .push(t.bc)
             .push(D)
             .push(Dv)
             .push(Dp)
             .push(Dvp)
             .push(static_cast<uint32_t>(a.group))
             .push(static_cast<uint32_t>(a.bias.defined()))
             .push(static_cast<uint32_t>(a.causal))
             .push(t.block)
             .push(static_cast<uint32_t>(coop_backward_shared_t(t.br, t.bc, Dp, Dvp)))
             .push(static_cast<uint32_t>(coop_backward_shared_a(t.br, t.bc, t.block, subgroups)))
             .push(uint32_t(1)) // S_COLS
             .push(uint32_t(1)) // O_COLS
             .push(uint32_t(1)) // SPLITS
             .push(uint32_t(0)) // SPLIT_KEYS
             .push(static_cast<uint32_t>(mode))
             .push(static_cast<uint32_t>(coop_backward_frags(mode, t.br, t.bc, Dp, Dvp, t.block, subgroups)));
    } else {
        uint32_t rows = mode == AttentionMode::BACKWARD_KV ? t.bc : t.br;
        d.key = ShaderKey{torchvulkan::Kernel::ATTENTION_SIMD_BACKWARD, a.dtype};
        d.subgroupSize = get_device_context(a.q)->default_subgroup_size;
        d.spd.push(static_cast<uint32_t>(mode))
             .push(t.wg)
             .push(t.br)
             .push(t.bc)
             .push(D)
             .push(Dv)
             .push(static_cast<uint32_t>(a.group))
             .push(static_cast<uint32_t>(a.bias.defined()))
             .push(static_cast<uint32_t>(a.causal))
             .push(static_cast<uint32_t>(simd_shared(mode, t.br, t.bc, D, Dv)))
             .push(static_cast<uint32_t>(ceil_div(rows * D, t.wg))) // ACC_A
             .push(static_cast<uint32_t>(mode == AttentionMode::BACKWARD_KV ? ceil_div(rows * Dv, t.wg) : 1)) // ACC_B
             .push(uint32_t(1)) // SPLITS
             .push(uint32_t(0)); // SPLIT_KEYS
    }

    d.pcs.push(get_batch_address(a.q, b0))
         .push(get_batch_address(a.k, b0))
         .push(get_batch_address(a.v, b0))
         .push(bias.defined() ? get_batch_address(bias, b0) : uint64_t(0))
         .push(get_batch_address(dout, b0))
         .push(get_batch_address(lse, b0))
         .push(get_batch_address(delta, b0))
         .push(get_batch_address(grad_a, b0))
         .push(grad_b.defined() ? get_batch_address(grad_b, b0) : uint64_t(0));
    for (int i = 0; i < 4; i++) d.pcs.push(bias.defined() ? static_cast<uint32_t>(bias.stride(i)) : uint32_t(0));
    d.pcs.push(static_cast<uint32_t>(a.L))
         .push(static_cast<uint32_t>(a.S))
         .push(static_cast<uint32_t>(a.Hq))
         .push(static_cast<float>(a.scale));

    d.read = {a.q, a.k, a.v, dout, lse, delta};
    if (a.bias.defined()) d.read.push_back(a.bias);
    d.write = {grad_a};
    if (grad_b.defined()) d.write.push_back(grad_b);
    d.groupX = static_cast<uint32_t>(mode == AttentionMode::BACKWARD_KV ? ceil_div(a.S, t.bc) : ceil_div(a.L, t.br));
    d.groupY = static_cast<uint32_t>(mode == AttentionMode::BACKWARD_KV ? a.Hk : a.Hq);
    d.groupZ = static_cast<uint32_t>(batch);
    return d;
}

// The tuned config of one backward pass, or nullptr if none fits.
const TuneConfig* tune_backward(
    AttentionMode mode,
    const Attention& a,
    const at::Tensor& dout,
    const at::Tensor& lse,
    const at::Tensor& delta,
    const at::Tensor& grad_a,
    const at::Tensor& grad_b)
{
    DeviceContext* device = get_device_context(a.q);
    const auto& limits = device->properties.limits;
    uint64_t max_threads = std::min(limits.maxComputeWorkGroupInvocations, limits.maxComputeWorkGroupSize[0]);
    size_t t_bytes = c10::elementSize(a.dtype);
    size_t a_bytes = c10::elementSize(a.accumulate);
    int64_t length = mode == AttentionMode::BACKWARD_KV ? a.S : a.L;
    uint64_t max_length = uint64_t(1) << log2_ceil(length);

    std::vector<uint32_t> blocks = coop_block_sizes(device, a.dtype, a.accumulate);
    std::vector<uint32_t> kernels = {0};
    if (blocks.empty()) {
        blocks = {0};
    } else {
        kernels.push_back(1);
    }

    TuneSpace space;
    space.params = {
        {"kernel", kernels},
        {"wg", {64, 128, 256}},
        {"br", {4, 8, 16, 32, 64}},
        {"bc", {8, 16, 32, 64}},
        {"block", blocks},
        {"subgroup", subgroup_size_options(device)},
    };
    space.kernel_param = 0;
    space.valid = [&](const TuneConfig& config) {
        AttentionTiling t = attention_tiling(config);
        if (t.wg > max_threads || t.bc > t.wg || t.wg % t.bc != 0) return false;
        uint64_t owned = mode == AttentionMode::BACKWARD_KV ? t.bc : t.br;
        if (owned > std::max<uint64_t>(2 * max_length, 8) || ceil_div(length, owned) > limits.maxComputeWorkGroupCount[0]) return false;
        if (!t.coop) {
            if (t.block != blocks.front() || t.subgroup != device->min_subgroup_size) return false;
            return a_bytes * simd_shared(mode, t.br, t.bc, a.D, a.Dv) <= limits.maxComputeSharedMemorySize;
        }
        uint32_t subgroups = t.wg / t.subgroup;
        if (t.wg % t.subgroup != 0 || t.br % t.block != 0 || t.bc % t.block != 0) return false;
        if (!fits_workgroup_size(device, t.wg, t.subgroup)) return false;
        uint64_t Dp = round_up(a.D, t.block);
        uint64_t Dvp = round_up(a.Dv, t.block);
        // the gradient blocks stay in registers
        if (coop_backward_frags(mode, t.br, t.bc, Dp, Dvp, t.block, subgroups) * t.block * t.block > 128 * t.subgroup) return false;
        uint64_t shared_bytes = t_bytes * coop_backward_shared_t(t.br, t.bc, Dp, Dvp) + a_bytes * coop_backward_shared_a(t.br, t.bc, t.block, subgroups);
        return shared_bytes <= limits.maxComputeSharedMemorySize;
    };

    int64_t max_batch = limits.maxComputeWorkGroupCount[2];
    const TuneConfig& config = device->tuner->select(
        ShaderKey{kernels.size() > 1 ? torchvulkan::Kernel::ATTENTION_COOP_BACKWARD : torchvulkan::Kernel::ATTENTION_SIMD_BACKWARD, a.dtype},
        attention_tune_key(a) | (static_cast<uint64_t>(mode) << 46),
        space,
        [&](const TuneConfig& c) { return attention_backward_dispatch(mode, a, c, dout, lse, delta, grad_a, grad_b, 0, std::min(max_batch, a.B)); }
    );
    return config.empty() ? nullptr : &config;
}

void run_backward(
    AttentionMode mode,
    const Attention& a,
    const TuneConfig& config,
    const at::Tensor& dout,
    const at::Tensor& lse,
    const at::Tensor& delta,
    const at::Tensor& grad_a,
    const at::Tensor& grad_b)
{
    DeviceContext* device = get_device_context(a.q);
    int64_t max_batch = device->properties.limits.maxComputeWorkGroupCount[2];
    for (int64_t b0 = 0; b0 < a.B; b0 += max_batch) {
        dispatch_tuned_shader(device, attention_backward_dispatch(mode, a, config, dout, lse, delta, grad_a, grad_b, b0, std::min(max_batch, a.B - b0)));
    }
}

std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> fused_backward(
    Attention a,
    const at::Tensor& grad_out,
    const at::Tensor& out,
    const at::Tensor& lse)
{
    DeviceContext* device = get_device_context(a.q);
    if (!is_dtype_supported(device, a.dtype)) return std::nullopt;

    a.q = a.q.contiguous();
    a.k = a.k.contiguous();
    a.v = a.v.contiguous();
    if (!fused_shapes_fit(a, device)) return std::nullopt;
    if (coop_block_sizes(device, a.dtype, a.accumulate).empty() && large_head(a)) return std::nullopt;

    at::Tensor dout = grad_out.to(a.dtype).contiguous();
    at::Tensor stats = lse.to(a.accumulate).contiguous();
    at::Tensor delta = at::sum(at::mul(dout.to(a.accumulate), out.to(a.accumulate)), {-1}).contiguous();

    at::Tensor grad_q = at::empty({a.B, a.Hq, a.L, a.D}, a.q.options());
    at::Tensor grad_k = at::empty({a.B, a.Hk, a.S, a.D}, a.q.options());
    at::Tensor grad_v = at::empty({a.B, a.Hk, a.S, a.Dv}, a.q.options());
    const TuneConfig* kv_config = tune_backward(AttentionMode::BACKWARD_KV, a, dout, stats, delta, grad_k, grad_v);
    if (!kv_config) return std::nullopt;
    const TuneConfig* q_config = tune_backward(AttentionMode::BACKWARD_Q, a, dout, stats, delta, grad_q, at::Tensor());
    if (!q_config) return std::nullopt;

    auto run = [&]() {
        run_backward(AttentionMode::BACKWARD_KV, a, *kv_config, dout, stats, delta, grad_k, grad_v);
        run_backward(AttentionMode::BACKWARD_Q, a, *q_config, dout, stats, delta, grad_q, at::Tensor());
    };
    if (large_head(a)) {
        // timed with its setup, as chunked_backward does its own
        auto fused = [&]() {
            at::sum(at::mul(grad_out.to(a.dtype).to(a.accumulate), out.to(a.accumulate)), {-1});
            run();
        };
        const ShaderKey shader{torchvulkan::Kernel::ATTENTION_COOP_BACKWARD, a.dtype};
        if (chunked_is_faster(a, AttentionMode::BACKWARD_KV, shader, fused, [&]() { chunked_backward(a, grad_out, out, lse, false); })) {
            return std::nullopt;
        }
    }
    run();
    return std::make_tuple(grad_q, grad_k, grad_v);
}

int64_t fused_sdp_choice_vulkan(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_mask,
    double dropout_p,
    bool /* is_causal */,
    std::optional<double> /* scale */,
    bool enable_gqa)
{
    int64_t math = static_cast<int64_t>(at::SDPBackend::math);
    if (!at::globalContext().userEnabledOverrideableSDP() || dropout_p > 0.0) return math;
    if (query.is_nested() || key.is_nested() || value.is_nested()) return math;
    if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) return math;
    at::ScalarType dtype = query.scalar_type();
    if (!is_dtype_float(dtype) || key.scalar_type() != dtype || value.scalar_type() != dtype) return math;
    if (!is_dtype_supported(query, dtype)) return math;
    if (attn_mask.has_value() && attn_mask->defined() && attn_mask->scalar_type() != at::kBool && attn_mask->scalar_type() != dtype) return math;

    int64_t heads = query.size(1);
    int64_t kv_heads = key.size(1);
    if (key.size(0) != query.size(0) || value.size(0) != query.size(0) || value.size(1) != kv_heads) return math;
    if (key.size(2) != value.size(2) || key.size(3) != query.size(3)) return math;
    if (heads != kv_heads && !(enable_gqa && kv_heads > 0 && heads % kv_heads == 0)) return math;
    return static_cast<int64_t>(at::SDPBackend::overrideable);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor> sdpa_forward_vulkan(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    double dropout_p,
    bool is_causal,
    bool /* return_debug_mask */,
    std::optional<double> scale)
{
    TORCH_CHECK(dropout_p == 0.0, "torchvulkan [ERROR]: fused attention doesn't support dropout; it runs on the math backend.");
    at::Tensor bias = attn_bias.has_value() ? *attn_bias : at::Tensor();
    Attention a = prepare_attention(query, key, value, bias, is_causal, scale);

    at::Tensor out, lse;
    std::optional<std::tuple<at::Tensor, at::Tensor>> fused = fused_forward(a);
    if (fused) {
        std::tie(out, lse) = *fused;
    } else {
        std::tie(out, lse) = chunked_forward(a, attention_output(a));
    }

    at::Tensor philox = at::empty({}, at::TensorOptions().dtype(at::kLong));
    return {out, lse, at::Tensor(), at::Tensor(), c10::SymInt(a.L), c10::SymInt(a.S), philox, philox.clone(), at::empty({0}, query.options())};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdpa_backward_vulkan(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& attn_bias,
    std::array<bool, 4> grad_input_mask,
    const at::Tensor& out,
    const at::Tensor& logsumexp,
    const at::Tensor& /* cum_seq_q */,
    const at::Tensor& /* cum_seq_k */,
    int64_t /* max_q */,
    int64_t /* max_k */,
    double dropout_p,
    bool is_causal,
    const at::Tensor& /* philox_seed */,
    const at::Tensor& /* philox_offset */,
    std::optional<double> scale)
{
    TORCH_CHECK(dropout_p == 0.0, "torchvulkan [ERROR]: fused attention doesn't support dropout; it runs on the math backend.");
    Attention a = prepare_attention(query, key, value, attn_bias, is_causal, scale);
    if (!grad_input_mask[0] && !grad_input_mask[1] && !grad_input_mask[2] && !grad_input_mask[3]) return {};

    bool want_bias = grad_input_mask[3] && attn_bias.defined();
    if (!want_bias) {
        std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>> fused = fused_backward(a, grad_out, out, logsumexp);
        if (fused) return {std::get<0>(*fused), std::get<1>(*fused), std::get<2>(*fused), at::Tensor()};
    }
    return chunked_backward(a, grad_out, out, logsumexp, want_bias);
}

} // namespace

// scaled_dot_product_attention asks this stub which backend to run, not the _fused_sdp_choice op
namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &fused_sdp_choice_vulkan);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_fused_sdp_choice", &fused_sdp_choice_vulkan);
    m.impl("_scaled_dot_product_fused_attention_overrideable", &sdpa_forward_vulkan);
    m.impl("_scaled_dot_product_fused_attention_overrideable_backward", &sdpa_backward_vulkan);
}
