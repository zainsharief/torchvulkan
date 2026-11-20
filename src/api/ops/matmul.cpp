#include <torch/extension.h>
#include <c10/core/MemoryFormat.h>
#include <functional>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

enum class MatmulActivation : uint32_t
{
    NONE = 0,
    RELU = 1,
    GELU = 2,
};

struct MatmulOperands {
    at::Tensor self_b;
    at::Tensor other_b;
    at::Tensor bias_b;
};

// prepare batched operand views for matmul, handling 2D inputs and optional bias.
MatmulOperands prepare_matmul_operands(
    const at::Tensor& self,
    const at::Tensor& other,
    const at::Tensor& bias_,
    bool has_bias,
    c10::ScalarType promoted_type,
    uint32_t M, uint32_t K, uint32_t N,
    int64_t B,
    const std::vector<int64_t>& batch_shape)
{
    MatmulOperands result;

    if (self.dim() == 2 && other.dim() == 2) 
    {
        result.self_b = self.to(promoted_type).unsqueeze(0);
        result.other_b = other.to(result.self_b.options()).unsqueeze(0);

        if (!has_bias) {
            result.bias_b = bias_;
            return result;
        }

        at::Tensor bias_2d = bias_.to(result.self_b.options());
        if (bias_2d.dim() != 2 || bias_2d.size(0) != (int64_t)M || bias_2d.size(1) != (int64_t)N) {
            bias_2d = bias_2d.expand({(int64_t)M, (int64_t)N});
        }
        result.bias_b = bias_2d.unsqueeze(0);
        return result;
    }

    std::vector<int64_t> self_expand_shape(batch_shape);
    self_expand_shape.push_back(M);
    self_expand_shape.push_back(K);

    std::vector<int64_t> other_expand_shape(batch_shape);
    other_expand_shape.push_back(K);
    other_expand_shape.push_back(N);

    result.self_b = self.to(promoted_type).expand(self_expand_shape).reshape({B, (int64_t)M, (int64_t)K});
    result.other_b = other.to(result.self_b.options()).expand(other_expand_shape).reshape({B, (int64_t)K, (int64_t)N});

    if (!has_bias) {
        result.bias_b = bias_;
        return result;
    }

    std::vector<int64_t> bias_expand_shape(batch_shape);
    bias_expand_shape.push_back(M);
    bias_expand_shape.push_back(N);
    result.bias_b = bias_.to(result.self_b.options()).expand(bias_expand_shape).reshape({B, (int64_t)M, (int64_t)N});
    return result;
}

at::Tensor apply_matmul_activation(
    const at::Tensor& out,
    MatmulActivation activation)
{
    switch (activation)
    {
        case MatmulActivation::RELU: return at::relu(out);
        case MatmulActivation::GELU: return at::gelu(out);
        default: return out;
    }
}

c10::ScalarType matmul_accumulate_dtype(c10::ScalarType dtype)
{
    return (dtype == at::kHalf || dtype == at::kBFloat16) ? at::kFloat : dtype;
}

struct MatmulShape {
    at::Tensor self;
    at::Tensor other;
    bool self_unsqueezed = false;
    bool other_unsqueezed = false;
    uint32_t M = 0, K = 0, N = 0;
    int64_t B = 1;
    std::vector<int64_t> batch_shape;
    std::vector<int64_t> out_shape;
};

MatmulShape get_matmul_shape(const at::Tensor& self_, const at::Tensor& other_)
{
    MatmulShape shape;
    shape.self = self_;
    shape.other = other_;
    if (shape.self.dim() < 2) {
        shape.self = shape.self.unsqueeze(0);
        shape.self_unsqueezed = true;
    }
    if (shape.other.dim() < 2) {
        shape.other = shape.other.unsqueeze(1);
        shape.other_unsqueezed = true;
    }

    TORCH_CHECK(shape.self.scalar_type() == shape.other.scalar_type(), "expected m1 and m2 to have the same dtype, but got: ",
        shape.self.dtype().name(), " != ", shape.other.dtype().name());
    TORCH_CHECK(shape.self.size(-1) == shape.other.size(-2), "mat1 and mat2 shapes cannot be multiplied (", shape.self.size(-2), "x", shape.self.size(-1),
        " and ", shape.other.size(-2), "x", shape.other.size(-1), ")");

    shape.M = static_cast<uint32_t>(shape.self.size(-2));
    shape.K = static_cast<uint32_t>(shape.self.size(-1));
    shape.N = static_cast<uint32_t>(shape.other.size(-1));

    at::IntArrayRef self_batch = shape.self.sizes().slice(0, shape.self.dim() - 2);
    at::IntArrayRef other_batch = shape.other.sizes().slice(0, shape.other.dim() - 2);
    shape.batch_shape = at::infer_size(self_batch, other_batch);
    for (int64_t s : shape.batch_shape) shape.B *= s;

    shape.out_shape = shape.batch_shape;
    shape.out_shape.push_back(shape.M);
    shape.out_shape.push_back(shape.N);
    return shape;
}

at::Tensor finish_matmul_output(const at::Tensor& out, const MatmulShape& shape)
{
    at::Tensor result = out.reshape(shape.out_shape);
    if (shape.self_unsqueezed && shape.other_unsqueezed) return result.squeeze(-1).squeeze(-1);
    if (shape.self_unsqueezed) return result.squeeze(-2);
    if (shape.other_unsqueezed) return result.squeeze(-1);
    return result;
}

bool fits_matmul_indexing(const MatmulOperands& ops, const at::Tensor& out)
{
    return fits_32bit_indexing({ops.self_b, ops.other_b, ops.bias_b, out});
}

const uint32_t MATMUL_TILE_N = 128;
const uint32_t MATMUL_TILE_K = 16;
const uint32_t MATMUL_WG = 16;

void strides_of(const at::Tensor& t, uint32_t (&strides)[4])
{
    for (int i = 0; i < 3; i++) strides[i] = static_cast<uint32_t>(t.stride(i));
    strides[3] = 0;
}

uint64_t matmul_tune_key(const MatmulOperands& ops, uint32_t M, uint32_t N, uint32_t K, int64_t B)
{
    uint64_t self_transposed = ops.self_b.stride(-1) != 1;
    uint64_t other_transposed = ops.other_b.stride(-1) != 1;
    return self_transposed | (other_transposed << 1) | (log2_ceil(M) << 8) | (log2_ceil(N) << 16) | (log2_ceil(K) << 24) | (log2_ceil(B) << 32);
}

// whether the SIMD kernel can run an M x N output with this tile and subgroup size
bool matmul_simd_fits(DeviceContext* device, c10::ScalarType compute_type, uint32_t tile_m, uint32_t subgroup, uint32_t M, uint32_t N)
{
    const auto& limits = device->properties.limits;
    uint64_t shared_bytes = c10::elementSize(compute_type) * ((tile_m * (MATMUL_TILE_K + 1)) + (MATMUL_TILE_K * MATMUL_TILE_N));
    uint64_t max_m = uint64_t(1) << log2_ceil(M);
    uint64_t max_n = uint64_t(1) << log2_ceil(N);
    if (MATMUL_WG * MATMUL_WG > limits.maxComputeWorkGroupInvocations || shared_bytes > limits.maxComputeSharedMemorySize) return false;
    if (ceil_div(max_m, tile_m) > limits.maxComputeWorkGroupCount[1] || ceil_div(max_n, MATMUL_TILE_N) > limits.maxComputeWorkGroupCount[0]) return false;
    return !device->support_subgroup_control || MATMUL_WG * MATMUL_WG <= subgroup * device->max_workgroup_subgroups;
}

// the cooperative-matrix tuner times this too: emulated cooperative matrices can lose to it by far
TuneDispatch matmul_simd_dispatch(
    const MatmulOperands& ops,
    const at::Tensor& out,
    uint32_t has_bias,
    MatmulActivation activation,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    uint32_t tile_m,
    uint32_t subgroup,
    int64_t b0,
    int64_t batch)
{
    c10::ScalarType result_dtype = out.scalar_type();
    c10::ScalarType compute_type = matmul_accumulate_dtype(result_dtype);
    uint32_t M = static_cast<uint32_t>(out.size(1));
    uint32_t N = static_cast<uint32_t>(out.size(2));
    uint32_t K = static_cast<uint32_t>(ops.self_b.size(2));

    uint32_t vecSize = get_dtype_vec_size(result_dtype);
    uint32_t isBiasAligned = has_bias ? ops.bias_b.is_contiguous() : 1;
    uint32_t isAligned = (K % vecSize == 0) && (N % vecSize == 0) &&
                         ops.self_b.is_contiguous() && ops.other_b.is_contiguous() && isBiasAligned;

    uint32_t strides_a[4], strides_b[4], strides_c[4];
    uint32_t strides_bias[4] = {0, 0, 0, 0};
    strides_of(ops.self_b, strides_a);
    strides_of(ops.other_b, strides_b);
    strides_of(out, strides_c);
    if (has_bias) strides_of(ops.bias_b, strides_bias);

    TuneDispatch d;
    d.key = ShaderKey{torchvulkan::Kernel::MATMUL_SIMD, result_dtype};
    d.subgroupSize = subgroup;
    d.spd.push(MATMUL_WG)
         .push(MATMUL_WG)
         .push(isAligned)
         .push(has_bias)
         .push(static_cast<uint32_t>(activation))
         .push(tile_m);

    d.pcs.push(get_batch_address(ops.self_b, b0))
         .push(get_batch_address(ops.other_b, b0))
         .push(get_batch_address(out, b0))
         .push(has_bias ? get_batch_address(ops.bias_b, b0) : uint64_t(0)) // the shader only reads the bias when it exists
         .push_array(strides_a)
         .push_array(strides_b)
         .push_array(strides_c)
         .push_array(strides_bias)
         .push(M)
         .push(N)
         .push(K)
         .push(uint32_t(0)) // padding
         .push_scalar(alpha, compute_type)
         .push_scalar(beta, compute_type);

    d.read = {ops.self_b, ops.other_b};
    if (has_bias) d.read.push_back(ops.bias_b);
    d.write = {out};
    d.groupX = static_cast<uint32_t>(ceil_div(N, MATMUL_TILE_N));
    d.groupY = static_cast<uint32_t>(ceil_div(M, tile_m));
    d.groupZ = static_cast<uint32_t>(batch);
    return d;
}

at::Tensor dispatch_matmul_simd_shader(
    const at::Tensor& self_,
    const at::Tensor& other_,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    MatmulActivation activation,
    std::function<at::Tensor()> cpu_fallback)
{
    uint32_t has_bias = (bias_.defined() && beta.toDouble() != 0.0) ? 1 : 0;
    MatmulShape shape = get_matmul_shape(self_, other_);
    uint32_t M = shape.M, K = shape.K, N = shape.N;
    int64_t B = shape.B;

    DeviceContext* device = get_device_context(self_);
    const auto& limits = device->properties.limits;
    c10::ScalarType result_dtype = at::result_type(shape.self, shape.other);
    if (!is_dtype_supported(device, result_dtype) || MATMUL_WG * MATMUL_WG > limits.maxComputeWorkGroupInvocations) {
        TORCHVULKAN_FALLBACK("A ", result_dtype, " matmul can't run on the Vulkan device. Falling back to CPU.");
        return cpu_fallback();
    }

    c10::ScalarType compute_type = matmul_accumulate_dtype(result_dtype);

    TORCH_CHECK(
        activation != MatmulActivation::GELU || at::isFloatingType(result_dtype),
        "torchvulkan [ERROR]: gelu is only defined for floating point matmuls, not ", result_dtype
    );

    uint32_t max_shared_memory = limits.maxComputeSharedMemorySize;

    MatmulOperands ops = prepare_matmul_operands(shape.self, shape.other, bias_, has_bias, result_dtype, M, K, N, B, shape.batch_shape);
    at::Tensor out = at::empty({B, M, N}, ops.self_b.options());

    if (M == 0 || N == 0 || B == 0) return finish_matmul_output(out, shape);
    if (K == 0) {
        at::Tensor result = has_bias ? (ops.bias_b * beta) : out.zero_();
        return finish_matmul_output(apply_matmul_activation(result, activation), shape);
    }
    if (!fits_matmul_indexing(ops, out)) {
        TORCHVULKAN_FALLBACK("A matmul is too large for 32-bit indexing. Falling back to CPU.");
        return cpu_fallback();
    }

    // a 1D grid past the limit would be wrapped into rows by dispatchShader, which the tiles don't index for
    TORCH_CHECK(ceil_div(N, MATMUL_TILE_N) <= limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: matmul with N = ", N, " needs more column tiles than the device allows.");

    materialize_cow(out);
    auto make_dispatch = [&](const TuneConfig& config, int64_t b0, int64_t batch) {
        return matmul_simd_dispatch(ops, out, has_bias, activation, alpha, beta, config[0], config[1], b0, batch);
    };

    TuneSpace space;
    space.params = {
        {"tile_m", {128, 64}},
        {"subgroup", subgroup_size_options(device)},
    };
    space.valid = [&](const TuneConfig& config) { return matmul_simd_fits(device, compute_type, config[0], config[1], M, N); };

    const int64_t max_batch = limits.maxComputeWorkGroupCount[2];
    const TuneConfig& config = device->tuner->select(
        ShaderKey{torchvulkan::Kernel::MATMUL_SIMD, result_dtype},
        matmul_tune_key(ops, M, N, K, B),
        space,
        [&](const TuneConfig& c) { return make_dispatch(c, 0, std::min(max_batch, B)); }
    );
    if (config.empty()) {
        TORCHVULKAN_FALLBACK("no ", result_dtype, " matmul tile fits in the device's ", max_shared_memory, " bytes of shared memory. Falling back to CPU.");
        return cpu_fallback();
    }

    // the batch is the grid's z dimension, so it goes in chunks of the device's limit
    for (int64_t b0 = 0; b0 < B; b0 += max_batch)
    {
        dispatch_tuned_shader(device, make_dispatch(config, b0, std::min(max_batch, B - b0)));
    }

    return finish_matmul_output(out, shape);
}

CoopTiling matmul_coop_tiling(const TuneConfig& config)
{
    return CoopTiling{config[0], config[1], config[2], config[3], config[4], config[5], config[7], config[6]};
}

// the tuner's pick between the cooperative-matrix kernel and the SIMD one
bool matmul_coop_kernel(const TuneConfig& config)
{
    return config[8] == 1;
}

bool coop_transposed(const at::Tensor& t)
{
    return t.stride(-1) != 1 && t.stride(-2) == 1;
}

// cooperative-matrix loads and stores need 16-byte aligned rows
bool coop_aligned(const at::Tensor& t, bool transposed)
{
    int64_t bytes = t.element_size();
    int64_t lead = transposed ? t.stride(-1) : t.stride(-2);
    return get_tensor_address(t) % 16 == 0 && (lead * bytes) % 16 == 0 && (t.stride(0) * bytes) % 16 == 0;
}

at::Tensor complex_matmul(const at::Tensor& self, const at::Tensor& other, const at::Tensor& bias, const at::Scalar& alpha, const at::Scalar& beta,
                          c10::ScalarType promoted)
{
    at::ScalarType real = c10::toRealValueType(promoted == at::kComplexHalf ? at::kComplexFloat : promoted);
    auto parts_of = [real](const at::Tensor& t) {
        torchvulkan::ComplexParts p = torchvulkan::complex_parts(t);
        return torchvulkan::ComplexParts{p.re.to(real), p.im.to(real)};
    };
    torchvulkan::ComplexParts a = parts_of(self), b = parts_of(other);
    torchvulkan::ComplexParts r{at::sub(at::matmul(a.re, b.re), at::matmul(a.im, b.im)), at::add(at::matmul(a.re, b.im), at::matmul(a.im, b.re))};
    if (alpha.toComplexDouble() != c10::complex<double>(1.0)) r = torchvulkan::complex_mul(r, torchvulkan::complex_scalar_parts(alpha, r.re));
    if (bias.defined() && beta.toComplexDouble() != c10::complex<double>(0.0)) {
        torchvulkan::ComplexParts c = torchvulkan::complex_mul(parts_of(bias), torchvulkan::complex_scalar_parts(beta, r.re));
        r = {at::add(r.re, c.re), at::add(r.im, c.im)};
    }
    return torchvulkan::complex_from(r, promoted);
}

at::Tensor dispatch_matmul_coop_shader(
    const at::Tensor& self_,
    const at::Tensor& other_,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    MatmulActivation activation,
    std::function<at::Tensor()> cpu_fallback)
{
    auto simd_fallback = [&]() { return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, activation, cpu_fallback); };

    DeviceContext* device = get_device_context(self_);
    const auto& limits = device->properties.limits;
    c10::ScalarType promoted_type = at::result_type(self_, other_);
    if (at::isComplexType(promoted_type)) return complex_matmul(self_, other_, bias_, alpha, beta, promoted_type);
    if (!is_dtype_supported(device, promoted_type)) return simd_fallback();

    c10::ScalarType accumulate_type = matmul_accumulate_dtype(promoted_type);
    std::vector<uint32_t> available_block_sizes = coop_block_sizes(device, promoted_type, accumulate_type);
    if (available_block_sizes.empty()) return simd_fallback();

    uint32_t has_bias = (bias_.defined() && beta.toDouble() != 0.0) ? 1 : 0;
    MatmulShape shape = get_matmul_shape(self_, other_);
    uint32_t M = shape.M, K = shape.K, N = shape.N;
    int64_t B = shape.B;

    MatmulOperands ops = prepare_matmul_operands(shape.self, shape.other, bias_, has_bias, promoted_type, M, K, N, B, shape.batch_shape);
    at::Tensor out = at::empty({B, M, N}, ops.self_b.options());

    if (M == 0 || N == 0 || B == 0) return finish_matmul_output(out, shape);
    if (K == 0) {
        at::Tensor result = has_bias ? (ops.bias_b * beta) : out.zero_();
        return finish_matmul_output(apply_matmul_activation(result, activation), shape);
    }

    // the direct path clamps a block's first row/column to m_dim - BLOCK (n_dim - BLOCK), which wraps when
    // the output is smaller than a block: its loads then run past the operands (NaN on an RDNA2 iGPU, a GPU
    // fault and driver reset on an RX 6800 XT)
    available_block_sizes.erase(std::remove_if(available_block_sizes.begin(), available_block_sizes.end(),
        [&](uint32_t block) { return block > M || block > N; }), available_block_sizes.end());
    if (available_block_sizes.empty()) return simd_fallback();

    auto coop_loadable = [](const at::Tensor& t) { return t.stride(-1) == 1 || t.stride(-2) == 1; };
    if (!coop_loadable(ops.self_b) || !coop_loadable(ops.other_b) || !fits_matmul_indexing(ops, out)) return simd_fallback();

    uint32_t strides_a[4], strides_b[4], strides_out[4];
    uint32_t strides_bias[4] = {0, 0, 0, 0};
    strides_of(ops.self_b, strides_a);
    strides_of(ops.other_b, strides_b);
    strides_of(out, strides_out);
    if (has_bias) strides_of(ops.bias_b, strides_bias);

    uint32_t self_transposed = coop_transposed(ops.self_b) ? 1 : 0;
    uint32_t other_transposed = coop_transposed(ops.other_b) ? 1 : 0;
    uint32_t out_aligned = coop_aligned(out, false) ? 1 : 0;

    const uint32_t vecSize = get_dtype_vec_size(promoted_type);
    auto vec_aligned = [&](const at::Tensor& t, uint32_t transposed, const uint32_t (&strides)[4]) {
        uint32_t lead = transposed ? strides[2] : strides[1];
        return get_tensor_address(t) % (vecSize * t.element_size()) == 0 && lead % vecSize == 0 && strides[0] % vecSize == 0;
    };
    uint32_t a_vec_aligned = vec_aligned(ops.self_b, self_transposed, strides_a) ? 1 : 0;
    uint32_t b_vec_aligned = vec_aligned(ops.other_b, other_transposed, strides_b) ? 1 : 0;

    // without shared memory whole blocks load straight from A and B, so M, N and the operands must be block aligned
    uint64_t m_align = std::min<uint64_t>(std::countr_zero(M), 5);
    uint64_t n_align = std::min<uint64_t>(std::countr_zero(N), 5);
    bool direct_aligned = coop_aligned(ops.self_b, self_transposed) && coop_aligned(ops.other_b, other_transposed);

    std::vector<at::Tensor> readTensors = {ops.self_b, ops.other_b};
    if (has_bias) readTensors.push_back(ops.bias_b);

    materialize_cow(out);
    auto make_dispatch = [&](const TuneConfig& config, int64_t b0, int64_t batch) {
        if (!matmul_coop_kernel(config)) return matmul_simd_dispatch(ops, out, has_bias, activation, alpha, beta, config[9], config[7], b0, batch);
        CoopTiling t = matmul_coop_tiling(config);
        uint32_t tile_m = coop_tile_m(t);
        uint32_t tile_n = coop_tile_n(t);
        uint32_t threads = coop_threads(t);

        TuneDispatch d;
        d.key = ShaderKey{torchvulkan::Kernel::MATMUL_COOP, promoted_type};
        d.subgroupSize = t.subgroup;
        d.spd.push(threads)
             .push(t.subgroup)
             .push(t.warps_m)
             .push(t.warps_n)
             .push(t.frags_m)
             .push(t.frags_n)
             .push(t.bk)
             .push(static_cast<uint32_t>(alpha.toDouble() != 1.0))
             .push(static_cast<uint32_t>(activation))
             .push(has_bias)
             .push(self_transposed)
             .push(other_transposed)
             .push(out_aligned)
             .push(static_cast<uint32_t>(ceil_div(tile_m * t.bk / vecSize, threads))) // LOADS_A
             .push(static_cast<uint32_t>(M % tile_m == 0))
             .push(static_cast<uint32_t>(N % tile_n == 0))
             .push(a_vec_aligned)
             .push(b_vec_aligned)
             .push(t.block)
             .push(t.shared)
             .push(static_cast<uint32_t>(ceil_div(t.bk * tile_n / vecSize, threads))); // LOADS_B

        d.pcs.push(get_batch_address(ops.self_b, b0))
             .push(get_batch_address(ops.other_b, b0))
             .push(get_batch_address(out, b0))
             .push(has_bias ? get_batch_address(ops.bias_b, b0) : uint64_t(0))
             .push_array(strides_a)
             .push_array(strides_b)
             .push_array(strides_out)
             .push_array(strides_bias)
             .push(M)
             .push(N)
             .push(K)
             .push(uint32_t(0)) // padding
             .push_scalar(alpha, accumulate_type)
             .push_scalar(beta, accumulate_type);

        d.read = readTensors;
        d.write = {out};
        d.groupX = static_cast<uint32_t>(ceil_div(N, tile_n));
        d.groupY = static_cast<uint32_t>(ceil_div(M, tile_m));
        d.groupZ = static_cast<uint32_t>(batch);
        return d;
    };

    uint64_t max_m = uint64_t(1) << log2_ceil(M);
    uint64_t max_n = uint64_t(1) << log2_ceil(N);
    uint64_t max_b = uint64_t(1) << log2_ceil(B);
    TuneSpace space;
    space.params = {
        {"block", available_block_sizes},
        {"warps_m", {1, 2, 4, 8}},
        {"warps_n", {1, 2, 4, 8}},
        {"frags_m", {1, 2, 4}},
        {"frags_n", {1, 2, 4}},
        {"bk", {8, 16, 32, 64}},
        {"shared", {0, 1}},
        {"subgroup", subgroup_size_options(device)},
        {"kernel", {0, 1}},
        {"simd_tile_m", {128, 64}},
    };
    space.kernel_param = 8;
    space.valid = [&](const TuneConfig& config) {
        // each kernel takes only the first option of the other's params
        for (size_t i = 0; i < 9; i++) {
            if (i != 7 && !matmul_coop_kernel(config) && config[i] != space.params[i].options.front()) return false;
        }
        if (!matmul_coop_kernel(config)) return matmul_simd_fits(device, accumulate_type, config[9], config[7], M, N);
        if (config[9] != space.params[9].options.front()) return false;

        CoopTiling t = matmul_coop_tiling(config);
        uint64_t subgroups = t.warps_m * t.warps_n;
        uint64_t tile_m = coop_tile_m(t);
        uint64_t tile_n = coop_tile_n(t);

        // must match Shared<T> and EpilogueShared in helpers/gemm_coop.slang
        uint64_t stage_a = self_transposed ? t.bk * (tile_m + COOP_PAD) : tile_m * (t.bk + COOP_PAD);
        uint64_t stage_b = other_transposed ? tile_n * (t.bk + COOP_PAD) : t.bk * (tile_n + COOP_PAD);
        uint64_t staged = t.shared ? 2 * (stage_a + stage_b) : tile_m * t.bk + t.bk * tile_n;
        uint64_t shared_bytes = c10::elementSize(promoted_type) * staged + c10::elementSize(accumulate_type) * subgroups * t.block * t.block;

        if (t.bk % t.block != 0 || shared_bytes > limits.maxComputeSharedMemorySize) return false;
        if (!t.shared && (!direct_aligned || (1u << m_align) % t.block != 0 || (1u << n_align) % t.block != 0)) return false;
        if (subgroups > 8 || !fits_workgroup_size(device, coop_threads(t), t.subgroup)) return false;
        if (tile_m > std::max<uint64_t>(2 * max_m, t.block) || tile_n > std::max<uint64_t>(2 * max_n, t.block)) return false;
        if (ceil_div(max_n, tile_n) > limits.maxComputeWorkGroupCount[0] || ceil_div(max_m, tile_m) > limits.maxComputeWorkGroupCount[1]) return false;
        return max_b * round_up(max_m, tile_m) * round_up(max_n, tile_n) <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max());
    };

    const int64_t max_batch = limits.maxComputeWorkGroupCount[2];
    const TuneConfig& config = device->tuner->select(
        ShaderKey{torchvulkan::Kernel::MATMUL_COOP, promoted_type},
        matmul_tune_key(ops, M, N, K, B) | (m_align << 40) | (n_align << 43) | (uint64_t(direct_aligned) << 46),
        space,
        [&](const TuneConfig& c) { return make_dispatch(c, 0, std::min(max_batch, B)); }
    );
    if (config.empty()) return simd_fallback();

    CoopTiling tiling = matmul_coop_tiling(config);
    const int64_t target_grid = 512;
    int64_t grid = ceil_div(M, coop_tile_m(tiling)) * ceil_div(N, coop_tile_n(tiling)) * B;
    int64_t split = 1;
    if (matmul_coop_kernel(config) && promoted_type == accumulate_type && B == 1 && grid < target_grid && K >= 4096) {
        while (
            split * 2 * grid <= target_grid && (K % (split * 2)) == 0 &&
            ((K / (split * 2)) % 64) == 0 && (K / (split * 2)) >= 512
        ) {
            split *= 2;
        }
    }

    if (split > 1) {
        at::Tensor a2 = ops.self_b.reshape({(int64_t)M, (int64_t)K});
        at::Tensor b2 = ops.other_b.reshape({(int64_t)K, (int64_t)N});
        int64_t ks = K / split;
        at::Tensor a_chunks = a2.as_strided({split, (int64_t)M, ks}, {ks * a2.stride(1), a2.stride(0), a2.stride(1)});
        at::Tensor b_chunks = b2.as_strided({split, ks, (int64_t)N}, {ks * b2.stride(0), b2.stride(0), b2.stride(1)});

        at::Tensor partial = dispatch_matmul_coop_shader(a_chunks, b_chunks, {}, alpha, 0, MatmulActivation::NONE, cpu_fallback);
        for (int64_t s = split / 2; s >= 1; s /= 2) {
            partial = torchvulkan::add_vulkan(partial.narrow(0, 0, s), partial.narrow(0, s, s), 1);
        }
        at::Tensor result = finish_matmul_output(partial, shape);
        if (has_bias) result = torchvulkan::add_vulkan(result, bias_.to(result.scalar_type()), beta);
        return apply_matmul_activation(result, activation);
    }

    for (int64_t b0 = 0; b0 < B; b0 += max_batch)
    {
        dispatch_tuned_shader(device, make_dispatch(config, b0, std::min(max_batch, B - b0)));
    }

    return finish_matmul_output(out, shape);
}

at::Tensor dispatch_matmul_shader(
    const at::Tensor& self, 
    const at::Tensor& other,
    const at::Tensor& bias,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    MatmulActivation activation,
    std::function<at::Tensor()> cpu_fallback)
{
    DeviceContext* device = get_device_context(self);

    if (device->support_coopmat) return dispatch_matmul_coop_shader(self, other, bias, alpha, beta, activation, cpu_fallback);
    return dispatch_matmul_simd_shader(self, other, bias, alpha, beta, activation, cpu_fallback);
}

// the CPU's mm / addmm checks (mat1 is "self" in its messages)
void check_matrix_product(const char* name, const at::Tensor& mat1, const at::Tensor& mat2)
{
    TORCH_CHECK(mat1.dim() == 2, name, ": self must be a matrix, got ", mat1.dim(), "-D tensor");
    TORCH_CHECK(mat2.dim() == 2, name, ": mat2 must be a matrix, got ", mat2.dim(), "-D tensor");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), name, ": mat1 and mat2 shapes cannot be multiplied (", mat1.size(0), "x", mat1.size(1), " and ",
        mat2.size(0), "x", mat2.size(1), ")");
}

at::Tensor mm_vulkan(
    const at::Tensor& self,
    const at::Tensor& other)
{
    check_matrix_product("mm", self, other);
    return dispatch_matmul_shader(
        self, other, {}, 1, 0, MatmulActivation::NONE,
        [&]() { return at::mm(self.to(at::kCPU), other.to(at::kCPU)).to(self.device()); }
    );
}

at::Tensor bmm_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other)
{
    // the structured meta function's checks
    TORCH_CHECK(self.dim() == 3, "batch1 must be a 3D tensor");
    TORCH_CHECK(other.dim() == 3, "batch2 must be a 3D tensor");
    TORCH_CHECK(self.device() == other.device(), "Expected all tensors to be on the same device, but found at least two devices, ", self.device(),
        " and ", other.device(), "!");
    TORCH_CHECK(self.scalar_type() == other.scalar_type(), "expected scalar type ", self.scalar_type(), " but found ", other.scalar_type());
    TORCH_CHECK(other.size(0) == self.size(0) && other.size(1) == self.size(2), "Expected size for first two dimensions of batch2 tensor to be: [",
        self.size(0), ", ", self.size(2), "] but got: [", other.size(0), ", ", other.size(1), "].");
    return dispatch_matmul_shader(
        self, other, {}, 1, 0, MatmulActivation::NONE,
        [&]() { return at::bmm(self.to(at::kCPU), other.to(at::kCPU)).to(self.device()); }
    );
}

at::Tensor addmm_vulkan(
    const at::Tensor& input, 
    const at::Tensor& mat1, 
    const at::Tensor& mat2, 
    const at::Scalar& beta, 
    const at::Scalar& alpha) 
{
    for (const at::Tensor* t : {&mat1, &mat2}) {
        TORCH_CHECK(t->device() == input.device(), "Expected all tensors to be on the same device, but found at least two devices, ", input.device(),
            " and ", t->device(), "!");
    }
    check_matrix_product("addmm", mat1, mat2);
    return dispatch_matmul_shader(
        mat1, mat2, input, alpha, beta, MatmulActivation::NONE,
        [&]() { return at::addmm(input.to(at::kCPU), mat1.to(at::kCPU), mat2.to(at::kCPU), beta, alpha).to(mat1.device()); }
    );
}

at::Tensor baddbmm_vulkan(
    const at::Tensor& input, 
    const at::Tensor& mat1, 
    const at::Tensor& mat2, 
    const at::Scalar& beta, 
    const at::Scalar& alpha)
{
    TORCH_CHECK(input.scalar_type() == mat1.scalar_type() && mat1.scalar_type() == mat2.scalar_type(), "Input dtypes must be the same, got: input ",
        input.dtype().name(), ", batch1: ", mat1.dtype().name(), ", batch2: ", mat2.dtype().name());
    return dispatch_matmul_shader(
        mat1, mat2, input, alpha, beta, MatmulActivation::NONE,
        [&]() { return at::baddbmm(input.to(at::kCPU), mat1.to(at::kCPU), mat2.to(at::kCPU), beta, alpha).to(mat1.device()); }
    );
}

at::Tensor addmm_activation_vulkan(
    const at::Tensor& input,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    bool use_gelu)
{
    MatmulActivation activation = use_gelu ? MatmulActivation::GELU : MatmulActivation::RELU;
    return dispatch_matmul_shader(
        mat1, mat2, input, alpha, beta, activation,
        [&]() { return at::_addmm_activation(input.to(at::kCPU), mat1.to(at::kCPU), mat2.to(at::kCPU), beta, alpha, use_gelu).to(mat1.device()); }
    );
}

at::Tensor& bmm_out_vulkan(const at::Tensor& self, const at::Tensor& other, at::Tensor& out)
{
    check_out_device(out, self);
    return fill_out(out, bmm_vulkan(self, other));
}

at::Tensor& addmm_out_vulkan(const at::Tensor& input, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out)
{
    return fill_out(out, addmm_vulkan(input, mat1, mat2, beta, alpha));
}

at::Tensor& baddbmm_out_vulkan(const at::Tensor& input, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha, at::Tensor& out)
{
    return fill_out(out, baddbmm_vulkan(input, mat1, mat2, beta, alpha));
}

at::Tensor& addmm_activation_out_vulkan(const at::Tensor& input, const at::Tensor& mat1, const at::Tensor& mat2, const at::Scalar& beta, const at::Scalar& alpha, bool use_gelu, at::Tensor& out)
{
    return fill_out(out, addmm_activation_vulkan(input, mat1, mat2, beta, alpha, use_gelu));
}

// torch's _scaled_mm_out_cpu_emulated: float8 operands upcast and scaled (one scale each, or a row of A by a column of B),
// multiplied in float32
at::Tensor& scaled_mm_out_vulkan(const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& scale_a, const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias, const std::optional<at::Tensor>& scale_result, std::optional<at::ScalarType> out_dtype,
    bool /* use_fast_accum */, at::Tensor& out)
{
    TORCH_CHECK_VALUE(!out_dtype || *out_dtype == out.scalar_type(), "out_dtype must match output matrix type");
    TORCH_CHECK(mat1.dim() == 2, "mat1 must be a matrix");
    TORCH_CHECK(mat2.dim() == 2, "mat2 must be a matrix");
    TORCH_CHECK(mat1.size(1) == mat2.size(0), "mat_a and mat_b shapes cannot be multiplied (", mat1.size(0), "x", mat1.size(1), " and ",
        mat2.size(0), "x", mat2.size(1), ")");
    TORCH_CHECK_VALUE(at::isFloat8Type(mat1.scalar_type()) && at::isFloat8Type(mat2.scalar_type()), "mat1 and mat2 must be fp8 types, got: ",
        mat1.scalar_type(), mat2.scalar_type());
    bool tensorwise = scale_a.numel() == 1 && scale_b.numel() == 1;
    bool rowwise = scale_a.dim() == 2 && scale_b.dim() == 2 && scale_a.size(0) == mat1.size(0) && scale_a.size(1) == 1 &&
        scale_b.size(0) == 1 && scale_b.size(1) == mat2.size(1);
    TORCH_CHECK_VALUE(tensorwise || rowwise, "Invalid scaling configuration");
    TORCH_CHECK_VALUE(scale_a.scalar_type() == at::kFloat, "scale_a must have Float elements");
    TORCH_CHECK_VALUE(scale_b.scalar_type() == at::kFloat, "scale_b must have Float elements");
    TORCH_CHECK(!scale_result || (scale_result->numel() == 1 && scale_result->scalar_type() == at::kFloat), "scale_result must be a float scalar");
    TORCH_CHECK(!bias || bias->numel() == mat2.size(1), "Bias must be size ", mat2.size(1), " but got ", bias->numel());

    at::native::resize_output(out, {mat1.size(0), mat2.size(1)});
    at::ScalarType target = out_dtype.value_or(at::kBFloat16);
    at::Tensor a = at::mul(mat1.contiguous().to(at::kFloat), tensorwise ? scale_a.reshape({}) : scale_a);
    at::Tensor b = at::mul(mat2.contiguous().to(at::kFloat), tensorwise ? scale_b.reshape({}) : scale_b);
    at::Tensor product = at::matmul(a, b);
    if (bias) product.add_(*bias);
    if (scale_result && (target == at::kFloat8_e4m3fn || target == at::kFloat8_e5m2)) {
        product = at::mul(product, at::reciprocal(scale_result->reshape({})));
    }
    out.copy_(product.to(out.scalar_type()));
    return out;
}

at::Tensor scaled_mm_vulkan(const at::Tensor& mat1, const at::Tensor& mat2, const at::Tensor& scale_a, const at::Tensor& scale_b,
    const std::optional<at::Tensor>& bias, const std::optional<at::Tensor>& scale_result, std::optional<at::ScalarType> out_dtype,
    bool use_fast_accum)
{
    at::Tensor out = at::empty({0}, mat1.options().dtype(out_dtype.value_or(mat1.scalar_type())));
    return scaled_mm_out_vulkan(mat1, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mm", &mm_vulkan);
    m.impl("_scaled_mm", &scaled_mm_vulkan);
    m.impl("_scaled_mm.out", &scaled_mm_out_vulkan);
    m.impl("bmm", &bmm_vulkan);
    m.impl("addmm", &addmm_vulkan);
    m.impl("baddbmm", &baddbmm_vulkan);
    m.impl("_addmm_activation", &addmm_activation_vulkan);
    m.impl("bmm.out", &bmm_out_vulkan);
    m.impl("addmm.out", &addmm_out_vulkan);
    m.impl("baddbmm.out", &baddbmm_out_vulkan);
    m.impl("_addmm_activation.out", &addmm_activation_out_vulkan);
}
