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

    TORCH_CHECK(shape.self.size(-1) == shape.other.size(-2), "torchvulkan [ERROR]: matmul inner dimensions must match, got ",
        shape.self.sizes(), " and ", shape.other.sizes(), ".");

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

at::Tensor dispatch_matmul_simd_shader(
    const at::Tensor& self_, 
    const at::Tensor& other_,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    MatmulActivation activation,
    std::function<at::Tensor()> cpu_fallback)
{
    const uint32_t TILE_N = 128;
    const uint32_t TILE_K = 16;
    const uint32_t WG = 16;

    uint32_t has_bias = (bias_.defined() && beta.toDouble() != 0.0) ? 1 : 0;
    MatmulShape shape = get_matmul_shape(self_, other_);
    uint32_t M = shape.M, K = shape.K, N = shape.N;
    int64_t B = shape.B;

    DeviceContext* device = get_device_context(self_);
    c10::ScalarType result_dtype = at::result_type(shape.self, shape.other);
    if (!is_dtype_supported(device, result_dtype) || WG * WG > device->properties.limits.maxComputeWorkGroupInvocations) {
        TORCHVULKAN_CPU_FALLBACK("A ", result_dtype, " matmul can't run on the Vulkan device. Falling back to CPU.");
        return cpu_fallback();
    }

    c10::ScalarType compute_type = matmul_accumulate_dtype(result_dtype);

    TORCH_CHECK(
        activation != MatmulActivation::GELU || at::isFloatingType(result_dtype),
        "torchvulkan [ERROR]: gelu is only defined for floating point matmuls, not ", result_dtype
    );

    uint32_t max_shared_memory = device->properties.limits.maxComputeSharedMemorySize;
    auto shared_bytes = [&](uint32_t tile_m) {
        return c10::elementSize(compute_type) * ((tile_m * (TILE_K + 1)) + (TILE_K * TILE_N));
    };

    uint32_t tile_m = c10::elementSize(result_dtype) < 8 ? 128 : 64;
    if (tile_m == 128 && shared_bytes(128) > max_shared_memory) tile_m = 64;
    if (shared_bytes(tile_m) > max_shared_memory) {
        TORCHVULKAN_CPU_FALLBACK("a ", result_dtype, " matmul tile needs ", shared_bytes(tile_m),
            " bytes of shared memory but the device provides ", max_shared_memory, ". Falling back to CPU.");
        return cpu_fallback();
    }

    MatmulOperands ops = prepare_matmul_operands(shape.self, shape.other, bias_, has_bias, result_dtype, M, K, N, B, shape.batch_shape);
    at::Tensor out = at::empty({B, M, N}, ops.self_b.options());

    if (M == 0 || N == 0 || B == 0) return finish_matmul_output(out, shape);
    if (K == 0) {
        at::Tensor result = has_bias ? (ops.bias_b * beta) : out.zero_();
        return finish_matmul_output(apply_matmul_activation(result, activation), shape);
    }
    if (!fits_matmul_indexing(ops, out)) {
        TORCHVULKAN_CPU_FALLBACK("A matmul is too large for 32-bit indexing. Falling back to CPU.");
        return cpu_fallback();
    }

    torchvulkan::Kernel kernel = tile_m == 128 ? torchvulkan::Kernel::MATMUL_SIMD_128 : torchvulkan::Kernel::MATMUL_SIMD_64;
    uint32_t vecSize = get_dtype_vec_size(result_dtype);
    uint32_t isBiasAligned = has_bias ? ops.bias_b.is_contiguous() : 1;
    uint32_t isAligned = (K % vecSize == 0) && (N % vecSize == 0) &&
                         ops.self_b.is_contiguous() && ops.other_b.is_contiguous() && isBiasAligned;

    SpecializationBuilder spd{};
    spd.push(WG)
       .push(WG)
       .push(isAligned)
       .push(has_bias)
       .push(static_cast<uint32_t>(activation));
    SpecializationArgs specialization = spd.build();

    auto strides_of = [](const at::Tensor& t, uint32_t (&strides)[4]) {
        for (int i = 0; i < 3; i++) strides[i] = static_cast<uint32_t>(t.stride(i));
        strides[3] = 0;
    };
    uint32_t strides_a[4], strides_b[4], strides_c[4];
    uint32_t strides_bias[4] = {0, 0, 0, 0};
    strides_of(ops.self_b, strides_a);
    strides_of(ops.other_b, strides_b);
    strides_of(out, strides_c);
    if (has_bias) strides_of(ops.bias_b, strides_bias);

    std::vector<at::Tensor> readTensors = {ops.self_b, ops.other_b};
    if (has_bias) readTensors.push_back(ops.bias_b);

    // a 1D grid past the limit would be wrapped into rows by dispatchShader, which the tiles don't index for
    TORCH_CHECK((N + TILE_N - 1) / TILE_N <= device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: matmul with N = ", N, " needs more column tiles than the device allows.");

    // the batch is the grid's z dimension, so it goes in chunks of the device's limit
    const int64_t max_batch = device->properties.limits.maxComputeWorkGroupCount[2];
    for (int64_t b0 = 0; b0 < B; b0 += max_batch)
    {
        int64_t batch = std::min(max_batch, B - b0);
        auto address_at = [&](const at::Tensor& t) { return get_tensor_address(t) + b0 * t.stride(0) * t.element_size(); };

        PushConstantBuilder pcs{};
        pcs.push(address_at(ops.self_b));
        pcs.push(address_at(ops.other_b));
        pcs.push(address_at(out));
        pcs.push(has_bias ? address_at(ops.bias_b) : (uint64_t)0); // the shader only reads the bias when it exists
        pcs.push_array(strides_a);
        pcs.push_array(strides_b);
        pcs.push_array(strides_c);
        pcs.push_array(strides_bias);
        pcs.push(M);
        pcs.push(N);
        pcs.push(K);
        pcs.push((uint32_t)0);
        pcs.push_scalar(alpha, compute_type);
        pcs.push_scalar(beta, compute_type);

        device->shader_manager->dispatchShader(
            ShaderKey{kernel, result_dtype},
            specialization,
            pcs.build(),
            /* read = */ readTensors,
            /* write = */ {out},
            (N + TILE_N - 1) / TILE_N, (M + tile_m - 1) / tile_m, static_cast<uint32_t>(batch)
        );
    }

    return finish_matmul_output(out, shape);
}

torchvulkan::Kernel coop_kernel_for_block_size(uint32_t block_size)
{
    switch (block_size)
    {
        case 8: return torchvulkan::Kernel::MATMUL_COOP_8;
        case 16: return torchvulkan::Kernel::MATMUL_COOP_16;
        case 32: return torchvulkan::Kernel::MATMUL_COOP_32;
        case 64: return torchvulkan::Kernel::MATMUL_COOP_64;
        default: TORCH_CHECK(false, "torchvulkan [ERROR]: no cooperative-matrix matmul for block size ", block_size, ".");
    }
}

at::Tensor dispatch_matmul_coop_shader(
    const at::Tensor& self_, 
    const at::Tensor& other_,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::function<at::Tensor()> cpu_fallback)
{
    DeviceContext* device = get_device_context(self_);
    c10::ScalarType promoted_type = at::result_type(self_, other_);
    if (!is_dtype_supported(device, promoted_type)) {
        TORCHVULKAN_CPU_FALLBACK("Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        return cpu_fallback();
    }

    c10::ScalarType accumulate_type = matmul_accumulate_dtype(promoted_type);
    std::vector<CoopMatConfig> config = device->cache.getCoopMatConfig(promoted_type, promoted_type, accumulate_type, accumulate_type);
    if (config.empty()) return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, MatmulActivation::NONE, cpu_fallback);

    std::vector<uint32_t> available_block_sizes;
    for (CoopMatConfig c : config) 
    {
        if (c.m != c.n || c.m != c.k) continue; // we do not support differing block_m and block_n sizes
        available_block_sizes.push_back(c.m);
    }

    MatmulShape shape = get_matmul_shape(self_, other_);
    uint32_t M = shape.M, K = shape.K, N = shape.N;
    int64_t B = shape.B;

    const CoopMatParams& params = device->cache.getCoopMatParams(promoted_type, available_block_sizes, M, N);
    if (!params.is_valid) return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, MatmulActivation::NONE, cpu_fallback);

    if (bias_.defined()) {
        at::Tensor out = dispatch_matmul_coop_shader(self_, other_, {}, alpha, 0, cpu_fallback);
        if (beta.toDouble() == 0.0) return out;
        return torchvulkan::add_vulkan(out, bias_.to(out.scalar_type()), beta);
    }

    if (M == 0 || N == 0 || B == 0) return finish_matmul_output(at::empty({B, M, N}, self_.options().dtype(promoted_type)), shape);
    if (K == 0) return finish_matmul_output(at::zeros({B, M, N}, self_.options().dtype(promoted_type)), shape);

    const uint32_t tile_m = params.block_size * params.warps_m * params.warp_frags_m;
    const uint32_t tile_n = params.block_size_acc * params.warps_n * params.warp_frags_n;

    const uint32_t target_grid = 512;
    uint32_t grid = ((M + tile_m - 1) / tile_m) * ((N + tile_n - 1) / tile_n) * (uint32_t)B;
    uint32_t split = 1;
    if (promoted_type == accumulate_type && B == 1 && grid < target_grid && K >= 4096) {
        while (
            split * 2 * grid <= target_grid && (K % (split * 2)) == 0 &&
            ((K / (split * 2)) % 64) == 0 && (K / (split * 2)) >= 512
        ) {
            split *= 2;
        }
    }

    if (split > 1) {
        at::Tensor a2 = shape.self.reshape({(int64_t)M, (int64_t)K});
        at::Tensor b2 = shape.other.reshape({(int64_t)K, (int64_t)N});
        int64_t ks = K / split;
        at::Tensor a_chunks = a2.as_strided({split, (int64_t)M, ks}, {ks * a2.stride(1), a2.stride(0), a2.stride(1)});
        at::Tensor b_chunks = b2.as_strided({split, ks, (int64_t)N}, {ks * b2.stride(0), b2.stride(0), b2.stride(1)});

        at::Tensor partial = dispatch_matmul_coop_shader(a_chunks, b_chunks, {}, alpha, 0, cpu_fallback);
        for (int64_t s = split / 2; s >= 1; s /= 2) {
            partial = torchvulkan::add_vulkan(partial.narrow(0, 0, s), partial.narrow(0, s, s), 1);
        }
        return finish_matmul_output(partial, shape);
    }

    MatmulOperands ops = prepare_matmul_operands(shape.self, shape.other, at::Tensor(), false, promoted_type, M, K, N, B, shape.batch_shape);

    uint32_t M_padded = ((M + tile_m - 1) / tile_m) * tile_m;
    uint32_t N_padded = ((N + tile_n - 1) / tile_n) * tile_n;
    at::Tensor out = at::empty({B, M_padded, N_padded}, ops.self_b.options());

    auto coop_loadable = [](const at::Tensor& t) { return t.stride(-1) == 1 || t.stride(-2) == 1; };
    if (!coop_loadable(ops.self_b) || !coop_loadable(ops.other_b) || !fits_32bit_indexing({ops.self_b, ops.other_b, out})) {
        return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, MatmulActivation::NONE, cpu_fallback);
    }

    auto strides_of = [](const at::Tensor& t, uint32_t (&strides)[4]) {
        for (int i = 0; i < 3; i++) strides[i] = static_cast<uint32_t>(t.stride(i));
        strides[3] = 0;
    };
    uint32_t strides_a[4], strides_b[4], strides_out[4];
    uint32_t strides_bias[4] = {0, 0, 0, 0};
    strides_of(ops.self_b, strides_a);
    strides_of(ops.other_b, strides_b);
    strides_of(out, strides_out);

    uint32_t self_transposed = (ops.self_b.stride(-1) != 1 && ops.self_b.stride(-2) == 1) ? 1 : 0;
    uint32_t other_transposed = (ops.other_b.stride(-1) != 1 && ops.other_b.stride(-2) == 1) ? 1 : 0;

    const uint32_t vecSize = get_dtype_vec_size(promoted_type);
    uint32_t a_vec_aligned = (!self_transposed && strides_a[1] % vecSize == 0 && strides_a[0] % vecSize == 0 && params.bk % vecSize == 0) ? 1 : 0;
    uint32_t b_vec_aligned = (!other_transposed && strides_b[1] % vecSize == 0 && strides_b[0] % vecSize == 0) ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(params.workgroup_size)
       .push(params.subgroup_size)
       .push(params.warps_m)
       .push(params.warps_n)
       .push(params.warp_frags_m)
       .push(params.warp_frags_n)
       .push(params.bk)
       .push(static_cast<uint32_t>(alpha.toDouble() != 1.0))
       .push(uint32_t(0)) // HAS_BETA
       .push(uint32_t(0)) // HAS_BIAS
       .push(self_transposed)
       .push(other_transposed)
       .push(uint32_t(0)) // OUT_TRANSPOSED
       .push(uint32_t(0)) // BIAS_TRANSPOSED
       .push(static_cast<uint32_t>(M % tile_m == 0))
       .push(static_cast<uint32_t>(N % tile_n == 0))
       .push(a_vec_aligned)
       .push(b_vec_aligned)
       .push(params.block_size)
       .push(static_cast<uint32_t>(params.use_shared_memory));
    SpecializationArgs specialization = spd.build();

    TORCH_CHECK(N_padded / tile_n <= device->properties.limits.maxComputeWorkGroupCount[0],
        "torchvulkan [ERROR]: matmul with N = ", N, " needs more column tiles than the device allows.");

    const int64_t max_batch = device->properties.limits.maxComputeWorkGroupCount[2];
    for (int64_t b0 = 0; b0 < B; b0 += max_batch)
    {
        int64_t batch = std::min(max_batch, B - b0);
        auto address_at = [&](const at::Tensor& t) { return get_tensor_address(t) + b0 * t.stride(0) * t.element_size(); };

        PushConstantBuilder pcs{};
        pcs.push(address_at(ops.self_b))
           .push(address_at(ops.other_b))
           .push(address_at(out))
           .push((uint64_t)0) // no bias
           .push_array(strides_a)
           .push_array(strides_b)
           .push_array(strides_out)
           .push_array(strides_bias)
           .push(M)
           .push(N)
           .push(K)
           .push((uint32_t)0) // padding
           .push_scalar(alpha, accumulate_type)
           .push_scalar(0, accumulate_type);

        device->shader_manager->dispatchShader(
            ShaderKey{coop_kernel_for_block_size(params.block_size), promoted_type},
            specialization,
            pcs.build(),
            /* read = */ {ops.self_b, ops.other_b},
            /* write = */ {out},
            N_padded / tile_n, M_padded / tile_m, static_cast<uint32_t>(batch)
        );
    }

    // torch returns matmul results contiguous, so a padded result is compacted
    if (M_padded != M || N_padded != N) out = out.narrow(1, 0, M).narrow(2, 0, N).contiguous();
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

    if (device->support_coopmat) {
        at::Tensor out = dispatch_matmul_coop_shader(self, other, bias, alpha, beta, cpu_fallback);
        return apply_matmul_activation(out, activation);
    }
    return dispatch_matmul_simd_shader(self, other, bias, alpha, beta, activation, cpu_fallback);
}

at::Tensor mm_vulkan(
    const at::Tensor& self,
    const at::Tensor& other)
{
    return dispatch_matmul_shader(
        self, other, {}, 1, 0, MatmulActivation::NONE,
        [&]() { return at::mm(self.to(at::kCPU), other.to(at::kCPU)).to(self.device()); }
    );
}

at::Tensor bmm_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other)
{
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

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("mm", &mm_vulkan);
    m.impl("bmm", &bmm_vulkan);
    m.impl("addmm", &addmm_vulkan);
    m.impl("baddbmm", &baddbmm_vulkan);
    m.impl("_addmm_activation", &addmm_activation_vulkan);
    m.impl("bmm.out", &bmm_out_vulkan);
    m.impl("addmm.out", &addmm_out_vulkan);
    m.impl("baddbmm.out", &baddbmm_out_vulkan);
    m.impl("_addmm_activation.out", &addmm_activation_out_vulkan);
}
