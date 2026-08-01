#include <torch/extension.h>
#include "api/ops/matmul.h"
#include "api/ops/binary.h"

namespace {

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

} // namespace

at::Tensor torchvulkan::dispatch_matmul_shader(
    const at::Tensor& self, 
    const at::Tensor& other,
    const at::Tensor& bias,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::function<at::Tensor()> cpu_fallback)
{
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    if (device->support_coopmat) return dispatch_matmul_coop_shader(self, other, bias, alpha, beta, cpu_fallback);
    return dispatch_matmul_simd_shader(self, other, bias, alpha, beta, cpu_fallback);
}

at::Tensor torchvulkan::dispatch_matmul_coop_shader(
    const at::Tensor& self_, 
    const at::Tensor& other_,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::function<at::Tensor()> cpu_fallback)
{
    at::Tensor self = self_;
    at::Tensor other = other_;
    c10::ScalarType promoted_type = at::result_type(self, other);

    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        return cpu_fallback();
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    std::vector<CoopMatConfig> config = device->cache.getCoopMatConfig(promoted_type, promoted_type, promoted_type, promoted_type);
    if (config.empty()) return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, cpu_fallback);
    std::vector<uint32_t> available_block_sizes;
    available_block_sizes.reserve(config.size());

    for (CoopMatConfig c : config) 
    {
        if (c.m != c.n || c.m != c.k) continue; // we do not support differing block_m and block_n sizes
        available_block_sizes.push_back(c.m);
    }

    bool self_unsqueezed = false;
    bool other_unsqueezed = false;

    if (self.dim() < 2) {
        self = self.unsqueeze(0);
        self_unsqueezed = true;
    }
    if (other.dim() < 2) {
        other = other.unsqueeze(1);
        other_unsqueezed = true;
    }

    TORCH_CHECK(self.size(-1) == other.size(-2), "torchvulkan: inner dimensions must match");

    uint32_t M = self.size(-2);
    uint32_t K = self.size(-1);
    uint32_t N = other.size(-1);

    CoopMatParams* params = device->cache.getCoopMatParams(promoted_type, available_block_sizes, M, N);
    if (!params->is_valid) return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, cpu_fallback);

    if (bias_.defined()) {
        at::Tensor out = dispatch_matmul_coop_shader(self_, other_, {}, alpha, 0, cpu_fallback);
        if (beta.toDouble() == 0.0) return out;
        return add_vulkan(out, bias_.to(out.scalar_type()), beta);
    }

    uint32_t has_alpha = (alpha.toDouble() != 1.0) ? 1 : 0;
    uint32_t has_bias = (bias_.defined()) ? 1 : 0;
    uint32_t has_beta = (!has_bias && beta.toDouble() != 0.0) ? 1 : 0;
    
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_matmul_coop(promoted_type, params->block_size);

    at::IntArrayRef self_batch = self.sizes().slice(0, self.dim() - 2);
    at::IntArrayRef other_batch = other.sizes().slice(0, other.dim() - 2);
    std::vector<int64_t> batch_shape = at::infer_size(self_batch, other_batch);

    int64_t B = 1;
    for (int64_t s : batch_shape) {
        B *= s;
    }

    std::vector<int64_t> out_shape_vec = batch_shape;
    out_shape_vec.push_back(M);
    out_shape_vec.push_back(N);
    at::IntArrayRef out_shape(out_shape_vec);

    if (M == 0 || N == 0 || B == 0) return at::empty({B, M, N}, self_.options()).reshape(out_shape);
    if (K == 0) {
        if (!has_bias) return at::zeros({B, M, N}, self_.options().dtype(promoted_type)).reshape(out_shape);
        std::vector<int64_t> bias_expand_shape(batch_shape);
        bias_expand_shape.push_back(M);
        bias_expand_shape.push_back(N);
        at::Tensor bias_b = bias_.to(promoted_type).expand(bias_expand_shape).reshape({B, M, N});
        return (bias_b * beta).reshape(out_shape);
    }

    const uint32_t tile_m = params->block_size * params->warps_m * params->warp_frags_m;
    const uint32_t tile_n = params->block_size_acc * params->warps_n * params->warp_frags_n;

    // might be better ways than to just hardcode this
    const uint32_t target_grid = 512;
    uint32_t grid = ((M + tile_m - 1) / tile_m) * ((N + tile_n - 1) / tile_n) * (uint32_t)B;
    uint32_t split = 1;
    if (
        B == 1 && 
        grid < target_grid && 
        K >= 4096 && 
        self.storage_offset() == 0 && 
        other.storage_offset() == 0
    ) {
        while (
            split * 2 * grid <= target_grid && (K % (split * 2)) == 0 &&
            ((K / (split * 2)) % 64) == 0 && (K / (split * 2)) >= 512
        ) {
            split *= 2;
        }
    }

    if (split > 1) {
        at::Tensor a2 = self.reshape({(int64_t)M, (int64_t)K});
        at::Tensor b2 = other.reshape({(int64_t)K, (int64_t)N});
        int64_t ks = K / split;
        at::Tensor a_chunks = a2.as_strided({split, (int64_t)M, ks}, {ks * a2.stride(1), a2.stride(0), a2.stride(1)});
        at::Tensor b_chunks = b2.as_strided({split, ks, (int64_t)N}, {ks * b2.stride(0), b2.stride(0), b2.stride(1)});

        at::Tensor partial = dispatch_matmul_coop_shader(a_chunks, b_chunks, {}, alpha, 0, cpu_fallback);
        for (int64_t s = split / 2; s >= 1; s /= 2) {
            partial = add_vulkan(partial.narrow(0, 0, s), partial.narrow(0, s, s), 1);
        }

        at::Tensor out = partial.reshape(out_shape);
        if (self_unsqueezed && other_unsqueezed) out = out.squeeze(-1).squeeze(-1);
        else if (self_unsqueezed) out = out.squeeze(-2);
        else if (other_unsqueezed) out = out.squeeze(-1);
        return out;
    }

    MatmulOperands ops = prepare_matmul_operands(self, other, bias_, has_bias, promoted_type, M, K, N, B, batch_shape);
    at::Tensor self_b = ops.self_b;
    at::Tensor other_b = ops.other_b;
    at::Tensor bias_b = ops.bias_b;
    uint32_t M_padded = ((M + tile_m - 1) / tile_m) * tile_m;
    uint32_t N_padded = ((N + tile_n - 1) / tile_n) * tile_n;
    at::Tensor out = at::empty({B, M_padded, N_padded}, self_b.options());

    uint32_t strides_a[4] = {static_cast<uint32_t>(self_b.stride(0)), static_cast<uint32_t>(self_b.stride(1)), static_cast<uint32_t>(self_b.stride(2)), 0};
    uint32_t strides_b[4] = {static_cast<uint32_t>(other_b.stride(0)), static_cast<uint32_t>(other_b.stride(1)), static_cast<uint32_t>(other_b.stride(2)), 0};
    uint32_t strides_out[4] = {static_cast<uint32_t>(out.stride(0)), static_cast<uint32_t>(out.stride(1)), static_cast<uint32_t>(out.stride(2)), 0};
    uint32_t strides_bias[4] = {0, 0, 0, 0};
    if (has_bias) {strides_bias[0] = static_cast<uint32_t>(bias_b.stride(0)); strides_bias[1] = static_cast<uint32_t>(bias_b.stride(1)); strides_bias[2] = static_cast<uint32_t>(bias_b.stride(2));}

    uint32_t self_transposed = (self_b.stride(-1) != 1 && self_b.stride(-2) == 1) ? 1 : 0;
    uint32_t other_transposed = (other_b.stride(-1) != 1 && other_b.stride(-2) == 1) ? 1 : 0;
    uint32_t out_transposed = (out.stride(-1) != 1 && out.stride(-2) == 1) ? 1 : 0;
    uint32_t bias_transposed = (has_bias && bias_b.stride(-1) != 1 && bias_b.stride(-2) == 1) ? 1 : 0;

    uint32_t m_aligned = (M % tile_m == 0) ? 1 : 0;
    uint32_t n_aligned = (N % tile_n == 0) ? 1 : 0;

    auto coop_loadable = [](const at::Tensor& t) { return t.stride(-1) == 1 || t.stride(-2) == 1; };
    if (!coop_loadable(self_b) || !coop_loadable(other_b) || !coop_loadable(out) ||
        (has_bias && !coop_loadable(bias_b))) {
        return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, cpu_fallback);
    }
    
    PushConstantBuilder pcs{};
    pcs.push_array(strides_a)
       .push_array(strides_b)
       .push_array(strides_out)
       .push_array(strides_bias)
       .push(M)
       .push(N)
       .push(K)
       .push((uint32_t)0) // padding
       .push_scalar(alpha, promoted_type)
       .push_scalar(beta, promoted_type);

    SpecializationBuilder spd{};
    spd.push(params->workgroup_size)
       .push(params->subgroup_size)
       .push(params->warps_m)
       .push(params->warps_n)
       .push(params->warp_frags_m)
       .push(params->warp_frags_n)
       .push(params->bk)
       .push(has_alpha)
       .push(has_beta)
       .push(has_bias)
       .push(self_transposed)
       .push(other_transposed)
       .push(out_transposed)
       .push(bias_transposed)
       .push(m_aligned)
       .push(n_aligned);
    uint64_t key = ((uint64_t)m_aligned << 51) | ((uint64_t)n_aligned << 50) |
                   ((uint64_t)self_transposed << 49) | ((uint64_t)other_transposed << 48) |
                   ((uint64_t)out_transposed << 47) | ((uint64_t)bias_transposed << 46) |
                   ((uint64_t)has_bias << 45) | ((uint64_t)has_beta << 44) |
                   ((uint64_t)has_alpha << 43) | ((uint64_t)params->bk << 36) |
                   ((uint64_t)params->warp_frags_n << 32) | ((uint64_t)params->warp_frags_m << 28) |
                   ((uint64_t)params->warps_n << 24) | ((uint64_t)params->warps_m << 20) |
                   ((uint64_t)params->subgroup_size << 12) | ((uint64_t)params->workgroup_size);
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};
    VulkanShader shader(shader_id, specialization, device);

    uint32_t groupX = N_padded / tile_n;
    uint32_t groupY = M_padded / tile_m;
    uint32_t groupZ = B;

    shader.dispatch(
        &pcs,
        pcs.size(),
        {self_b, other_b, out, has_bias ? bias_b : out},
        groupX, groupY, groupZ
    );

    out = out.narrow(1, 0, M).narrow(2, 0, N);
    out = out.reshape(out_shape);
    if (self_unsqueezed || other_unsqueezed) {
        if (self_unsqueezed && other_unsqueezed) out = out.squeeze(-1).squeeze(-1); 
        else if (self_unsqueezed) out = out.squeeze(-2); 
        else out = out.squeeze(-1); 
    }

    return out;
}

at::Tensor torchvulkan::dispatch_matmul_simd_shader(
    const at::Tensor& self_, 
    const at::Tensor& other_,
    const at::Tensor& bias_,
    const at::Scalar& alpha,
    const at::Scalar& beta,
    std::function<at::Tensor()> cpu_fallback)
{
    // must match the tile sizes defined in the shader - if you change one, change the other!
    static const uint32_t TILE_M = 128;
    static const uint32_t TILE_N = 128;
    static const uint32_t TILE_K = 16;

    at::Tensor self = self_;
    at::Tensor other = other_;
    uint32_t has_bias = (bias_.defined() && beta.toDouble() != 0.0) ? 1 : 0;
    
    bool self_unsqueezed = false;
    bool other_unsqueezed = false;

    if (self.dim() < 2) {
        self = self.unsqueeze(0);
        self_unsqueezed = true;
    }
    if (other.dim() < 2) {
        other = other.unsqueeze(1);
        other_unsqueezed = true;
    }

    TORCH_CHECK(self.size(-1) == other.size(-2), "torchvulkan: inner dimensions must match");

    uint32_t M = self.size(-2);
    uint32_t K = self.size(-1);
    uint32_t N = other.size(-1);

    at::IntArrayRef self_batch = self.sizes().slice(0, self.dim() - 2);
    at::IntArrayRef other_batch = other.sizes().slice(0, other.dim() - 2);
    std::vector<int64_t> batch_shape = at::infer_size(self_batch, other_batch);

    int64_t B = 1;
    for (int64_t s : batch_shape) {
        B *= s;
    }

    std::vector<int64_t> out_shape_vec = batch_shape;
    out_shape_vec.push_back(M);
    out_shape_vec.push_back(N);
    at::IntArrayRef out_shape(out_shape_vec);

    c10::ScalarType promoted_type = at::result_type(self, other);
    if (!is_dtype_supported(promoted_type)) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", promoted_type, ". Falling back to CPU.");
        return cpu_fallback();
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t required_bytes = c10::elementSize(promoted_type) * ((TILE_M * (TILE_K + 1)) + (TILE_K * TILE_N));
    uint32_t device_tile_m = TILE_M;
    if (required_bytes > device->properties.limits.maxComputeSharedMemorySize) {
        device_tile_m /= 2;
    }
        
    MatmulOperands ops = prepare_matmul_operands(self, other, bias_, has_bias, promoted_type, M, K, N, B, batch_shape);
    at::Tensor self_b = ops.self_b;
    at::Tensor other_b = ops.other_b;
    at::Tensor bias_b = ops.bias_b;

    at::Tensor out = at::empty({B, M, N}, self_b.options());
    if (beta.toFloat() == 0) out.zero_();

    if (M == 0 || N == 0 || B == 0) return out.reshape(out_shape);
    if (K == 0) {
        at::Tensor result = has_bias ? (bias_b * beta) : out.zero_();
        return result.reshape(out_shape);
    }

    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_matmul_simd(promoted_type);
    uint32_t vecSize = get_dtype_vec_size(promoted_type);

    uint32_t workgroupSizeX = 16; 
    uint32_t workgroupSizeY = 16; 
    uint32_t isBiasAligned = has_bias ? bias_b.is_contiguous() : 1;
    // vector loads/stores need whole vecSize rows, so K (A) and N (B/out) must both divide
    uint32_t isAligned = (K % vecSize == 0) && (N % vecSize == 0) &&
                         self_b.is_contiguous() && other_b.is_contiguous() && isBiasAligned;

    SpecializationBuilder spd{};
    spd.push(workgroupSizeX)
       .push(workgroupSizeY)
       .push(isAligned)
       .push(has_bias);
    uint32_t key = (has_bias << 21) | (isAligned << 20) | (workgroupSizeX << 10) | workgroupSizeY;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    uint32_t strides_a[4] = {static_cast<uint32_t>(self_b.stride(0)), static_cast<uint32_t>(self_b.stride(1)), static_cast<uint32_t>(self_b.stride(2)), 0};
    uint32_t strides_b[4] = {static_cast<uint32_t>(other_b.stride(0)), static_cast<uint32_t>(other_b.stride(1)), static_cast<uint32_t>(other_b.stride(2)), 0};
    uint32_t strides_c[4] = {static_cast<uint32_t>(out.stride(0)), static_cast<uint32_t>(out.stride(1)), static_cast<uint32_t>(out.stride(2)), 0};
    uint32_t strides_bias[4] = {0, 0, 0, 0};
    if (has_bias) {strides_bias[0] = static_cast<uint32_t>(bias_b.stride(0)); strides_bias[1] = static_cast<uint32_t>(bias_b.stride(1)); strides_bias[2] = static_cast<uint32_t>(bias_b.stride(2));}

    PushConstantBuilder pcs{};
    pcs.push_array(strides_a);
    pcs.push_array(strides_b);
    pcs.push_array(strides_c);
    pcs.push_array(strides_bias);
    pcs.push(M);
    pcs.push(N);
    pcs.push(K);
    pcs.push((uint32_t)0);
    pcs.push_scalar(alpha, promoted_type);
    pcs.push_scalar(beta, promoted_type);
    
    uint32_t groupX = (N + TILE_N - 1) / TILE_N;
    uint32_t groupY = (M + device_tile_m - 1) / device_tile_m;
    uint32_t groupZ = static_cast<uint32_t>(B);

    VulkanShader shader(shader_id, specialization, device);
    
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_b, other_b, out, has_bias ? bias_b : out}, 
        groupX, groupY, groupZ
    );

    out = out.reshape(out_shape);
    if (self_unsqueezed || other_unsqueezed) {
        if (self_unsqueezed && other_unsqueezed) out = out.squeeze(-1).squeeze(-1); 
        else if (self_unsqueezed) out = out.squeeze(-2); 
        else out = out.squeeze(-1); 
    }

    return out;
}

at::Tensor torchvulkan::mm_vulkan(
    const at::Tensor& self,
    const at::Tensor& other)
{
    return dispatch_matmul_shader(
        self, other, {}, 1, 0,
        [&]() { return at::mm(self.to(at::kCPU), other.to(at::kCPU)); }
    );
}

at::Tensor torchvulkan::bmm_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other)
{
    return dispatch_matmul_shader(
        self, other, {}, 1, 0,
        [&]() { return at::bmm(self.to(at::kCPU), other.to(at::kCPU)); }
    );
}

at::Tensor torchvulkan::matmul_vulkan(
    const at::Tensor& self, 
    const at::Tensor& other)
{
    return dispatch_matmul_shader(
        self, other, {}, 1, 0,
        [&]() { return at::matmul(self.to(at::kCPU), other.to(at::kCPU)); }
    );
}

at::Tensor torchvulkan::addmm_vulkan(
    const at::Tensor& input, 
    const at::Tensor& mat1, 
    const at::Tensor& mat2, 
    const at::Scalar& beta, 
    const at::Scalar& alpha) 
{
    return dispatch_matmul_shader(
        mat1, mat2, input, alpha, beta,
        [&]() { return at::addmm(input.to(at::kCPU), mat1.to(at::kCPU), mat2.to(at::kCPU), beta, alpha); }
    );
}

at::Tensor torchvulkan::baddbmm_vulkan(
    const at::Tensor& input, 
    const at::Tensor& mat1, 
    const at::Tensor& mat2, 
    const at::Scalar& beta, 
    const at::Scalar& alpha)
{
    return dispatch_matmul_shader(
        mat1, mat2, input, alpha, beta,
        [&]() { return at::baddbmm(input.to(at::kCPU), mat1.to(at::kCPU), mat2.to(at::kCPU), beta, alpha); }
    );
}