#include <torch/extension.h>
#include "api/ops/matmul.h"

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

    CoopMatParams* params = device->cache.getCoopMatParams(promoted_type, available_block_sizes);
    if (!params->is_valid) return dispatch_matmul_simd_shader(self_, other_, bias_, alpha, beta, cpu_fallback);

    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_matmul_coop(promoted_type, params->block_size);

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
    std::vector<int64_t> out_shape_vec = at::infer_size(self_batch, other_batch);
    
    int64_t B = 1;
    for (int64_t s : out_shape_vec) {
        B *= s;
    }

    out_shape_vec.push_back(M);
    out_shape_vec.push_back(N);
    at::IntArrayRef out_shape(out_shape_vec);

    at::Tensor self_b = self.to(promoted_type).expand({B, M, K});
    at::Tensor other_b = other.to(self_b.options()).expand({B, K, N});
    at::Tensor out = at::empty({B, M, N}, self_b.options());

    uint32_t strides_a[4] = {static_cast<uint32_t>(self_b.stride(0)), static_cast<uint32_t>(self_b.stride(1)), static_cast<uint32_t>(self_b.stride(2)), 0};
    uint32_t strides_b[4] = {static_cast<uint32_t>(other_b.stride(0)), static_cast<uint32_t>(other_b.stride(1)), static_cast<uint32_t>(other_b.stride(2)), 0};
    uint32_t strides_out[4] = {static_cast<uint32_t>(out.stride(0)), static_cast<uint32_t>(out.stride(1)), static_cast<uint32_t>(out.stride(2)), 0};
    
    PushConstantBuilder pcs{};
    pcs.push_array(strides_a)
       .push_array(strides_b)
       .push_array(strides_out)
       .push(M)
       .push(N)
       .push(K);

    SpecializationBuilder spd{};
    spd.push(params->workgroup_size)
       .push(params->subgroup_size)
       .push(params->warps_m)
       .push(params->warps_n)
       .push(params->warp_frags_m)
       .push(params->warp_frags_n)
       .push(params->bk);
    uint32_t key = (params->bk << 35) | (params->warp_frags_n << 31) | (params->warp_frags_m << 27) | (params->warps_n << 23) | (params->warps_m << 19) | (params->subgroup_size << 12) | (params->workgroup_size);
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};
    VulkanShader shader(shader_id, specialization, device);

    const uint32_t tile_m = params->block_size * params->warps_m * params->warp_frags_m;
    const uint32_t tile_n = params->block_size_acc * params->warps_n * params->warp_frags_n;
    uint32_t groupX = (N + tile_n - 1) / tile_n;
    uint32_t groupY = (M + tile_m - 1) / tile_m;
    uint32_t groupZ = B;

    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_b, other_b, out}, 
        groupX, groupY, groupZ
    );

    // Reshape the flat {B, M, N} output back to its true N-D shape
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
    at::Tensor bias = bias_;
    uint32_t bias_defined = (bias_.defined() && beta.toDouble() != 0.0) ? 1 : 0;
    
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
    std::vector<int64_t> out_shape_vec = at::infer_size(self_batch, other_batch);
    
    int64_t B = 1;
    for (int64_t s : out_shape_vec) {
        B *= s;
    }

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
        
    at::Tensor self_b = self.to(promoted_type).expand({B, M, K});
    at::Tensor other_b = other.to(self_b.options()).expand({B, K, N});
    at::Tensor bias_b = bias_defined ? bias.to(self_b.options()).expand({B, M, N}) : bias;
    at::Tensor out = at::empty({B, M, N}, self_b.options());
    if (beta.toFloat() == 0) out.zero_();

    if (M == 0 || N == 0 || B == 0) return out.reshape(out_shape);
    if (K == 0) return out.zero_().reshape(out_shape);

    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_matmul_simd(promoted_type);
    uint32_t vecSize = get_dtype_vec_size(promoted_type);

    uint32_t workgroupSizeX = 16; 
    uint32_t workgroupSizeY = 16; 
    uint32_t isBiasAligned = bias_defined ? bias_b.is_contiguous() : 1;
    uint32_t isAligned = (K % TILE_K == 0) && self_b.is_contiguous() && other_b.is_contiguous() && isBiasAligned;

    SpecializationBuilder spd{};
    spd.push(workgroupSizeX)
       .push(workgroupSizeY)
       .push(isAligned)
       .push(bias_defined);
    uint32_t key = (bias_defined << 12) | (isAligned << 8) | (workgroupSizeX << 4) | workgroupSizeY;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    uint32_t strides_a[4] = {static_cast<uint32_t>(self_b.stride(0)), static_cast<uint32_t>(self_b.stride(1)), static_cast<uint32_t>(self_b.stride(2)), 0};
    uint32_t strides_b[4] = {static_cast<uint32_t>(other_b.stride(0)), static_cast<uint32_t>(other_b.stride(1)), static_cast<uint32_t>(other_b.stride(2)), 0};
    uint32_t strides_c[4] = {static_cast<uint32_t>(out.stride(0)), static_cast<uint32_t>(out.stride(1)), static_cast<uint32_t>(out.stride(2)), 0};
    uint32_t strides_bias[4] = {0};
    if (bias_defined) {
        strides_bias[0] = static_cast<uint32_t>(bias_b.stride(0));
        strides_bias[1] = static_cast<uint32_t>(bias_b.stride(1));
        strides_bias[2] = static_cast<uint32_t>(bias_b.stride(2));
    }

    PushConstantBuilder pcs{};
    pcs.push(M);
    pcs.push(N);
    pcs.push(K);
    pcs.push((uint32_t)0);
    pcs.push_array(strides_a);
    pcs.push_array(strides_b);
    pcs.push_array(strides_c);
    pcs.push_array(strides_bias);
    pcs.push_scalar(alpha, promoted_type);
    pcs.push_scalar(beta, promoted_type);
    
    uint32_t groupX = (N + TILE_N - 1) / TILE_N;
    uint32_t groupY = (M + device_tile_m - 1) / device_tile_m;
    uint32_t groupZ = static_cast<uint32_t>(B);

    VulkanShader shader(shader_id, specialization, device);
    
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {self_b, other_b, out, bias_defined ? bias_b : out}, 
        groupX, groupY, groupZ
    );

    // Reshape the flat {B, M, N} output back to its true N-D shape
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