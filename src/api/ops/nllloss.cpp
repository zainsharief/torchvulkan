#include <torch/extension.h>
#include "api/ops/nllloss.h"
#include "api/ops/binary.h"
#include "api/ops/reduce.h"

namespace {

enum class NllLossMode : uint32_t {
    FORWARD_GATHER = 0,
    BACKWARD_SCATTER = 1,
    WEIGHT_GATHER = 2
};

void dispatch_nllloss_shader(
    const at::Tensor& in,
    const at::Tensor& target,
    at::Tensor& out,
    NllLossMode mode,
    uint32_t N,
    uint32_t C,
    int32_t ignore_index,
    uint32_t total_threads)
{
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    uint32_t workgroupSizeX = get_dtype_workgroup_size(in.scalar_type(), 1);
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_nllloss(in.scalar_type());

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(mode))
       .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 2) | static_cast<uint32_t>(mode);
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    PushConstantBuilder pcs{};
    pcs.push(N).push(C).push(ignore_index);

    uint32_t groupX = (total_threads + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs,
        pcs.size(),
        {in, target, out},
        groupX, 1, 1
    );
}

// gathers per-class weight[target[i]] into an [N] tensor (0 where target[i] == ignore_index);
// returns all-ones (except ignored entries) if weight is absent
at::Tensor gather_weight(
    const c10::optional<at::Tensor>& weight,
    const at::Tensor& target_c,
    const at::Tensor& options_like,
    uint32_t N,
    uint32_t C,
    int32_t ignore_index)
{
    at::Tensor weight_c = weight.has_value() && weight->defined()
        ? weight->to(options_like.scalar_type()).contiguous()
        : at::ones({(int64_t)C}, options_like.options());

    at::Tensor gathered = at::empty({(int64_t)N}, options_like.options());
    if (N > 0) dispatch_nllloss_shader(weight_c, target_c, gathered, NllLossMode::WEIGHT_GATHER, N, C, ignore_index, N);
    return gathered;
}

} // namespace

std::tuple<at::Tensor, at::Tensor> torchvulkan::nll_loss_forward_vulkan(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index)
{
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "torchvulkan [ERROR]: nll_loss only supports 1D (unbatched) or 2D input.");

    if (!is_dtype_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", self.scalar_type(), ". Falling back to CPU.");
        c10::optional<at::Tensor> cpu_weight = weight.has_value() && weight->defined() ? c10::optional<at::Tensor>(weight->cpu()) : c10::nullopt;
        auto [cpu_output, cpu_total_weight] = at::nll_loss_forward(self.cpu(), target.cpu(), cpu_weight, reduction, ignore_index);
        return {cpu_output.to(self.device()), cpu_total_weight.to(self.device())};
    }

    bool unbatched = self.dim() == 1;
    at::Tensor input = (unbatched ? self.unsqueeze(0) : self).contiguous();
    at::Tensor target_c = (unbatched ? target.unsqueeze(0) : target).to(at::kLong).contiguous();

    uint32_t N = static_cast<uint32_t>(input.size(0));
    uint32_t C = static_cast<uint32_t>(input.size(1));
    int32_t ignore_index32 = static_cast<int32_t>(ignore_index);

    at::Tensor losses = at::empty({(int64_t)N}, input.options());
    if (N > 0) dispatch_nllloss_shader(input, target_c, losses, NllLossMode::FORWARD_GATHER, N, C, ignore_index32, N);

    at::Tensor gathered_weight = gather_weight(weight, target_c, input, N, C, ignore_index32);
    at::Tensor weighted_losses = multiply_vulkan(losses, gathered_weight);
    at::Tensor total_weight = sum_vulkan(gathered_weight, c10::nullopt);

    at::Tensor output;
    if (reduction == 0) output = unbatched ? weighted_losses.squeeze(0) : weighted_losses;
    else if (reduction == 1) output = divide_vulkan(sum_vulkan(weighted_losses, c10::nullopt), total_weight);
    else output = sum_vulkan(weighted_losses, c10::nullopt);

    return {output, total_weight};
}

at::Tensor torchvulkan::nll_loss_backward_vulkan(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight,
    int64_t reduction,
    int64_t ignore_index,
    const at::Tensor& total_weight)
{
    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "torchvulkan [ERROR]: nll_loss only supports 1D (unbatched) or 2D input.");

    if (!is_dtype_supported(self.scalar_type())) {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", self.scalar_type(), ". Falling back to CPU.");
        c10::optional<at::Tensor> cpu_weight = weight.has_value() && weight->defined() ? c10::optional<at::Tensor>(weight->cpu()) : c10::nullopt;
        at::Tensor cpu_grad_input = at::nll_loss_backward(
            grad_output.cpu(), self.cpu(), target.cpu(), cpu_weight, reduction, ignore_index, total_weight.cpu());
        return cpu_grad_input.to(self.device());
    }

    bool unbatched = self.dim() == 1;
    at::Tensor input = (unbatched ? self.unsqueeze(0) : self).contiguous();
    at::Tensor target_c = (unbatched ? target.unsqueeze(0) : target).to(at::kLong).contiguous();

    uint32_t N = static_cast<uint32_t>(input.size(0));
    uint32_t C = static_cast<uint32_t>(input.size(1));
    int32_t ignore_index32 = static_cast<int32_t>(ignore_index);

    at::Tensor gathered_weight = gather_weight(weight, target_c, input, N, C, ignore_index32);

    // pre-scale the (possibly 0-dim/broadcast) grad_output + reduction + per-class weight into one [N] tensor,
    // so the shader only has to scatter it back into grad_input.
    at::Tensor grad_out_c = grad_output.to(input.scalar_type());
    at::Tensor per_sample_grad = reduction == 1
        ? divide_vulkan(grad_out_c, total_weight.to(input.scalar_type()))
        : grad_out_c;
    per_sample_grad = multiply_vulkan(per_sample_grad.expand({(int64_t)N}), gathered_weight);
    per_sample_grad = per_sample_grad.contiguous();

    at::Tensor grad_input = at::empty({(int64_t)N, (int64_t)C}, input.options());
    if (N > 0) dispatch_nllloss_shader(per_sample_grad, target_c, grad_input, NllLossMode::BACKWARD_SCATTER, N, C, ignore_index32, N * C);

    return unbatched ? grad_input.squeeze(0) : grad_input;
}
