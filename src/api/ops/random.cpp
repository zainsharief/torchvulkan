#include <torch/extension.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <cmath>
#include <limits>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "vulkan/generator.h"

namespace {

enum class RandomOp {
    UNIFORM = 0,
    NORMAL = 1,
    EXPONENTIAL = 2,
    LOG_NORMAL = 3,
    CAUCHY = 4,
    GEOMETRIC = 5,
    BERNOULLI_SCALAR = 6,
    BERNOULLI_TENSOR = 7,
    RANDOM = 8
};

struct RandomParams {
    double a = 0;       // from, mean, lambda, median or p
    double b = 0;       // to, std, sigma, or the exponential's lower bound
    int64_t from = 0;
    uint64_t range = 0;
    at::Tensor probs;   // BERNOULLI_TENSOR, already expanded to the output's shape
};

// random_() with no bounds draws from [0, 2^digits] for floats and [0, max] for integers
uint64_t default_random_range(at::ScalarType dtype)
{
    if (dtype == at::kBool) return 2;
    if (dtype == at::kUInt64) return 0;

    uint64_t range = uint64_t(1) << get_dtype_digits(dtype);
    return at::isFloatingType(dtype) ? range + 1 : range;
}

// update_from and update_to move integer bounds inwards so rounding to a floating dtype stays inside [from, to)
int64_t update_from(int64_t from, at::ScalarType dtype)
{
    int64_t from_plus_1 = round_to_dtype(from + 1, dtype);
    if (from_plus_1 >= from) return from;

    int64_t magnitude = std::abs(from + 1);
    int n = 0;
    while (magnitude >>= 1) n++;
    return from_plus_1 + (int64_t(1) << (n - get_dtype_digits(dtype) + 1));
}

int64_t update_to(int64_t to, at::ScalarType dtype)
{
    int64_t to_minus_1 = round_to_dtype(to - 1, dtype);
    if (to_minus_1 < to) return to;

    int64_t magnitude = std::abs(to - 1);
    int n = 0;
    while (magnitude >>= 1) n++;
    return to_minus_1 - (int64_t(1) << (n - get_dtype_digits(dtype) + 1));
}

void check_random_bounds(int64_t from, int64_t to_inc, const at::Tensor& self)
{
    auto [lowest, highest] = get_dtype_limits(self.scalar_type());
    if (self.scalar_type() == at::kUInt64) highest = static_cast<double>(std::numeric_limits<int64_t>::max());

    TORCH_CHECK(from >= lowest && from <= highest, "from is out of bounds for ", self.dtype());
    TORCH_CHECK(to_inc >= lowest && to_inc <= highest, "to - 1 is out of bounds for ", self.dtype());
}

void check_normal_std(const at::Tensor& std)
{
    TORCH_CHECK(!std.is_complex(), "normal expects standard deviation to be non-complex");
    TORCH_CHECK(std.numel() == 0 || std.min().ge(0).item<bool>(), "normal expects all elements of std >= 0.0");
}

void dispatch_random_shader(
    const at::Tensor& self,
    RandomOp op,
    const RandomParams& params,
    const std::optional<at::Generator>& generator)
{
    if (self.numel() == 0) return;

    bool supported = is_dtype_supported(self, self.scalar_type());
    bool too_many_dims = self.dim() > MAX_DIMS || !fits_32bit_indexing({self});
    bool needs_contiguous = op == RandomOp::BERNOULLI_TENSOR && !self.is_contiguous();
    if (!supported || too_many_dims || needs_contiguous)
    {
        at::ScalarType dtype = self.scalar_type();
        if (!supported) {
            dtype = random_proxy_dtype(dtype);
            TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support ", self.scalar_type(), ". Generating random values in ", dtype, " instead.");
        }
        at::Tensor temp = at::empty(self.sizes(), self.options().dtype(dtype));
        dispatch_random_shader(temp, op, params, generator);
        self.copy_(temp);

        // rounding to self's dtype can land on `to`, which [from, to) excludes
        if (op == RandomOp::UNIFORM && dtype != self.scalar_type()) self.masked_fill_(self == params.b, params.a);
        return;
    }

    DeviceContext* device = get_device_context(self);
    at::ScalarType real = random_real_dtype(self.scalar_type());

    at::TensorIterator iter = at::TensorIterator::borrowing_nullary_op(self);
    uint64_t numel = iter.numel();
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    uint32_t contiguous = iter.is_contiguous() ? 1 : 0;
    uint32_t vecSize = get_dtype_vec_size(self.scalar_type());
    uint32_t workgroupSizeX = get_dtype_workgroup_size(self.scalar_type(), vecSize, device);

    SpecializationBuilder spd{};
    spd.push(static_cast<uint32_t>(op))
        .push(contiguous)
        .push(out_dims)
        .push(workgroupSizeX);
    SpecializationArgs specialization = spd.build();

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {0});
    Metadata metadata = metadataBuilder.build();

    uint64_t param0 = random_real_bits(params.a, real);
    uint64_t param1 = random_real_bits(params.b, real);
    if (op == RandomOp::RANDOM) {
        param0 = static_cast<uint64_t>(params.from);
        param1 = params.range;
    }

    at::Tensor probs;
    uint64_t probs_address = 0;
    if (op == RandomOp::BERNOULLI_TENSOR) {
        probs = params.probs.to(self.device(), real).contiguous();
        probs_address = get_tensor_address(probs);
    }

    auto [seed, offset] = VulkanGeneratorImpl::reserve(generator, self.device().index());

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(self))
        .push_metadata(contiguous ? nullptr : &metadata)
        .push(numel)
        .push(seed)
        .push(offset)
        .push(param0)
        .push(param1)
        .push(probs_address);
    PushConstants pushConstants = pcs.build();

    uint64_t threads = (numel + vecSize - 1) / vecSize;
    uint32_t groupX = static_cast<uint32_t>((threads + workgroupSizeX - 1) / workgroupSizeX);

    std::vector<at::Tensor> reads;
    if (probs.defined()) reads.push_back(probs);

    device->shader_manager->dispatchShader(
        ShaderKey{torchvulkan::Kernel::RANDOM, self.scalar_type()},
        device->default_subgroup_size,
        specialization,
        pushConstants,
        /* read = */ reads,
        /* write = */ {self},
        groupX, 1, 1
    );
}


at::Tensor& uniform_vulkan(at::Tensor& self, double from, double to, std::optional<at::Generator> generator)
{
    if (self.is_complex()) {
        if (self.numel() == 0) return self;
        at::Tensor real = at::view_as_real(self);
        uniform_vulkan(real, from, to, generator);
        return self;
    }

    at::ScalarType dtype = self.scalar_type();
    TORCH_CHECK_NOT_IMPLEMENTED(at::isFloatingType(dtype), "\"check_uniform_bounds\" not implemented for '", dtype, "'");
    auto [lowest, highest] = get_dtype_limits(dtype);
    TORCH_CHECK(from >= lowest && from <= highest, "from is out of bounds for ", self.dtype());
    TORCH_CHECK(to >= lowest && to <= highest, "to is out of bounds for ", self.dtype());
    TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range, but found from=", from, " > to=", to);
    TORCH_CHECK(to - from <= highest, "uniform_ expects to-from <= std::numeric_limits<", dtype, ">::max(), but found to=", to,
        " and from=", from, " which result in to-from to exceed the limit"
    );

    dispatch_random_shader(self, RandomOp::UNIFORM, {from, to}, generator);
    return self;
}

at::Tensor& normal_vulkan(at::Tensor& self, double mean, double std, std::optional<at::Generator> generator)
{
    TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);
    if (self.numel() == 0) return self;

    // the real and imaginary parts each get half the variance
    if (self.is_complex()) {
        at::Tensor real = at::view_as_real(self);
        dispatch_random_shader(real, RandomOp::NORMAL, {mean, std / std::sqrt(2)}, generator);
        return self;
    }

    check_floating(self, "normal_");
    dispatch_random_shader(self, RandomOp::NORMAL, {mean, std}, generator);
    return self;
}

at::Tensor& normal_tensor_float_out_vulkan(const at::Tensor& mean, double std, std::optional<at::Generator> generator, at::Tensor& out)
{
    TORCH_CHECK(std >= 0.0, "normal expects std >= 0.0, but found std ", std);
    at::native::resize_output(out, at::infer_size(mean.sizes(), out.sizes()));
    normal_vulkan(out, 0, std, generator);
    return out.add_(mean);
}

at::Tensor normal_tensor_float_vulkan(const at::Tensor& mean, double std, std::optional<at::Generator> generator)
{
    at::Tensor out = at::empty_like(mean, at::MemoryFormat::Contiguous);
    return normal_tensor_float_out_vulkan(mean, std, generator, out);
}

at::Tensor& normal_float_tensor_out_vulkan(double mean, const at::Tensor& std, std::optional<at::Generator> generator, at::Tensor& out)
{
    check_normal_std(std);
    at::Tensor mean_tensor = at::full({}, mean, out.options());
    at::native::resize_output(out, at::infer_size(mean_tensor.sizes(), std.sizes()));
    normal_vulkan(out, 0, 1, generator);
    return out.mul_(std).add_(mean_tensor);
}

at::Tensor normal_float_tensor_vulkan(double mean, const at::Tensor& std, std::optional<at::Generator> generator)
{
    at::Tensor out = at::empty_like(std, at::MemoryFormat::Contiguous);
    return normal_float_tensor_out_vulkan(mean, std, generator, out);
}

at::Tensor& normal_tensor_tensor_out_vulkan(const at::Tensor& mean, const at::Tensor& std, std::optional<at::Generator> generator, at::Tensor& out)
{
    check_normal_std(std);
    at::native::resize_output(out, at::infer_size(mean.sizes(), std.sizes()));
    normal_vulkan(out, 0, 1, generator);
    return out.mul_(std).add_(mean);
}

at::Tensor normal_tensor_tensor_vulkan(const at::Tensor& mean, const at::Tensor& std, std::optional<at::Generator> generator)
{
    at::Tensor out = at::empty(at::infer_size(mean.sizes(), std.sizes()), mean.options(), at::MemoryFormat::Contiguous);
    return normal_tensor_tensor_out_vulkan(mean, std, generator, out);
}

at::Tensor& exponential_vulkan(at::Tensor& self, double lambda, std::optional<at::Generator> generator)
{
    TORCH_CHECK(lambda > 0.0, "exponential_ expects lambda > 0.0, but found lambda=", lambda);
    if (self.numel() == 0) return self;

    check_floating(self, "exponential_");
    dispatch_random_shader(self, RandomOp::EXPONENTIAL, {lambda, get_dtype_epsilon(self.scalar_type()) / 2}, generator);
    return self;
}

at::Tensor& log_normal_vulkan(at::Tensor& self, double mean, double std, std::optional<at::Generator> generator)
{
    TORCH_CHECK(std > 0.0, "log_normal_ expects std > 0.0, but found std=", std);
    if (self.numel() == 0) return self;

    check_floating(self, "log_normal_");
    dispatch_random_shader(self, RandomOp::LOG_NORMAL, {mean, std}, generator);
    return self;
}

at::Tensor& cauchy_vulkan(at::Tensor& self, double median, double sigma, std::optional<at::Generator> generator)
{
    TORCH_CHECK(sigma > 0.0, "cauchy_ expects sigma > 0.0, but found sigma=", sigma);
    TORCH_CHECK(at::isFloatingType(self.scalar_type()),
        "Cauchy distribution is a continuous probability distribution. dtype must be a floating point but you specified ", self.dtype());

    dispatch_random_shader(self, RandomOp::CAUCHY, {median, sigma}, generator);
    return self;
}

at::Tensor& geometric_vulkan(at::Tensor& self, double p, std::optional<at::Generator> generator)
{
    TORCH_CHECK(0 < p && p < 1, "geometric_ expects p to be in (0, 1), but got p=", p);

    dispatch_random_shader(self, RandomOp::GEOMETRIC, {p, 0}, generator);
    return self;
}

at::Tensor& bernoulli_scalar_vulkan(at::Tensor& self, double p, std::optional<at::Generator> generator)
{
    TORCH_CHECK(0 <= p && p <= 1, "bernoulli_ expects p to be in [0, 1], but got p=", p);
    if (self.numel() == 0) return self;

    at::assert_no_internal_overlap(self);
    dispatch_random_shader(self, RandomOp::BERNOULLI_SCALAR, {p, 0}, generator);
    return self;
}

at::Tensor& bernoulli_tensor_vulkan(at::Tensor& self, const at::Tensor& p, std::optional<at::Generator> generator)
{
    if (self.numel() == 0) return self;

    at::assert_no_internal_overlap(self);
    RandomParams params;
    params.probs = p.expand(self.sizes());
    dispatch_random_shader(self, RandomOp::BERNOULLI_TENSOR, params, generator);
    return self;
}

at::Tensor& bernoulli_out_vulkan(const at::Tensor& self, std::optional<at::Generator> generator, at::Tensor& out)
{
    out.resize_(self.sizes());
    return bernoulli_tensor_vulkan(out, self, generator);
}

at::Tensor& random_vulkan(at::Tensor& self, std::optional<at::Generator> generator)
{
    RandomParams params;
    params.range = default_random_range(self.scalar_type());
    dispatch_random_shader(self, RandomOp::RANDOM, params, generator);
    return self;
}

at::Tensor& random_from_vulkan(at::Tensor& self, int64_t from, std::optional<int64_t> to, std::optional<at::Generator> generator)
{
    at::ScalarType dtype = self.scalar_type();
    bool floating = at::isFloatingType(dtype);
    RandomParams params;

    if (to.has_value()) {
        int64_t end = *to;
        TORCH_CHECK(from < end, "random_ expects 'from' to be less than 'to', but got from=", from, " >= to=", end);
        if (floating) {
            from = update_from(from, dtype);
            end = update_to(end, dtype);
            TORCH_CHECK(from < end, "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=", from, " >= to=", end);
        }
        check_random_bounds(from, end - 1, self);
        params.range = static_cast<uint64_t>(end) - static_cast<uint64_t>(from);
    } else if (from != std::numeric_limits<int64_t>::lowest()) {
        // with no `to`, the same maximum as random_()
        int64_t to_inc = static_cast<int64_t>(default_random_range(dtype) - 1);
        if (floating) {
            from = update_from(from, dtype);
            TORCH_CHECK(from < to_inc, "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=", from, " > to_inc=", to_inc);
        }
        check_random_bounds(from, to_inc, self);
        params.range = static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
    }

    params.from = from;
    dispatch_random_shader(self, RandomOp::RANDOM, params, generator);
    return self;
}

at::Tensor& random_to_vulkan(at::Tensor& self, int64_t to, std::optional<at::Generator> generator)
{
    return random_from_vulkan(self, 0, to, generator);
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("uniform_", &uniform_vulkan);
    m.impl("normal_", &normal_vulkan);
    m.impl("normal.Tensor_float", &normal_tensor_float_vulkan);
    m.impl("normal.Tensor_float_out", &normal_tensor_float_out_vulkan);
    m.impl("normal.float_Tensor", &normal_float_tensor_vulkan);
    m.impl("normal.float_Tensor_out", &normal_float_tensor_out_vulkan);
    m.impl("normal.Tensor_Tensor", &normal_tensor_tensor_vulkan);
    m.impl("normal.Tensor_Tensor_out", &normal_tensor_tensor_out_vulkan);
    m.impl("exponential_", &exponential_vulkan);
    m.impl("log_normal_", &log_normal_vulkan);
    m.impl("cauchy_", &cauchy_vulkan);
    m.impl("geometric_", &geometric_vulkan);
    m.impl("bernoulli_.float", &bernoulli_scalar_vulkan);
    m.impl("bernoulli_.Tensor", &bernoulli_tensor_vulkan);
    m.impl("bernoulli.out", &bernoulli_out_vulkan);
    m.impl("random_", &random_vulkan);
    m.impl("random_.from", &random_from_vulkan);
    m.impl("random_.to", &random_to_vulkan);
}
