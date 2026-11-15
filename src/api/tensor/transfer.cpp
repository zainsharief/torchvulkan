#include <torch/extension.h>
#include <c10/util/bit_cast.h>
#include <limits>
#include <vector>
#include <c10/core/DeviceGuard.h>
#include <c10/core/MemoryFormat.h>
#include "vulkan/memory.h"
#include "vulkan/vulkan_context.h"
#include "vulkan/allocator.h"
#include "api/ops/helpers.h"
#include "api/ops/internal.h"
#include "api/fallback.h"

namespace {

void dispatch_copy_shader(const at::Tensor& src, const at::Tensor& dst)
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(dst)
        .add_const_input(src)
        .build();

    uint64_t numel = iter.numel();
    if (numel == 0) return;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    TORCH_CHECK(out_dims <= MAX_DIMS, "torchvulkan [ERROR]: A strided copy has ", out_dims, " dimensions after coalescing; the copy shader supports ", MAX_DIMS, ".");
    TORCH_CHECK(fits_32bit_indexing(iter), "torchvulkan [ERROR]: A strided copy of ", numel, " elements is too large for the copy shader's 32-bit indexing.");

    DeviceContext* device = get_device_context(dst);
    ShaderKey shader_key{torchvulkan::Kernel::COPY, get_dtype_width_variant(dst.scalar_type())};
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), 1, device);

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(workgroupSizeX);

    MetadataBuilder metadataBuilder = get_iter_metadata(iter, {1, 0});
    Metadata metadata = metadataBuilder.build();

    PushConstantBuilder pcs{};
    pcs.push(get_tensor_address(src))
        .push(get_output_address(dst))
        .push_metadata(&metadata)
        .push(numel);

    device->shader_manager->dispatchShader(
        shader_key,
        device->default_subgroup_size,
        spd.build(),
        pcs.build(),
        /* read = */ {src},
        /* write = */ {dst},
        get_group_count(numel, workgroupSizeX), 1, 1
    );
}

void copy_host_to_vulkan(at::Tensor src, const at::Tensor& dst)
{
    if (src.scalar_type() != dst.scalar_type()) src = src.to(dst.scalar_type());
    if (!src.is_contiguous()) src = src.contiguous();

    materialize_cow(dst);
    void* dest_ptr = extract_buffer(dst.storage());
    uint64_t dest_offset = extract_buffer_offset(dst.storage()) + dst.storage_offset() * dst.itemsize();
    if (dst.is_contiguous()) {
        globalVulkanAllocator.copy_host_to_device(dest_ptr, dest_offset, src.data_ptr(), dst.nbytes());
        return;
    }

    at::Tensor staging = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
    void* staging_ptr = extract_buffer(staging.storage());
    uint64_t staging_offset = extract_buffer_offset(staging.storage()) + staging.storage_offset() * dst.itemsize();
    globalVulkanAllocator.copy_host_to_device(staging_ptr, staging_offset, src.data_ptr(), dst.nbytes());
    dispatch_copy_shader(staging, dst);
}

void copy_vulkan_to_host(at::Tensor src, const at::Tensor& dst)
{
    bool expanded = false;
    at::Tensor distinct = src;
    for (int64_t d = 0; d < src.dim(); d++)
    {
        if (src.stride(d) == 0 && src.size(d) > 1)
        {
            distinct = distinct.narrow(d, 0, 1);
            expanded = true;
        }
    }

    if (expanded)
    {
        at::Tensor host = at::empty(distinct.sizes(), dst.options().memory_format(at::MemoryFormat::Contiguous));
        copy_vulkan_to_host(distinct, host);
        dst.copy_(host.expand(src.sizes()));
        return;
    }

    if (src.scalar_type() != dst.scalar_type())
    {
        if (!is_dtype_supported(src, src.scalar_type()) || !is_dtype_supported(src, dst.scalar_type())) {
            // the bytes copy over unchanged; the CPU converts them
            at::Tensor host = at::empty(src.sizes(), src.options().device(at::kCPU).memory_format(at::MemoryFormat::Contiguous));
            copy_vulkan_to_host(src, host);
            dst.copy_(host);
            return;
        }

        at::Tensor temp = at::empty_like(src, src.options().dtype(dst.scalar_type()).memory_format(at::MemoryFormat::Contiguous));
        torchvulkan::dispatch_cast_shader(src, temp);
        src = std::move(temp);
    }

    if (!src.is_contiguous()) src = src.contiguous();

    void* src_ptr = extract_buffer(src.storage());
    uint64_t src_offset = extract_buffer_offset(src.storage()) + src.storage_offset() * src.itemsize();

    if (dst.is_contiguous()) {
        globalVulkanAllocator.copy_device_to_host(dst.data_ptr(), src_ptr, src_offset, dst.nbytes());
        return;
    }

    at::Tensor staging = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
    globalVulkanAllocator.copy_device_to_host(staging.data_ptr(), src_ptr, src_offset, staging.nbytes());
    dst.copy_(staging);
}

// float8 has no shader type. Both directions come from c10's own conversions, so they match the CPU bit for bit:
// decoding gathers from a table of all 256 codes, and encoding searches, for each code, the smallest float32 that
// c10 rounds to it (c10's conversion is monotone in the value), then mirrors the sign through c10 as well.
struct Float8Tables {
    std::vector<float> decode;      // the value of each of the 256 codes
    std::vector<int32_t> thresholds;  // the bits of the smallest |x| that rounds to each of codes
    std::vector<int32_t> codes;
    std::vector<int32_t> negated;   // c10's code for -x, for each of codes
    int32_t nan_code;
    int32_t negative_nan_code;
};

Float8Tables float8_tables(at::ScalarType dtype)
{
    uint8_t (*encode)(float) = nullptr;
    float (*decode)(uint8_t) = nullptr;
    switch (dtype) {
        case at::kFloat8_e4m3fn:
            encode = [](float v) { return c10::Float8_e4m3fn(v).x; };
            decode = [](uint8_t b) { return static_cast<float>(c10::Float8_e4m3fn(b, c10::Float8_e4m3fn::from_bits())); };
            break;
        case at::kFloat8_e5m2:
            encode = [](float v) { return c10::Float8_e5m2(v).x; };
            decode = [](uint8_t b) { return static_cast<float>(c10::Float8_e5m2(b, c10::Float8_e5m2::from_bits())); };
            break;
        case at::kFloat8_e4m3fnuz:
            encode = [](float v) { return c10::Float8_e4m3fnuz(v).x; };
            decode = [](uint8_t b) { return static_cast<float>(c10::Float8_e4m3fnuz(b, c10::Float8_e4m3fnuz::from_bits())); };
            break;
        case at::kFloat8_e5m2fnuz:
            encode = [](float v) { return c10::Float8_e5m2fnuz(v).x; };
            decode = [](uint8_t b) { return static_cast<float>(c10::Float8_e5m2fnuz(b, c10::Float8_e5m2fnuz::from_bits())); };
            break;
        case at::kFloat8_e8m0fnu:
            encode = [](float v) { return c10::Float8_e8m0fnu(v).x; };
            decode = [](uint8_t b) { return static_cast<float>(c10::Float8_e8m0fnu(b, c10::Float8_e8m0fnu::from_bits())); };
            break;
        default: TORCH_CHECK(false, "torchvulkan [ERROR]: ", dtype, " is not a float8 dtype.");
    }

    Float8Tables t;
    for (int b = 0; b < 256; b++) t.decode.push_back(decode(static_cast<uint8_t>(b)));
    auto code_at = [encode](uint32_t bits) { return static_cast<int32_t>(encode(c10::bit_cast<float>(bits))); };
    constexpr uint32_t POSITIVE_INF = 0x7F800000u;
    for (int32_t k = code_at(0); k <= code_at(POSITIVE_INF); k++) {
        uint32_t lo = 0, hi = POSITIVE_INF;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            if (code_at(mid) >= k) hi = mid; else lo = mid + 1;
        }
        if (code_at(lo) != k) continue;  // no float32 rounds to this code
        t.thresholds.push_back(static_cast<int32_t>(lo));
        t.codes.push_back(k);
        t.negated.push_back(encode(-c10::bit_cast<float>(lo)));
    }
    t.nan_code = encode(std::numeric_limits<float>::quiet_NaN());
    t.negative_nan_code = encode(-std::numeric_limits<float>::quiet_NaN());
    return t;
}

at::Tensor table_on(const at::Tensor& like, const std::vector<float>& values)
{
    return at::from_blob(const_cast<float*>(values.data()), {static_cast<int64_t>(values.size())}, at::kFloat).to(like.device());
}

at::Tensor table_on(const at::Tensor& like, const std::vector<int32_t>& values)
{
    return at::from_blob(const_cast<int32_t*>(values.data()), {static_cast<int64_t>(values.size())}, at::kInt).to(like.device());
}

at::Tensor float8_to_float(const at::Tensor& src)
{
    Float8Tables t = float8_tables(src.scalar_type());
    at::Tensor codes = src.contiguous().view(at::kByte).reshape({-1}).to(at::kLong);
    return table_on(src, t.decode).index_select(0, codes).view(src.sizes());
}

at::Tensor float_to_float8_codes(const at::Tensor& x, at::ScalarType dtype)
{
    Float8Tables t = float8_tables(dtype);
    // searched on the bits, which order non-negative floats and which a GPU that flushes subnormals can't flush
    at::Tensor bits = x.to(at::kFloat).contiguous().reshape({-1}).view(at::kInt);
    at::Tensor magnitude = at::bitwise_and(bits, 0x7FFFFFFF);
    at::Tensor position = at::clamp_min(at::sub(at::searchsorted(table_on(x, t.thresholds), magnitude, /*out_int32=*/false, /*right=*/true), 1), 0);
    at::Tensor negative = at::lt(bits, 0);
    at::Tensor code = at::where(negative, table_on(x, t.negated).index_select(0, position), table_on(x, t.codes).index_select(0, position));
    at::Tensor nan_code = at::where(negative, at::full_like(code, t.negative_nan_code), at::full_like(code, t.nan_code));
    code = at::where(at::gt(magnitude, 0x7F800000), nan_code, code);
    return code.to(at::kByte).view(x.sizes());
}

// a copy to, from or between float8 dtypes, through float32 (which holds every float8 value exactly)
void copy_float8_vulkan_to_vulkan(const at::Tensor& src, const at::Tensor& dst)
{
    at::Tensor value = c10::isFloat8Type(src.scalar_type()) ? float8_to_float(src) : src;
    if (c10::isFloat8Type(dst.scalar_type())) {
        dst.view(at::kByte).copy_(float_to_float8_codes(value, dst.scalar_type()));
    } else {
        dst.copy_(value);
    }
}

void copy_complex_vulkan_to_vulkan(const at::Tensor& src, const at::Tensor& dst)
{
    if (src.is_complex() && dst.is_complex()) {
        at::view_as_real(dst).copy_(at::view_as_real(src));
    } else if (dst.is_complex()) {
        at::Tensor pairs = at::view_as_real(dst);
        pairs.select(-1, 0).copy_(src);
        pairs.select(-1, 1).zero_();
    } else {
        dst.copy_(at::view_as_real(src).select(-1, 0));
    }
}

void copy_vulkan_to_vulkan(const at::Tensor& src, const at::Tensor& dst)
{
    if (src.scalar_type() != dst.scalar_type() && (src.is_complex() || dst.is_complex()))
    {
        copy_complex_vulkan_to_vulkan(src, dst);
        return;
    }

    if (src.scalar_type() != dst.scalar_type() && (c10::isFloat8Type(src.scalar_type()) || c10::isFloat8Type(dst.scalar_type())))
    {
        copy_float8_vulkan_to_vulkan(src, dst);
        return;
    }

    if (src.scalar_type() != dst.scalar_type())
    {
        if (!is_dtype_supported(src, src.scalar_type()) || !is_dtype_supported(src, dst.scalar_type())) {
            TORCHVULKAN_FALLBACK("A copy from ", src.scalar_type(), " to ", dst.scalar_type(), " can't run on the Vulkan device. Falling back to CPU.");
            dst.copy_(src.cpu());
            return;
        }

        torchvulkan::dispatch_cast_shader(src, dst);
        return;
    }

    if (src.scalar_type() == at::kBool) 
    {
        torchvulkan::dispatch_cast_shader(src, dst);
        return;
    }

    if (src.is_contiguous() && dst.is_contiguous())
    {
        void* src_ptr = extract_buffer(src.storage());
        uint64_t src_offset = extract_buffer_offset(src.storage()) + src.storage_offset() * src.itemsize();
        materialize_cow(dst);
        void* dest_ptr = extract_buffer(dst.storage());
        uint64_t dest_offset = extract_buffer_offset(dst.storage()) + dst.storage_offset() * dst.itemsize();
        globalVulkanAllocator.copy_device_to_device(dest_ptr, dest_offset, src_ptr, src_offset, dst.nbytes());
        return;
    }

    dispatch_copy_shader(src, dst);
}

} // namespace

at::Tensor torchvulkan::copy_vulkan(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool /* non_blocking */)
{
    if (dst.numel() == 0) return dst;

    // as at::native::copy_, copying a tensor onto its own memory is a no-op (and vkCmdCopyBuffer forbids overlapping regions)
    if (self.is_alias_of(dst) && self.storage_offset() == dst.storage_offset() && self.sizes().equals(dst.sizes()) &&
        self.strides().equals(dst.strides()) && self.scalar_type() == dst.scalar_type() &&
        self.is_conj() == dst.is_conj() && self.is_neg() == dst.is_neg()) {
        return dst;
    }

    if (dst.is_conj() || dst.is_neg()) 
    {
        at::Tensor stored = dst, value = self;
        if (stored.is_conj()) {
            stored = at::_conj(stored);
            value = value.conj();
        }
        if (stored.is_neg()) {
            stored = at::_neg_view(stored);
            value = at::_neg_view(value);
        }
        torchvulkan::copy_vulkan(value, stored, false);
        return dst;
    }

    // a lazily conjugated or negated source copies its stored values, then the copy applies the pending op
    if (self.is_conj() || self.is_neg()) {
        if (self.device().is_cpu()) return torchvulkan::copy_vulkan(self.resolve_conj().resolve_neg(), dst, false);
        at::Tensor stored = self;
        if (stored.is_conj()) stored = at::_conj(stored);
        if (stored.is_neg()) stored = at::_neg_view(stored);
        at::Tensor resolved = stored.clone();
        at::Tensor pairs = resolved.is_complex() ? at::view_as_real(resolved) : resolved;
        if (self.is_conj()) pairs.select(-1, 1).neg_();
        if (self.is_neg()) pairs.neg_();
        return torchvulkan::copy_vulkan(resolved, dst, false);
    }

    at::Tensor src = (self.sizes() == dst.sizes()) ? self : self.expand(dst.sizes());
    bool src_vulkan = src.device().type() == c10::DeviceType::PrivateUse1;
    bool dst_vulkan = dst.device().type() == c10::DeviceType::PrivateUse1;

    // work happens on the vulkan side's device; the guard puts the caller's device back afterwards
    c10::OptionalDeviceGuard guard(dst_vulkan ? dst.device() : src.device());

    if (!src_vulkan && dst_vulkan) copy_host_to_vulkan(src.cpu(), dst);
    else if (src_vulkan && dst.device().is_cpu()) copy_vulkan_to_host(src, dst);
    else if (src_vulkan && !dst_vulkan) dst.copy_(src.cpu());
    else if (src.device() == dst.device()) copy_vulkan_to_vulkan(src, dst);
    else {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Source and destination are on different Vulkan devices. Copying through the CPU.");
        dst.copy_(src.cpu());
    }

    return dst;
}

namespace {

at::Tensor copy_from_vulkan(
    const at::Tensor& self,
    const at::Tensor& dst,
    bool non_blocking)
{
    return torchvulkan::copy_vulkan(self, dst, non_blocking);
}

at::Tensor copy_from_and_resize_vulkan(
    const at::Tensor& self,
    const at::Tensor& dst)
{
    torchvulkan::resize_vulkan(dst, self.sizes(), c10::nullopt);
    return torchvulkan::copy_vulkan(self, dst, false);
}

at::Tensor& copy_vulkan_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    // swap the arguments to match the expected order
    torchvulkan::copy_vulkan(src, self, non_blocking);
    return self;
}

at::Scalar local_scalar_dense_vulkan(const at::Tensor& self)
{
    TORCH_CHECK(self.numel() > 0, "torchvulkan [ERROR]: _local_scalar_dense: Empty tensor not supported");
    return self.cpu().item();
}

// without this, Module.to("vulkan") replaces each Parameter instead of setting .data, which unties shared weights
bool has_compatible_shallow_copy_type_vulkan(const at::Tensor& self, const at::Tensor& from)
{
    auto dense = [](const at::Tensor& t) { return t.layout() == at::kStrided && (t.is_cpu() || t.is_privateuseone()); };
    return self.unsafeGetTensorImpl()->has_compatible_shallow_copy_type(from.key_set()) || (dense(self) && dense(from));
}

} // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("_copy_from", &copy_from_vulkan);
    m.impl("_copy_from_and_resize", &copy_from_and_resize_vulkan);
    m.impl("copy_", &copy_vulkan_);
    m.impl("_local_scalar_dense", &local_scalar_dense_vulkan);
    m.impl("_has_compatible_shallow_copy_type", &has_compatible_shallow_copy_type_vulkan);
}
