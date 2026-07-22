#include <torch/extension.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>
#include "api/ops/factory.h"

at::Tensor torchvulkan::empty_memory_format_vulkan(
    c10::SymIntArrayRef size,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> layout,
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */,
    c10::optional<at::MemoryFormat> memory_format) 
{
    at::ScalarType dtype_ = dtype.value_or(at::kFloat);
    at::Layout layout_ = layout.value_or(at::kStrided);
    at::MemoryFormat memory_format_ = memory_format.value_or(at::MemoryFormat::Contiguous);

    TORCH_CHECK(layout_ == at::kStrided, "torchvulkan [ERROR]: Only Strided layout is supported");
    
    std::vector<int64_t> concrete_sizes;
    concrete_sizes.reserve(size.size());
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }
    
    int64_t numel = c10::multiply_integers(concrete_sizes);
    size_t nbytes = numel * c10::elementSize(dtype_);

    std::vector<int64_t> strides = compute_strides(concrete_sizes, memory_format_);

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetAllocator(c10::DeviceType::PrivateUse1),
        /* resizeable = */ true 
    );
    
    at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
        std::move(storage_impl),
        c10::DispatchKey::PrivateUse1,
        c10::scalarTypeToTypeMeta(dtype_)
    );
    
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, strides);

    return tensor;
}

at::Tensor torchvulkan::empty_strided_vulkan(
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<at::ScalarType> dtype,
    c10::optional<at::Layout> /* layout */, // assumed to be strided, so ignored
    c10::optional<at::Device> /* device */,
    c10::optional<bool> /* pin_memory */) 
{
    at::ScalarType dtype_ = dtype.value_or(at::kFloat);
    
    std::vector<int64_t> concrete_sizes;
    std::vector<int64_t> concrete_strides;
    concrete_sizes.reserve(size.size());
    concrete_strides.reserve(stride.size());
    
    for (const auto& s : size) {
        int64_t val = s.guard_int(__FILE__, __LINE__);
        concrete_sizes.push_back(val);
    }
    for (const auto& s : stride) {
        int64_t val = s.guard_int(__FILE__, __LINE__);
        concrete_strides.push_back(val);
    }
    
    size_t nbytes = at::detail::computeStorageNbytes(
        concrete_sizes, 
        concrete_strides, 
        c10::elementSize(dtype_)
    );

    auto storage_impl = c10::make_intrusive<c10::StorageImpl>(
        c10::StorageImpl::use_byte_size_t(),
        nbytes,
        c10::GetAllocator(c10::DeviceType::PrivateUse1),
        /* resizeable = */ true 
    );
    
    at::Tensor tensor = at::detail::make_tensor<at::TensorImpl>(
        std::move(storage_impl),
        c10::DispatchKey::PrivateUse1,
        c10::scalarTypeToTypeMeta(dtype_)
    );
    
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, concrete_strides);

    return tensor;
}

at::Tensor torchvulkan::copy_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst, 
    bool non_blocking) 
{
    TORCH_CHECK(self.sizes() == dst.sizes(), "torchvulkan [ERROR]: Copy sizes mismatch");

    at::Tensor src = self;
    c10::DeviceType src_type = src.device().type();
    c10::DeviceType dst_type = dst.device().type();

    if (dst_type == c10::DeviceType::PrivateUse1) VulkanContext::SetCurrentDevice(dst.device().index());
    else if (src_type == c10::DeviceType::PrivateUse1) VulkanContext::SetCurrentDevice(src.device().index());
    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();

    if (src_type == at::DeviceType::CPU && dst_type == c10::DeviceType::PrivateUse1) 
    {
        if (self.scalar_type() != dst.scalar_type()) src = self.to(dst.scalar_type());
        
        void* dest_ptr = (void*)dst.storage().data_ptr().get_context();
        uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
        
        if (!src.is_contiguous()) src = src.contiguous();
        if (dst.is_contiguous()) {
            globalVulkanAllocator.copy_host_to_device(dest_ptr, dest_offset, src.data_ptr(), dst.nbytes());
            return dst;
        }

        at::Tensor stagingBuffer = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
        void* staging_buffer_ptr = (void*)stagingBuffer.storage().data_ptr().get_context();
        uint64_t staging_buffer_offset = stagingBuffer.storage_offset() * dst.itemsize();

        globalVulkanAllocator.copy_host_to_device(staging_buffer_ptr, staging_buffer_offset, src.data_ptr(), dst.nbytes());
        torchvulkan::dispatch_copy_shader(stagingBuffer, dst);
    }
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == at::DeviceType::CPU) 
    {
        if (self.scalar_type() != dst.scalar_type()) 
        {
            if (!is_dtype_supported(src.scalar_type()) || !is_dtype_supported(dst.scalar_type())) { 
                TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support source or destination dtype. Falling back to CPU for copy.");
                at::Tensor out = dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
                return out;
            }

            at::Tensor temp = at::empty_like(src, src.options().dtype(dst.scalar_type()).memory_format(at::MemoryFormat::Contiguous));
            torchvulkan::dispatch_cast_shader(src, temp);
            src = std::move(temp);
        }

        if (!src.is_contiguous()) src = src.contiguous();

        void* src_ptr = (void*)src.storage().data_ptr().get_context();
        uint64_t src_offset = src.storage_offset() * src.itemsize();

        if (dst.is_contiguous()) {    
            globalVulkanAllocator.copy_device_to_host(dst.data_ptr(), src_ptr, src_offset, dst.nbytes());
            return dst;
        }

        at::Tensor stagingBuffer = at::empty_like(dst, dst.options().memory_format(at::MemoryFormat::Contiguous));
        globalVulkanAllocator.copy_device_to_host(stagingBuffer.data_ptr(), src_ptr, src_offset, stagingBuffer.nbytes());
        dst.copy_(stagingBuffer, non_blocking);
    } 
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == c10::DeviceType::PrivateUse1 && src.device().index() == dst.device().index()) 
    {        
        if (self.scalar_type() != dst.scalar_type()) 
        {
            if (!is_dtype_supported(src.scalar_type()) || !is_dtype_supported(dst.scalar_type())) { 
                TORCH_WARN_ONCE("torchvulkan [WARNING]: Vulkan device does not support source or destination dtype. Falling back to CPU for copy.");
                at::Tensor out = dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
                return out;
            }

            torchvulkan::dispatch_cast_shader(src, dst);
            return dst;
        }
        
        if (!src.is_contiguous() || !dst.is_contiguous()) {
            torchvulkan::dispatch_copy_shader(src, dst);
            return dst;
        }

        void* src_ptr = (void*)src.storage().data_ptr().get_context();
        uint64_t src_offset = src.storage_offset() * src.itemsize();
        void* dest_ptr = (void*)dst.storage().data_ptr().get_context();
        uint64_t dest_offset = dst.storage_offset() * dst.itemsize();
        globalVulkanAllocator.copy_device_to_device(dest_ptr, dest_offset, src_ptr, src_offset, dst.nbytes());  
    } 
    else if (src_type == c10::DeviceType::PrivateUse1 && dst_type == c10::DeviceType::PrivateUse1 && src.device().index() != dst.device().index()) 
    {
        TORCH_WARN_ONCE("torchvulkan [WARNING]: Source and destination are on different Vulkan devices. Falling back to CPU for copy.");
        dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
        return dst;
    } 
    else {
        // should never get here, but just in case
        dst.copy_(src.to(at::DeviceType::CPU), non_blocking);
    }

    return dst;
}

at::Tensor torchvulkan::copy_from_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst, 
    bool non_blocking) 
{
    return torchvulkan::copy_vulkan(self, dst, non_blocking);
}

at::Tensor torchvulkan::copy_from_and_resize_vulkan(
    const at::Tensor& self, 
    const at::Tensor& dst) 
{
    torchvulkan::resize_vulkan(dst, self.sizes(), c10::nullopt);
    return torchvulkan::copy_vulkan(self, dst, false);
}

at::Tensor& torchvulkan::copy_vulkan_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
    // swap the arguments to match the expected order
    torchvulkan::copy_vulkan(src, self, non_blocking);
    return self;
}

at::Tensor torchvulkan::as_strided_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size,
    c10::SymIntArrayRef stride,
    c10::optional<c10::SymInt> storage_offset)
{
    std::vector<int64_t> concrete_sizes;
    std::vector<int64_t> concrete_strides;
    concrete_sizes.reserve(size.size());
    concrete_strides.reserve(stride.size());
    
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }
    for (const auto& s : stride) {
        concrete_strides.push_back(s.guard_int(__FILE__, __LINE__));
    }
    
    int64_t offset = storage_offset.has_value() 
        ? storage_offset.value().guard_int(__FILE__, __LINE__) 
        : self.storage_offset();

    at::Tensor result = at::detail::make_tensor<at::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype()
    );
    
    // update the sizes, strides, and offset in the view
    result.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, concrete_strides);
    result.unsafeGetTensorImpl()->set_storage_offset(offset);
    
    return result;
}

const at::Tensor& torchvulkan::resize_vulkan(
    const at::Tensor& self, 
    c10::IntArrayRef size, 
    c10::optional<at::MemoryFormat> memory_format) 
{
    at::TensorImpl* impl = self.unsafeGetTensorImpl();

    int64_t numel = 1;
    for (auto s : size) numel *= s;
    size_t new_bytes = numel * self.itemsize();

    if (impl->storage().nbytes() < new_bytes) {
        auto new_storage = c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            new_bytes,
            c10::GetAllocator(c10::DeviceType::PrivateUse1),
            /* resizeable = */ true
        );
        impl->set_storage_keep_dtype(std::move(new_storage));
    }

    at::MemoryFormat format = memory_format.value_or(at::MemoryFormat::Contiguous);
    std::vector<int64_t> strides = compute_strides(size, format);
    impl->set_sizes_and_strides(size, strides);

    return self;
}

at::Tensor torchvulkan::reshape_alias_vulkan(
    const at::Tensor& self, 
    c10::SymIntArrayRef sizes, 
    c10::SymIntArrayRef strides) 
{
    auto result = at::detail::make_tensor<c10::TensorImpl>(
        c10::TensorImpl::VIEW,
        c10::Storage(self.storage()),
        self.key_set(),
        self.dtype()
    );
    
    auto* result_impl = result.unsafeGetTensorImpl();
    result_impl->set_storage_offset(self.storage_offset());
    result_impl->set_sizes_and_strides(sizes, strides);
    
    return result;
}

at::Tensor torchvulkan::t_vulkan(const at::Tensor& self)
{
    TORCH_CHECK(self.dim() <= 2, "torchvulkan [ERROR]: t() expects a tensor with <= 2 dimensions.");
    if (self.dim() < 2) return self.alias();
    return transpose_int_vulkan(self, 0, 1);
}

at::Tensor torchvulkan::transpose_int_vulkan(const at::Tensor& self, int64_t dim0, int64_t dim1)
{
    int64_t ndim = self.dim();
    dim0 = c10::maybe_wrap_dim(dim0, ndim);
    dim1 = c10::maybe_wrap_dim(dim1, ndim);

    if (dim0 == dim1) return self.alias();

    std::vector<int64_t> sizes(self.sizes().begin(), self.sizes().end());
    std::vector<int64_t> strides(self.strides().begin(), self.strides().end());
    std::swap(sizes[dim0], sizes[dim1]);
    std::swap(strides[dim0], strides[dim1]);

    return self.as_strided(sizes, strides, self.storage_offset());
}

at::Tensor torchvulkan::permute_vulkan(const at::Tensor& self, c10::IntArrayRef dims)
{
    int64_t ndim = self.dim();
    TORCH_CHECK((int64_t)dims.size() == ndim, "torchvulkan [ERROR]: permute dims size does not match tensor dimensions.");

    std::vector<int64_t> sizes(ndim);
    std::vector<int64_t> strides(ndim);
    for (int64_t i = 0; i < ndim; i++) {
        int64_t d = c10::maybe_wrap_dim(dims[i], ndim);
        sizes[i] = self.size(d);
        strides[i] = self.stride(d);
    }

    return self.as_strided(sizes, strides, self.storage_offset());
}

at::Tensor torchvulkan::view_vulkan(
    const at::Tensor& self,
    c10::SymIntArrayRef size)
{
    std::vector<int64_t> concrete_sizes;
    concrete_sizes.reserve(size.size());
    for (const auto& s : size) {
        concrete_sizes.push_back(s.guard_int(__FILE__, __LINE__));
    }

    at::DimVector inferred_sizes = at::infer_size_dv(concrete_sizes, self.numel());
    concrete_sizes.assign(inferred_sizes.begin(), inferred_sizes.end());

    auto stride = at::detail::computeStride(self.sizes(), self.strides(), concrete_sizes);
    TORCH_CHECK(stride.has_value(), "torchvulkan [ERROR]: View size is not compatible with input tensor's size and stride.");
    
    at::Tensor result = self.alias();
    result.unsafeGetTensorImpl()->set_sizes_and_strides(concrete_sizes, stride.value());
    result.unsafeGetTensorImpl()->set_storage_offset(self.storage_offset());
    
    return result;
}

at::Scalar torchvulkan::local_scalar_dense_vulkan(const at::Tensor& self)
{
    return self.cpu().item();
}

at::Tensor torchvulkan::contiguous_vulkan(const at::Tensor& self, at::MemoryFormat memory_format)
{
    if (self.is_contiguous(memory_format)) return self;
    at::Tensor result = at::empty_like(self, self.options().memory_format(memory_format));

    // call directly rather than through dispatcher to avoid weird fallbacks that trigger infinite recursive loops
    return torchvulkan::copy_vulkan(self, result, false); 
}

at::Tensor torchvulkan::clone_vulkan(const at::Tensor& self, c10::optional<at::MemoryFormat> memory_format) 
{
    c10::TensorOptions options = self.options();
    if (memory_format.has_value()) options = options.memory_format(memory_format.value());
    at::Tensor result = at::empty_like(self, options);

    // call directly rather than through dispatcher to avoid weird fallbacks that trigger infinite recursive loops
    return torchvulkan::copy_vulkan(self, result, false);
}

void torchvulkan::dispatch_copy_shader(const at::Tensor& src, const at::Tensor& dst) 
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .add_output(dst)
        .add_input(src)
        .build();

    uint32_t numel = iter.numel();
    if (numel == 0) return;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_CHECK(false, "torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_copy(dst.scalar_type());
    uint32_t vecSize = get_dtype_vec_size(dst.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize);

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(workgroupSizeX);
    uint32_t key = (workgroupSizeX << 4) | out_dims;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes; 
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    int64_t el_size = iter.element_size(0);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_out = iter.strides(0);
    at::IntArrayRef iter_strides_in = iter.strides(1);

    for (int i = 0; i < out_dims; i++) {
        sizes.set(i, iter_shape[i]);
        strides_in[i] = iter_strides_in[i] / el_size;
        strides_out[i] = iter_strides_out[i] / el_size;
    }
    
    PushConstantBuilder pcs{};
    pcs.push(sizes)
        .push_array(strides_in)
        .push_array(strides_out)
        .push(numel);

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {src, dst}, 
        groupX, 1, 1
    );
}

void torchvulkan::dispatch_cast_shader(const at::Tensor& src, const at::Tensor& dst)
{
    at::TensorIterator iter = at::TensorIteratorConfig()
        .set_check_mem_overlap(true)
        .check_all_same_dtype(false)
        .add_output(dst)
        .add_input(src)
        .build();

    uint32_t numel = iter.numel();
    if (numel == 0) return;
    int32_t out_dims = static_cast<int32_t>(iter.ndim());
    if (out_dims > MAX_DIMS) {
        TORCH_CHECK(false, "torchvulkan [WARNING]: Coalesced dimensions (", out_dims, ") exceed maximum supported (", MAX_DIMS, "). Falling back to CPU.");
    }

    DeviceContext* device = VulkanContext::Instance().CurrentDeviceContext();
    torchvulkan::ShaderID shader_id = torchvulkan::get_shader_id_cast(src.scalar_type(), dst.scalar_type());
    uint32_t vecSize = get_dtype_vec_size(dst.scalar_type()); // our workgroup must match the shader workgroup
    uint32_t workgroupSizeX = get_dtype_workgroup_size(dst.scalar_type(), vecSize);
    uint32_t isBoolCast = (dst.scalar_type() == at::kBool) ? 1 : 0;

    SpecializationBuilder spd{};
    spd.push(out_dims)
        .push(workgroupSizeX)
        .push(isBoolCast);
    uint32_t key = (isBoolCast << 8) | (workgroupSizeX << 4) | out_dims;
    SpecializationArgs specialization = {spd.data(), spd.offsets(), spd.sizes(), spd.numConstants(), key};

    IntDivider sizes; 
    uint32_t strides_in[MAX_DIMS] = {0};
    uint32_t strides_out[MAX_DIMS] = {0};
    
    int64_t el_size_out = iter.element_size(0);
    int64_t el_size_in = iter.element_size(1);
    at::IntArrayRef iter_shape = iter.shape();
    at::IntArrayRef iter_strides_out = iter.strides(0);
    at::IntArrayRef iter_strides_in = iter.strides(1);

    for (int i = 0; i < out_dims; i++) {
        sizes.set(i, iter_shape[i]);
        strides_in[i] = iter_strides_in[i] / el_size_in;
        strides_out[i] = iter_strides_out[i] / el_size_out;
    }

    PushConstantBuilder pcs{};
    pcs.push(sizes)
        .push_array(strides_in)
        .push_array(strides_out)
        .push(numel);

    uint32_t groupX = (numel + (workgroupSizeX - 1)) / workgroupSizeX;

    VulkanShader shader(shader_id, specialization, device);
    shader.dispatch(
        &pcs, 
        pcs.size(), 
        {src, dst}, 
        groupX, 1, 1
    );
}