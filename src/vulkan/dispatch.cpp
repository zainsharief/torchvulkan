#include "dispatch.h"
#include "vulkan_context.h"
#include <vector>
#include <unordered_map>
#include <cassert>

bool MemoryRange::overlaps(const MemoryRange& other) const
{
    return (byte_offset < other.byte_offset + other.byte_size) &&
           (byte_offset + byte_size > other.byte_offset);
}

void DAGDispatcher::recordOp(VkCommandBuffer cmd, const OpInfo* op)
{
    switch (op->type)
    {
        case OpType::COMPUTE: recordCompute(cmd, op); break;
        case OpType::COPY: recordCopy(cmd, op); break;
    }
}

void DAGDispatcher::recordCompute(VkCommandBuffer cmd, const OpInfo* op)
{
    device->device_table.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, op->pipeline);
    if (op->pushConstantSize > 0)
    {
        device->device_table.vkCmdPushConstants(
            cmd, op->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, op->pushConstantSize, op->pushConstantData.data()
        );
    }
    device->device_table.vkCmdDispatch(cmd, op->groupX, op->groupY, op->groupZ);
}

void DAGDispatcher::recordCopy(VkCommandBuffer cmd, const OpInfo* op)
{
    device->device_table.vkCmdCopyBuffer(cmd, op->srcBuffer, op->dstBuffer, 1, &op->region);
}

void DAGDispatcher::recordBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                            VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers    = &barrier;

    device->device_table.vkCmdPipelineBarrier2KHR(cmd, &dep);
}

void DAGDispatcher::dispatch(const std::vector<OpInfo*> operations)
{
    read_dependency.clear();
    write_dependency.clear();

    std::unordered_map<uint64_t, OpInfo*> last_writer;
    std::unordered_map<uint64_t, std::vector<OpInfo*>> readers_since_write;
    std::unordered_map<OpInfo*, int> unresolved_deps;

    for (OpInfo* op : operations)
    {
        for (MemoryRange* input : op->inputs)
        {
            // RAW: this read must wait for the previous write
            auto it = last_writer.find(input->base_addr);
            if (it != last_writer.end())
            {
                read_dependency[it->second].push_back(op);
                unresolved_deps[op]++;
            }
            readers_since_write[input->base_addr].push_back(op);
        }

        for (MemoryRange* output : op->outputs)
        {
            auto it_write = last_writer.find(output->base_addr);
            auto it_read = readers_since_write.find(output->base_addr);

            // WAW: this write must wait for the previous writer
            if (it_write != last_writer.end())
            {
                OpInfo* write_dep = it_write->second;
                write_dependency[write_dep].push_back(op);
                unresolved_deps[op]++;
            }

            // WAR: this write must wait for all readers since the last write
            if (it_read != readers_since_write.end())
            {
                std::vector<OpInfo*> read_dep = it_read->second;
                for (OpInfo* dep : read_dep)
                {
                    if (dep == op) continue;
                    read_dependency[dep].push_back(op);
                    unresolved_deps[op]++;
                }
            }

            last_writer[output->base_addr] = op;
            readers_since_write.erase(output->base_addr);
        }
    }

    VkCommandBuffer cmd = device->cache.allocateCommandBuffer(device->commandPool);

    std::vector<OpInfo*> ready;
    for (OpInfo* op : operations)
    {
        if (unresolved_deps[op] == 0) ready.push_back(op);
    }

    while (!ready.empty())
    {
        // record the whole wave together so each op keeps its own pipeline / push constants / copy region
        for (OpInfo* op : ready) recordOp(cmd, op);

        std::vector<OpInfo*> next_wave;

        for (OpInfo* op : ready)
        {
            for (OpInfo* dependent : read_dependency[op])
            {
                unresolved_deps[dependent]--;
                if (unresolved_deps[dependent] == 0) next_wave.push_back(dependent);
            }
            
            for (OpInfo* dependent : write_dependency[op])
            {
                unresolved_deps[dependent]--;
                if (unresolved_deps[dependent] == 0) next_wave.push_back(dependent);
            }
        }

        if (!next_wave.empty()) recordBarrier(cmd);

        ready = std::move(next_wave);
    }

    device->device_table.vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFence fence = device->cache.allocateFence();

    std::unique_lock<std::mutex> lock(mutex_);
    VkResult submit_result = device->device_table.vkQueueSubmit(device->computeQueue, 1, &submitInfo, fence);
    lock.unlock();

    device->cache.deleteCommandBuffer(cmd);

    TORCH_CHECK(submit_result == VK_SUCCESS, "torchvulkan [ERROR]: vkQueueSubmit failed with VkResult ", submit_result);

    VkResult wait_result = device->device_table.vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (wait_result != VK_SUCCESS) {
        TORCH_CHECK(false, "torchvulkan [ERROR]: Wait for fence failed with VkResult ", wait_result);
    }

    device->cache.deleteFence(fence);
}