#pragma once
#include <volk.h>
#include <array>
#include <cstdint>
#include <vector>
#include <memory>
#include <unordered_map>

#include "builders.h"

class DeviceContext;
struct OpInfo;
struct MemoryRange;

class DAGDispatcher 
{
public:
    DAGDispatcher(DeviceContext* device) : device(device) {}
    void dispatch(std::vector<OpInfo*> operations);

private:
    DeviceContext* device = nullptr;
    std::mutex mutex_;

    void recordOp(VkCommandBuffer cmd, const OpInfo* op);
    void recordCompute(VkCommandBuffer cmd, const OpInfo* op);
    void recordCopy(VkCommandBuffer cmd, const OpInfo* op);
    void recordBarrier(VkCommandBuffer cmd);

    // all elements in the vector depend on the key
    std::unordered_map<OpInfo*, std::vector<OpInfo*>> read_dependency;
    std::unordered_map<OpInfo*, std::vector<OpInfo*>> write_dependency;
};
