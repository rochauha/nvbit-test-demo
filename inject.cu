#include <stdint.h>

// NVBit requires injected functions to be extern "C" and __noinline__
extern "C" __device__ __noinline__ void count_ldg_per_warp(uint64_t* warp_counters, int max_warps) {

    // 1. Calculate a flat, global Warp ID for this specific kernel launch
    int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    int warps_per_block = (threads_per_block + 31) / 32;

    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int warp_id_in_block = thread_id_in_block / 32;

    int global_warp_id = (block_id * warps_per_block) + warp_id_in_block;

    // Safety check to prevent out-of-bounds memory access
    if (global_warp_id >= max_warps) return;

    // 2. Elect a leader thread among the ACTIVE threads in this warp
    unsigned int active_mask = __activemask();
    int leader_lane = __ffs(active_mask) - 1;
    int lane_id = thread_id_in_block % 32;

    // 3. Only the leader updates the counter for the whole warp
    if (lane_id == leader_lane) {
        // Count how many threads are actually active right now
        int active_thread_count = __popc(active_mask);

        // Add the number of active threads to the warp's total LDG memory requests
        atomicAdd((unsigned long long*)&warp_counters[global_warp_id], active_thread_count);
    }
}
