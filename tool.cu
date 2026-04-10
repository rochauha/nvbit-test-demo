#include <stdint.h>
#include <stdio.h>
#include <vector>
#include "nvbit_tool.h"
#include "nvbit.h"

// Arbitrary max for this example to allocate memory safely
#define MAX_WARPS 100000

// Global pointer for our managed memory counters
uint64_t* warp_counters = nullptr;

// Runs once when the tool is loaded via LD_PRELOAD
void nvbit_at_init() {
    // Allocate unified memory so the GPU can write to it and the CPU can read it
    cudaMallocManaged(&warp_counters, MAX_WARPS * sizeof(uint64_t));
    cudaMemset(warp_counters, 0, MAX_WARPS * sizeof(uint64_t));
}

// Runs right before any kernel is launched
void nvbit_at_kernel_launch(CUcontext ctx, CUfunction func, nvbit_api_cuda_t cbid,
                            uint32_t gridX, uint32_t gridY, uint32_t gridZ,
                            uint32_t blockX, uint32_t blockY, uint32_t blockZ) {

    // Get all SASS instructions for this kernel
    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, func);

    for (Instr* instr : instrs) {
        // Check if the instruction is a Load AND targets Global Memory (LDG)
        if (instr->isLoad() && instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL) {

            // Inject a call to our device function BEFORE the LDG executes
            nvbit_insert_call(instr, "count_ldg_per_warp", IPOINT_BEFORE);

            // Pass the arguments to the injected device function
            nvbit_add_call_arg_const_val64(instr, (uint64_t)warp_counters);
            nvbit_add_call_arg_const_val32(instr, MAX_WARPS);
        }
    }
}

// Runs when the main application exits
void nvbit_at_term() {
    printf("\n========== NVBit: Global Memory Requests (LDG) Per Warp ==========\n");
    int printed = 0;

    // Print only the warps that actually requested memory
    for (int i = 0; i < MAX_WARPS; i++) {
        if (warp_counters[i] > 0) {
            printf("Warp ID %d: %llu individual memory requests\n", i, (unsigned long long)warp_counters[i]);
            printed++;
        }
    }

    if (printed == 0) {
        printf("No LDG instructions recorded.\n");
    }
    printf("==================================================================\n\n");

    cudaFree(warp_counters);
}
