# IMPORTANT: Update this path to where you extracted the NVBit release
NVBIT_PATH ?= /path/to/nvbit_release

# IMPORTANT: Update this to match your GPU architecture (e.g., sm_70, sm_80)
ARCH ?= sm_70

NVCC ?= nvcc

# Include paths for NVBit headers and libraries
INCLUDES  := -I$(NVBIT_PATH)/core
LIBS      := -L$(NVBIT_PATH)/core -lnvbit

# The final tool library name
TOOL_SO   := warp_ldg_counter.so

all: $(TOOL_SO)

# 1. Compile the injected device code into a cubin
inject.cubin: inject.cu
	$(NVCC) -arch=$(ARCH) -c -cubin inject.cu -o inject.cubin

# 2. Compile the host tool and link it into a shared library (.so)
$(TOOL_SO): tool.cu inject.cubin
	$(NVCC) -arch=$(ARCH) -O3 $(INCLUDES) -Xcompiler -fPIC -shared tool.cu $(LIBS) -o $(TOOL_SO)

clean:
	rm -f *.so *.cubin
