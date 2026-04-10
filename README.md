1. Change `NVBIT_PATH` in `Makefile` to point to where you downloaded and extracted NVBit.

2. Change `ARCH` in `Makefile` to match the compute capability of your specific GPU.

3. Run make to compile the tool into `warp_ldg_counter.so`.

4. Run your normal, unmodified CUDA application, but prefix the command with `LD_PRELOAD`:
```
LD_PRELOAD=./warp_ldg_counter.so ./your_cuda_app
```
