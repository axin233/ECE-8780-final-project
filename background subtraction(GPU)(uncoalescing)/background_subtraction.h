#include <cuda_runtime.h> 
#include <cuda.h> 
#include "utils.h"
#include <algorithm>

#define BLOCK 32.0

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif


// The function for constructing background
void construct_background(uchar3 *d_bg_1, uchar3 *d_bg_2, uchar3 *d_bg_3, 
	uchar3 *d_bg_4, uchar3 *d_bg_5, uchar3 *d_bg,  int img_width, int img_height);

// The background subtraction algorithm
void background_subtract_gpu(uchar3 *d_bg, uchar3 *d_cur_img, 
	unsigned char *d_result, int img_width, int img_height);

