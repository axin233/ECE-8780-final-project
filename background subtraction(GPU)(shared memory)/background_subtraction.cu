#include "background_subtraction.h"
#include "device_launch_parameters.h"

__global__ void background_subtract_shared_mem(uchar3 *bg_img, uchar3 *cur_img, unsigned char *result_img, int img_width, int img_height) {

	__shared__ uchar3 ds_bg[TILE_WIDTH][TILE_WIDTH];
	__shared__ uchar3 ds_cur[TILE_WIDTH][TILE_WIDTH];
	__shared__ int tile_idx;

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	double square_dist;
	int tx, ty;

	// Load data to tiles
	tx = threadIdx.x;
	ty = threadIdx.y;
	tile_idx = x / TILE_WIDTH;
	if ((y < img_height) && ((tile_idx*TILE_WIDTH + threadIdx.x) < img_width)) {
		ds_bg[ty][tx] = bg_img[y*img_width + tile_idx*TILE_WIDTH + tx];
		ds_cur[ty][tx] = cur_img[y*img_width + tile_idx*TILE_WIDTH + tx];
	}

	// Barrier synchronization
	__syncthreads();

	// Calculate the square of distance in BGR color space
	tx = threadIdx.x;
	ty = threadIdx.y;
	tile_idx = x / TILE_WIDTH;
	if ((y < img_height) && ((tile_idx*TILE_WIDTH + threadIdx.x) < img_width)) {

		square_dist = (ds_bg[ty][tx].x - ds_cur[ty][tx].x) * (ds_bg[ty][tx].x - ds_cur[ty][tx].x) +
			(ds_bg[ty][tx].y - ds_cur[ty][tx].y) * (ds_bg[ty][tx].y - ds_cur[ty][tx].y) +
			(ds_bg[ty][tx].z - ds_cur[ty][tx].z) * (ds_bg[ty][tx].z - ds_cur[ty][tx].z);

		// Generate the detection result
		if (square_dist >= 1100) {
			result_img[y*img_width + tile_idx*TILE_WIDTH + tx] = 255;
		}
		else {
			result_img[y*img_width + tile_idx*TILE_WIDTH + tx] = 0;
		}
	}


}

// The function for the background subtraction algorithm
void background_subtract_gpu(uchar3 *d_bg, uchar3 *d_cur_img,
	unsigned char *d_result, int img_width, int img_height) {

	dim3 gridSize(int(ceil(img_width / BLOCK) + 1), int(ceil(img_height / BLOCK) + 1), 1);
	dim3 blockSize(int(BLOCK), int(BLOCK), 1);

	background_subtract_shared_mem KERNEL_ARGS2(gridSize, blockSize) (d_bg, d_cur_img, d_result, img_width, img_height);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}