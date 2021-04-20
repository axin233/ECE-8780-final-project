#include "background_subtraction.h"
#include "device_launch_parameters.h"

__global__ void background_subtract_shared_mem(uchar3 *bg_img, uchar3 *cur_img, unsigned char *result_img, int img_width, int img_height) {

	__shared__ uchar3 ds_bg[TILE_WIDTH][TILE_WIDTH];
	__shared__ uchar3 ds_cur[TILE_WIDTH][TILE_WIDTH];
	
	double square_dist;
	int tx, ty, img_x, img_y;

	// Make sure the tile is within the padded image
	int block_end_col = blockIdx.x*blockDim.x + blockDim.x;
	int block_end_row = blockIdx.y*blockDim.y + blockDim.y;
	const unsigned int blockEndClampedCol = ((block_end_col<img_width) ? block_end_col : img_width);
	const unsigned int blockEndClampedRow = ((block_end_row<img_height) ? block_end_row : img_height);

	// Load data to tiles
	tx = threadIdx.x;
	ty = threadIdx.y;
	img_x = blockIdx.x*blockDim.x + threadIdx.x;
	img_y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((img_y < blockEndClampedRow) && (img_x < blockEndClampedCol)) {
		ds_bg[ty][tx] = bg_img[img_y*img_width + img_x];
		ds_cur[ty][tx] = cur_img[img_y*img_width + img_x];
	}

	// Barrier synchronization
	__syncthreads();

	// Calculate the square of distance in BGR color space
	tx = threadIdx.x;
	ty = threadIdx.y;
	img_x = blockIdx.x*blockDim.x + threadIdx.x;
	img_y = blockIdx.y*blockDim.y + threadIdx.y;
	if ((img_y >= (blockIdx.y*blockDim.y)) && (img_y<blockEndClampedRow) && (img_x >= (blockIdx.x*blockDim.x)) && (img_x < blockEndClampedCol)) {

		square_dist = (ds_bg[ty][tx].x - ds_cur[ty][tx].x) * (ds_bg[ty][tx].x - ds_cur[ty][tx].x) +
			(ds_bg[ty][tx].y - ds_cur[ty][tx].y) * (ds_bg[ty][tx].y - ds_cur[ty][tx].y) +
			(ds_bg[ty][tx].z - ds_cur[ty][tx].z) * (ds_bg[ty][tx].z - ds_cur[ty][tx].z);

		// Generate the detection result
		if (square_dist >= 1100) {
			result_img[img_y*img_width + img_x] = 255;
		}
		else {
			result_img[img_y*img_width + img_x] = 0;
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