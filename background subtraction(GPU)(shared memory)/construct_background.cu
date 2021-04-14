#include "background_subtraction.h"
#include "device_launch_parameters.h"

__global__ void construct_background_shared_mem(uchar3 *img_1, uchar3 *img_2, uchar3 *img_3, uchar3 *img_4,
	uchar3 *img_5, uchar3 *img, int img_width, int img_height) {

	__shared__ uchar3 ds_img_1[TILE_WIDTH][TILE_WIDTH];
	__shared__ uchar3 ds_img_2[TILE_WIDTH][TILE_WIDTH];
	__shared__ uchar3 ds_img_3[TILE_WIDTH][TILE_WIDTH];
	__shared__ uchar3 ds_img_4[TILE_WIDTH][TILE_WIDTH];
	__shared__ uchar3 ds_img_5[TILE_WIDTH][TILE_WIDTH];
	__shared__ int tile_idx;

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int tx, ty;

	// Load data to tiles
	tx = threadIdx.x;
	ty = threadIdx.y;
	tile_idx = x / TILE_WIDTH;
	if ((y < img_height) && ((tile_idx*TILE_WIDTH + threadIdx.x) < img_width)) {
		ds_img_1[ty][tx] = img_1[y*img_width + tile_idx*TILE_WIDTH + tx];
		ds_img_2[ty][tx] = img_2[y*img_width + tile_idx*TILE_WIDTH + tx];
		ds_img_3[ty][tx] = img_3[y*img_width + tile_idx*TILE_WIDTH + tx];
		ds_img_4[ty][tx] = img_4[y*img_width + tile_idx*TILE_WIDTH + tx];
		ds_img_5[ty][tx] = img_5[y*img_width + tile_idx*TILE_WIDTH + tx];
	}

	// Barrier synchronization
	__syncthreads();

	// Construct the background by averaging pixels within 5 images
	tx = threadIdx.x;
	ty = threadIdx.y;
	tile_idx = x / TILE_WIDTH;
	if ((y < img_height) && ((tile_idx*TILE_WIDTH + threadIdx.x) < img_width)) {

		img[y*img_width + tile_idx*TILE_WIDTH + tx].x 
			= unsigned char((0.2) * (ds_img_1[ty][tx].x + ds_img_2[ty][tx].x + ds_img_3[ty][tx].x + ds_img_4[ty][tx].x + ds_img_5[ty][tx].x));
		img[y*img_width + tile_idx*TILE_WIDTH + tx].y
			= unsigned char((0.2) * (ds_img_1[ty][tx].y + ds_img_2[ty][tx].y + ds_img_3[ty][tx].y + ds_img_4[ty][tx].y + ds_img_5[ty][tx].y));
		img[y*img_width + tile_idx*TILE_WIDTH + tx].z
			= unsigned char((0.2) * (ds_img_1[ty][tx].z + ds_img_2[ty][tx].z + ds_img_3[ty][tx].z + ds_img_4[ty][tx].z + ds_img_5[ty][tx].z));
	}

	

}

// The function for constructing background
void construct_background(uchar3 *d_bg_1, uchar3 *d_bg_2, uchar3 *d_bg_3,
	uchar3 *d_bg_4, uchar3 *d_bg_5, uchar3 *d_bg, int img_width, int img_height) {

	dim3 gridSize(int(ceil(img_width / BLOCK) + 1), int(ceil(img_height / BLOCK) + 1), 1);
	dim3 blockSize(int(BLOCK), int(BLOCK), 1);

	construct_background_shared_mem KERNEL_ARGS2(gridSize, blockSize) (d_bg_1, d_bg_2, d_bg_3, d_bg_4, d_bg_5, d_bg, img_width, img_height);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}