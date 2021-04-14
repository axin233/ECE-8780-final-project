#include "background_subtraction.h"
#include "device_launch_parameters.h"

__global__ void construct_background_kernel(uchar3 *img_1, uchar3 *img_2, uchar3 *img_3, uchar3 *img_4, 
	uchar3 *img_5, uchar3 *img, int img_width, int img_height) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	int idx = 0;

	// Check the boundary
	if ((x < img_width) && (y < img_height)) {

		// A coalesced access
		idx = y*img_width + x;

		// Construct the background by averaging pixels within 5 images
		img[idx].x =unsigned char((0.2) * (img_1[idx].x + img_2[idx].x + img_3[idx].x + img_4[idx].x + img_5[idx].x));
		img[idx].y = unsigned char((0.2) * (img_1[idx].y + img_2[idx].y + img_3[idx].y + img_4[idx].y + img_5[idx].y));
		img[idx].z = unsigned char((0.2) * (img_1[idx].z + img_2[idx].z + img_3[idx].z + img_4[idx].z + img_5[idx].z));
	}
}

// The function for constructing background
void construct_background(uchar3 *d_bg_1, uchar3 *d_bg_2, uchar3 *d_bg_3,
	uchar3 *d_bg_4, uchar3 *d_bg_5, uchar3 *d_bg, int img_width, int img_height) {

	dim3 gridSize(int(ceil(img_width / BLOCK) + 1), int(ceil(img_height / BLOCK) + 1), 1);
	dim3 blockSize(int(BLOCK), int(BLOCK), 1);

	construct_background_kernel KERNEL_ARGS2(gridSize, blockSize) (d_bg_1, d_bg_2, d_bg_3, d_bg_4, d_bg_5, d_bg, img_width, img_height);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}