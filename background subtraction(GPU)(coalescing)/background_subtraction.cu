#include "background_subtraction.h"
#include "device_launch_parameters.h"

__global__ void background_subtract_kernel(uchar3 *bg_img, uchar3 *cur_img, unsigned char *result_img, int img_width, int img_height) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	double square_dist;
	int idx = 0;

	// Boundary check
	if ((x < img_width) && (y < img_height)) {

		// A coalesced access
		idx = y*img_width + x;

		// Calculate the square of distance in BGR color space
		square_dist = (bg_img[idx].x - cur_img[idx].x) * (bg_img[idx].x - cur_img[idx].x) +
			(bg_img[idx].y - cur_img[idx].y) * (bg_img[idx].y - cur_img[idx].y) + 
			(bg_img[idx].z - cur_img[idx].z) * (bg_img[idx].z - cur_img[idx].z);

		// Generate the detection result
		if (square_dist >= 1100) {
			result_img[idx] = 255;
		}
		else {
			result_img[idx] = 0;
		}
	}
}

// The function for the background subtraction algorithm
void background_subtract_gpu(uchar3 *d_bg, uchar3 *d_cur_img,
	unsigned char *d_result, int img_width, int img_height) {

	dim3 gridSize(int(ceil(img_width / BLOCK) + 1), int(ceil(img_height / BLOCK) + 1), 1);
	dim3 blockSize(int(BLOCK), int(BLOCK), 1);

	background_subtract_kernel KERNEL_ARGS2(gridSize, blockSize) (d_bg, d_cur_img, d_result, img_width, img_height);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}