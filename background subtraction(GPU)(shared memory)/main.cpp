#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "utils.h"
#include "background_subtraction.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])//
{
	int frameNumber = 1;
	int numRow, numCol, numPixels;
	Mat Original, mask_for_circle, membrane_img, membrane_img_copy;
	bool is_first_frame = true;
	Point membrane_center;
	int membrane_radius;
	chrono::high_resolution_clock::time_point start_time, current_time;
	uchar3 *h_bg_1, *h_bg_2, *h_bg_3, *h_bg_4, *h_bg_5;
	uchar3 *d_bg_1, *d_bg_2, *d_bg_3, *d_bg_4, *d_bg_5, *d_bg;
	uchar3 *h_cur_img, *d_cur_img;
	unsigned char *d_result;
	Mat bg_img_cpu_temp, result_cpu_temp;
	int64_t time_diff;
	float time_diff_float;
	FILE* frame_rate_f;
	string frame_rate_n, needle_n, thread_n, original_n;
	string file_path = "D:/visual_studio_code/HPC_GPU/results/";

	// The membrane center
	membrane_center.x = 321;
	membrane_center.y = 226;

	// The membrane radius
	membrane_radius = 210;

	// For checking the frame rate
	start_time = chrono::high_resolution_clock::now();
	frame_rate_n = file_path + "frame_rate/frame_rate(GPU).txt";
	frame_rate_f = fopen(frame_rate_n.c_str(), "w");

	// Set the compression ratio of png images
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	// Video directory
	VideoCapture cap("D:/videos_from_the_internal_realSense/using_12v_adaptor/Subject_21_1_Internal_Camera.avi");

	// Check if the video has been opened successfully
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video file. Press any key to exit." << endl;
		cin.get(); 
		return -1;
	}

	// Obtain video frames one by one
	while (1)
	{
		// Obtain images from the video
		bool bSuccess = cap.read(Original);

		// Break the loop when reaching the end of the video
		if (bSuccess == false)
		{
			cout << "Found the end of the video" << endl;
			break;
		}

		// Initialization
		if (is_first_frame == true) {

			// Create a mask for the membrane 
			mask_for_circle.create(Original.rows, Original.cols, CV_8UC1);
			mask_for_circle.setTo(Scalar(0, 0, 0));
			circle(mask_for_circle, membrane_center, membrane_radius, Scalar(255, 255, 255), -1, 8, 0);

			// (GPU) Allocate memory on GPU
			numRow = Original.rows;
			numCol = Original.cols;
			numPixels = numRow*numCol;
			checkCudaErrors(cudaMalloc((void**)&d_bg_1, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_2, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_3, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_4, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_5, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_cur_img, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(unsigned char)*numPixels));

			// Reset parameters
			is_first_frame = false;
		}

		// Remove the peripheral area within video frames
		// After the process, only the membrane area remains within video frames 
		Original.copyTo(membrane_img, mask_for_circle);
		membrane_img_copy = membrane_img.clone();

		// (GPU) Construct the background
		if (frameNumber == 1) {
			//h_bg_1 = Original.ptr<uchar3>(0);
			h_bg_1 = membrane_img_copy.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_1, h_bg_1, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if(frameNumber==2) {
			//h_bg_2 = Original.ptr<uchar3>(0);
			h_bg_2 = membrane_img_copy.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_2, h_bg_2, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if (frameNumber == 3) {
			//h_bg_3 = Original.ptr<uchar3>(0);
			h_bg_3 = membrane_img_copy.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_3, h_bg_3, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if (frameNumber == 4) {
			//h_bg_4 = Original.ptr<uchar3>(0);
			h_bg_4 = membrane_img_copy.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_4, h_bg_4, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if (frameNumber == 5) {
			//h_bg_5 = Original.ptr<uchar3>(0);
			h_bg_5 = membrane_img_copy.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_5, h_bg_5, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));

			// Construct the background via GPU
			construct_background(d_bg_1, d_bg_2, d_bg_3, d_bg_4, d_bg_5, d_bg, numCol, numRow);

			//// test
			//uchar3 *h_bg;
			//bg_img_cpu_temp.create(numRow, numCol, CV_8UC3);
			//h_bg = bg_img_cpu_temp.ptr<uchar3>(0);

			//// (test) (GPU) Copy the data from device to host
			//checkCudaErrors(cudaMemcpy(h_bg, d_bg, numPixels * sizeof(uchar3), cudaMemcpyDeviceToHost));

			//// (test) 
			//Mat bg_img_cpu(numRow, numCol, CV_8UC3, (void*)h_bg);
			//namedWindow("bg_img_cpu", WINDOW_NORMAL);
			//moveWindow("bg_img_cpu", 650, 5);
			//imshow("bg_img_cpu", bg_img_cpu);
		}

		// Start the background subtraction algorithm after constructing the background
		if (frameNumber > 5) {
			h_cur_img = membrane_img_copy.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_cur_img, h_cur_img, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));

			// (GPU) The background subtraction algorithm
			background_subtract_gpu(d_bg, d_cur_img, d_result, numCol, numRow);

			// test
			unsigned char *h_result;
			result_cpu_temp.create(numRow, numCol, CV_8UC1);
			h_result = result_cpu_temp.ptr<unsigned char>(0);

			// (test) (GPU) Copy the data from device to host
			checkCudaErrors(cudaMemcpy(h_result, d_result, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
			Mat bg_result(numRow, numCol, CV_8UC1, h_result);

			// Save specific images (Original frames (PNG))
			if(frameNumber==846 || frameNumber==1041){
				original_n = file_path + "original_frames/" + to_string(frameNumber) + ".png";
				imwrite(original_n, Original, compression_params);
			}

			// Save specific images (Needle detection (PNG))
			if(frameNumber==846){
				needle_n = file_path + "needle/" + to_string(frameNumber) + "_(GPU).png";
				imwrite(needle_n, bg_result, compression_params);
			}

			// Save specific images (Thread detection (PNG))
			if(frameNumber==1041){
				thread_n = file_path + "thread/" + to_string(frameNumber) + "_(GPU).png";
				imwrite(thread_n, bg_result, compression_params);
			}

			// test
			namedWindow("bg_result", WINDOW_NORMAL);
			moveWindow("bg_result", 650, 5);
			imshow("bg_result", bg_result);
		}

		// Display the original image
		namedWindow("Original image", WINDOW_NORMAL);
		moveWindow("Original image", 5, 5);
		imshow("Original image", membrane_img_copy);

		// For checking the frame rate
		current_time = chrono::high_resolution_clock::now();
		time_diff = chrono::duration_cast<chrono::milliseconds> (current_time - start_time).count();
		time_diff_float = float(time_diff) / 1000;
		fprintf(frame_rate_f, "%d, %f\n",frameNumber, time_diff_float);

		//// test 
		//cout << "The current frame: " << frameNumber << endl;

		// Update the frame number
		frameNumber = frameNumber + 1;

		// If the 'Esc' key is pressed, break the while loop.
		if (waitKey(1) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}

	}

	// (GPU) free any necessary memory.
	cudaFree(d_bg_1);
	cudaFree(d_bg_2);
	cudaFree(d_bg_3);
	cudaFree(d_bg_4);
	cudaFree(d_bg_5);
	cudaFree(d_bg);
	cudaFree(d_cur_img);
	cudaFree(d_result);

	fclose(frame_rate_f);
	destroyAllWindows();
	system("pause");
	return 0;

}