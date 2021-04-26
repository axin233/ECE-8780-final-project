#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include "background_subtraction.h"

#define PI 3.14159265

using namespace std;
using namespace cv;

int main(int argc, char* argv[])//
{
	int frameNumber = 1;
	Rect bounding_rect, prev_bounding_rect;
	Mat Original, Original_hsv, mask_for_circle, mask_for_circle_copy,membrane_img, membrane_img_copy;
	Mat crop_img_temp, crop_img, crop_bg_img_temp, crop_bg_img, bg_img_cpu, bg_img_cpu_tri, bg_img_cpu_tri_copy;
	Mat threshold_thread, threshold_thread_temp, threshold_needle, result_gpu_temp, crop_img_hsv;
	bool is_first_frame = true;
	Point membrane_center,circu_coord,last_circu_coord;
	int membrane_radius, numPixels, numPixels_tri;
	chrono::high_resolution_clock::time_point start_time, current_time;
	int64_t time_diff;
	float time_diff_float;
	FILE* frame_rate_f=NULL;
	string frame_rate_n, needle_n, thread_n, original_n;
	string file_path = "D:/visual_studio_code/HPC_GPU/results/";
	string file_path_needle = file_path + "needle/";
	string processed_result_n = file_path + "Subject_21_1_processed.mov";
	Point pts[1][3];
	const Point* ppt[1];
	int curSeg=12, countP=0;
	uchar3 *h_bg_1, *h_bg_2, *h_bg_3, *h_bg_4, *h_bg_5, *h_bg_crop;
	uchar3 *d_bg_1, *d_bg_2, *d_bg_3, *d_bg_4, *d_bg_5, *d_bg, *d_bg_crop;
	uchar3 *h_cur_img_crop, *d_cur_img_crop, *d_crop_img_hsv, *h_crop_img_hsv;
	unsigned char *d_result;

	// For recording detection result
	Mat blackBoard(480, 960, CV_8UC3, 0.0);
	Mat combination;
	VideoWriter processed_video(processed_result_n, CV_FOURCC('D', 'I', 'V', 'X'), 70, Size(960, 480));

	// Initialization
	prev_bounding_rect.x = 0;
	prev_bounding_rect.y = 0;
	prev_bounding_rect.width = 0;
	prev_bounding_rect.height = 0;

	// The membrane center
	membrane_center.x = 321;
	membrane_center.y = 226;

	// The membrane radius
	membrane_radius = 210;

	// File for checking the frame rate
	start_time = chrono::high_resolution_clock::now();
	frame_rate_n = file_path + "frame_rate/frame_rate(CPU).txt";
	frame_rate_f = fopen(frame_rate_n.c_str(), "wb");

	// Check if the text file is opened successfully
	if(frame_rate_f==NULL){
		cout<<"error\n";
		exit(1);
	}

	// Compression ratio (for saving png images)
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	// Video directory
	VideoCapture cap("D:/videos_from_the_internal_realSense/using_12v_adaptor/Subject_21_1_Internal_Camera.avi");

	// Check if the video is available or not
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video file" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	// Process video frames one by one
	while (1)
	{
		// Extract a single frame from video
		bool bSuccess = cap.read(Original);

		// Exit the program when reaching the end of the video
		if (bSuccess == false)
		{
			cout << "Found the end of the video" << endl;
			break;
		}

		// Initialization
		if (is_first_frame == true) {
			
			//
			mask_for_circle.create(Original.rows, Original.cols, CV_8UC1);
			mask_for_circle.setTo(Scalar(0, 0, 0));

			// (GPU) Allocate memory on GPU
			numPixels = (Original.rows)*(Original.cols);
			checkCudaErrors(cudaMalloc((void**)&d_bg_1, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_2, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_3, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_4, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_5, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg, sizeof(uchar3)*numPixels));
			checkCudaErrors(cudaMalloc((void**)&d_bg_crop, sizeof(uchar3)*1));
			checkCudaErrors(cudaMalloc((void**)&d_cur_img_crop, sizeof(uchar3)*1));
			checkCudaErrors(cudaMalloc((void**)&d_crop_img_hsv, sizeof(uchar3) * 1));
			checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(unsigned char)*1));

			//
			is_first_frame = false;
		}

		// (GPU) Construct the background
		if (frameNumber == 1) {
			h_bg_1 = Original.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_1, h_bg_1, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if (frameNumber == 2) {
			h_bg_2 = Original.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_2, h_bg_2, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if (frameNumber == 3) {
			h_bg_3 = Original.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_3, h_bg_3, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if (frameNumber == 4) {
			h_bg_4 = Original.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_4, h_bg_4, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));
		}
		else if (frameNumber == 5) {
			h_bg_5 = Original.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_5, h_bg_5, sizeof(uchar3)*numPixels, cudaMemcpyHostToDevice));

			// (GPU) Construct the background via GPU
			construct_background(d_bg_1, d_bg_2, d_bg_3, d_bg_4, d_bg_5, d_bg, (Original.cols), (Original.rows));

			// (GPU) Copy the data from device to host
			uchar3 *h_bg;
			Mat bg_img_cpu_temp;
			bg_img_cpu_temp.create((Original.rows), (Original.cols), CV_8UC3);
			h_bg = bg_img_cpu_temp.ptr<uchar3>(0);			
			checkCudaErrors(cudaMemcpy(h_bg, d_bg, numPixels * sizeof(uchar3), cudaMemcpyDeviceToHost));

			// Convert data to OpenCV mat 
			Mat bg_img_cpu_final((Original.rows), (Original.cols), CV_8UC3, (void*)h_bg);
			bg_img_cpu = bg_img_cpu_final.clone();
		}

		// Start the background subtraction algorithm after constructing the background
		if (frameNumber > 5) {

			// Reset pixel values
			mask_for_circle.setTo(Scalar(0, 0, 0));
			membrane_img.setTo(Scalar(0, 0, 0));
			bg_img_cpu_tri.setTo(Scalar(0, 0, 0));
			blackBoard.setTo(Scalar(0, 0, 0));

			// Calculate the two end points of each triangle. 
			// The third end point of the triangle is the membrane center
			last_circu_coord.x = int(membrane_radius*cos(30.0*float(curSeg - 1)*PI / 180.0) + membrane_center.x);
			last_circu_coord.y = int(membrane_radius*sin(30.0*float(curSeg - 1)*PI / 180.0) + membrane_center.y);
			circu_coord.x = int(membrane_radius*cos(30.0*float(curSeg)*PI / 180.0) + membrane_center.x);
			circu_coord.y = int(membrane_radius*sin(30.0*float(curSeg)*PI / 180.0) + membrane_center.y);
			pts[0][0] = circu_coord;
			pts[0][1] = last_circu_coord;
			pts[0][2] = membrane_center;

			// Create a triangular mask
			ppt[0] = { pts[0] };
			int npt[] = { 3 };
			fillPoly(mask_for_circle, ppt, npt, 1, Scalar(255, 255, 255), 8);

			// Obtain contents within the triangular mask
			Original.copyTo(membrane_img, mask_for_circle);
			membrane_img_copy = membrane_img.clone();

			// Calculate the bounding rectangle for the triangular area
			bounding_rect = boundingRect(mask_for_circle);

			// Crop the triangular area from the video frame
			crop_img_temp = membrane_img_copy(bounding_rect);
			crop_img = crop_img_temp.clone();

			// Convert the cropped image from BGR to HSV
			cvtColor(crop_img, crop_img_hsv, COLOR_BGR2HSV, 3);

			// Obtain background within the triangular mask
			bg_img_cpu.copyTo(bg_img_cpu_tri, mask_for_circle);
			bg_img_cpu_tri_copy = bg_img_cpu_tri.clone();

			// Crop the triangular area from the background
			crop_bg_img_temp = bg_img_cpu_tri_copy(bounding_rect);
			crop_bg_img = crop_bg_img_temp.clone();

			// (GPU) Re-allocate memory after changing the triangular area
			if ((bounding_rect.width != prev_bounding_rect.width) || (bounding_rect.height != prev_bounding_rect.height)) {

				// 
				numPixels_tri = (bounding_rect.width) * (bounding_rect.height);
				cudaFree(d_bg_crop);
				cudaFree(d_cur_img_crop);
				cudaFree(d_crop_img_hsv);
				cudaFree(d_result);
				checkCudaErrors(cudaMalloc((void**)&d_bg_crop, sizeof(uchar3) * numPixels_tri));
				checkCudaErrors(cudaMalloc((void**)&d_cur_img_crop, sizeof(uchar3) * numPixels_tri));
				checkCudaErrors(cudaMalloc((void**)&d_crop_img_hsv, sizeof(uchar3) * numPixels_tri));
				checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(unsigned char) * numPixels_tri));

				// Update parameters
				prev_bounding_rect = bounding_rect;				
			}

			// (GPU) Upload data to GPU
			h_cur_img_crop = crop_img.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_cur_img_crop, h_cur_img_crop, sizeof(uchar3)*numPixels_tri, cudaMemcpyHostToDevice));
			h_bg_crop = crop_bg_img.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_bg_crop, h_bg_crop, sizeof(uchar3)*numPixels_tri, cudaMemcpyHostToDevice));
			h_crop_img_hsv=crop_img_hsv.ptr<uchar3>(0);
			checkCudaErrors(cudaMemcpy(d_crop_img_hsv, h_crop_img_hsv, sizeof(uchar3)*numPixels_tri, cudaMemcpyHostToDevice));

			// (GPU) The background subtraction algorithm
			background_subtract_gpu(d_bg_crop, d_cur_img_crop, d_crop_img_hsv, d_result, (crop_img.cols), (crop_img.rows));

			// (GPU) Copy the data from device to host
			unsigned char *h_result;
			result_gpu_temp.create((crop_img.rows), (crop_img.cols), CV_8UC1);
			h_result = result_gpu_temp.ptr<unsigned char>(0);
			checkCudaErrors(cudaMemcpy(h_result, d_result, numPixels_tri * sizeof(unsigned char), cudaMemcpyDeviceToHost));
			Mat bg_result((crop_img.rows), (crop_img.cols), CV_8UC1, h_result);

			// Obtain the needle detection result
			threshold(bg_result, threshold_needle, 150, 255, THRESH_BINARY);

			// Obtain the thread detection result
			threshold(bg_result, threshold_thread_temp, 150, 255, THRESH_TOZERO_INV);
			threshold(threshold_thread_temp, threshold_thread, 100, 255, THRESH_BINARY);

			// (Visualize result) Draw the triangular area
			line(Original, circu_coord, membrane_center, Scalar(255, 0, 0), 1, 8);
			line(Original, last_circu_coord, membrane_center, Scalar(255, 0, 0), 1, 8);
			line(Original, circu_coord, last_circu_coord, Scalar(255, 0, 0), 1, 8);

			// (Visualize result) Combine the needle detection result and the thread detection result
			Mat channels[3] = {threshold_thread, threshold_needle, threshold_thread};
			merge(channels, 3, combination);

			// Save the detection result
			Original.copyTo(blackBoard.rowRange(0, Original.rows).colRange(0, Original.cols));
			combination.copyTo(blackBoard.rowRange(0, threshold_thread.rows).colRange(Original.cols, (Original.cols + threshold_thread.cols)));
			processed_video.write(blackBoard);

			// Check the frame rate
			current_time = chrono::high_resolution_clock::now();
			time_diff = chrono::duration_cast<chrono::milliseconds> (current_time - start_time).count();
			time_diff_float = float(time_diff) / 1000;
			fprintf(frame_rate_f, "%d, %f\n", frameNumber, time_diff_float);

			//count detected pixels, if none then move the mask
			countP = countNonZero(bg_result);
			if (countP == 0) {
				curSeg = curSeg % 12 + 1;
			}

			// Display the original image
			namedWindow("Original image", WINDOW_NORMAL);
			moveWindow("Original image", 5, 5);
			imshow("Original image", blackBoard);

			// Display the needle detection result
			namedWindow("threshold_needle", WINDOW_NORMAL);
			moveWindow("threshold_needle", 650, 5);
			imshow("threshold_needle", threshold_needle);

			// Display the thread detection result
			namedWindow("threshold_thread", WINDOW_NORMAL);
			moveWindow("threshold_thread", 1250, 5);
			imshow("threshold_thread", threshold_thread);
		}

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
	cudaFree(d_bg_crop);
	cudaFree(d_crop_img_hsv);
	cudaFree(d_cur_img_crop);
	cudaFree(d_result);

	processed_video.release();
	fclose(frame_rate_f);
	system("pause");
	return 0;
}