#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char* argv[])//
{
	int frameNumber = 1;
	//char buffer2[80];
	Mat Original, Original_hsv, mask_for_circle, membrane_img, membrane_img_copy;
	Mat threshold_thread, threshold_needle, background_total, background_img;
	Mat one_mat, bg_blue, bg_green, bg_red, bg_mask_add, subtraction_result;
	Mat mul_blue, mul_green, mul_red;
	Mat add_result_temp, add_result, bg_sub_result_float, bg_sub_result;
	Mat split_bg[3], split_sub_result[3];
	vector<Mat> bg_arr;
	bool is_first_frame = true;
	Point membrane_center;
	int membrane_radius;
	chrono::high_resolution_clock::time_point start_time, current_time;
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

	// Check the frame rate
	start_time = chrono::high_resolution_clock::now();
	frame_rate_n = file_path + "frame_rate/frame_rate(CPU).txt";
	frame_rate_f = fopen(frame_rate_n.c_str(), "w");

	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	VideoCapture cap("D:/videos_from_the_internal_realSense/using_12v_adaptor/Subject_21_1_Internal_Camera.avi");

	// 
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video file" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	while (1)
	{
		//
		bool bSuccess = cap.read(Original);

		//
		if (bSuccess == false)
		{
			cout << "Found the end of the video" << endl;
			break;
		}

		// Initialization
		if (is_first_frame == true) {
			background_total.create(Original.rows, Original.cols, CV_32FC3);
			background_total.setTo(Scalar(0, 0, 0));
			mask_for_circle.create(Original.rows, Original.cols, CV_8UC1);
			mask_for_circle.setTo(Scalar(0, 0, 0));
			bg_mask_add = Mat::ones(Original.rows, Original.cols, CV_8UC1);

			circle(mask_for_circle, membrane_center, membrane_radius, Scalar(255, 255, 255), -1, 8, 0);

			is_first_frame = false;
		}

		// Remove the peripheral area
		Original.copyTo(membrane_img, mask_for_circle);
		membrane_img_copy = membrane_img.clone();

		//test
		//imshow("membrane_img_copy", membrane_img_copy);

		// Construct the background by the use of Frame 16 to Frame 20
		if ((frameNumber > 15) && (frameNumber <= 20)) {

			// Accumulate pixel values
			add(background_total, membrane_img_copy, background_total, bg_mask_add, CV_32FC3);

			// Calculate the average of pixel values
			if (frameNumber == 20) {

				// Create a one matrix
				one_mat = Mat::ones(Original.rows, Original.cols, CV_8UC1);

				// Split 'background_total' into 3 channel
				split(background_total, split_bg);

				// Divide each pixel values by 5 
				multiply(split_bg[0], one_mat, bg_blue, 0.2, CV_8UC1);
				multiply(split_bg[1], one_mat, bg_green, 0.2, CV_8UC1);
				multiply(split_bg[2], one_mat, bg_red, 0.2, CV_8UC1);

				// Collect the modified channels
				bg_arr.push_back(bg_blue);
				bg_arr.push_back(bg_green);
				bg_arr.push_back(bg_red);
				merge(bg_arr, background_img);

				//// test
				//namedWindow("The background image", WINDOW_AUTOSIZE);
				//moveWindow("The background image", 255, 5);
				//imshow("The background image", background_img);
			}
		}

		// Start the background subtraction algorithm
		if (frameNumber > 20) {

			// Subtract the background image from the current image
			subtract(membrane_img_copy, background_img, subtraction_result, mask_for_circle, CV_32FC3);

			// Split the current image into 3 one-channel images
			split(subtraction_result, split_sub_result);

			// Element-wise multiplication
			multiply(split_sub_result[0], split_sub_result[0], mul_blue, 1, CV_32FC1);
			multiply(split_sub_result[1], split_sub_result[1], mul_green, 1, CV_32FC1);
			multiply(split_sub_result[2], split_sub_result[2], mul_red, 1, CV_32FC1);

			// Element-wise addition 
			add(mul_blue, mul_green, add_result_temp, mask_for_circle, CV_32FC1);
			add(add_result_temp, mul_red, add_result, mask_for_circle, CV_32FC1);

			// Image thresholding
			threshold(add_result, bg_sub_result_float, 1100, 255, THRESH_BINARY);

			// test
			bg_sub_result_float.convertTo(bg_sub_result, CV_8UC1);
			namedWindow("Background subtration result", WINDOW_AUTOSIZE);
			moveWindow("Background subtration result", 450, 5);
			imshow("Background subtration result", bg_sub_result);
		}

		// Save specific images (Original frames (PNG))
		if(frameNumber==846 || frameNumber==1041){
			original_n = file_path + "original_frames/" + to_string(frameNumber) + ".png";
			imwrite(original_n, Original, compression_params);
		}

		// Needle detection (PNG)
		if(frameNumber==846){
			needle_n = file_path + "needle/" + to_string(frameNumber) + ".png";
			imwrite(needle_n, bg_sub_result, compression_params);
		}

		// Thread detection (PNG)
		if(frameNumber==1041){
			thread_n = file_path + "thread/" + to_string(frameNumber) + ".png";
			imwrite(thread_n, bg_sub_result, compression_params);
		}

		// Display the original image
		namedWindow("Original image", WINDOW_AUTOSIZE);
		moveWindow("Original image", 5, 5);
		imshow("Original image", Original);

		//// Display the detection result
		//namedWindow("Thresholded image", WINDOW_NORMAL);
		//moveWindow("Thresholded image", 650, 5);
		//imshow("Thresholded image", threshold_needle);

		// Check the frame rate
		current_time = chrono::high_resolution_clock::now();
		time_diff = chrono::duration_cast<chrono::milliseconds> (current_time - start_time).count();
		time_diff_float = float(time_diff) / 1000;
		fprintf(frame_rate_f, "%d, %f\n",frameNumber, time_diff_float);

		// 
		cout << "The current frame: " << frameNumber << endl;
		frameNumber = frameNumber + 1;

		// If the 'Esc' key is pressed, break the while loop.
		if (waitKey(1) == 27)
		{
			cout << "Esc key is pressed by user. Stoppig the video" << endl;
			break;
		}

	}

	fclose(frame_rate_f);
	destroyAllWindows();
	system("pause");
	return 0;

}