#define _CRT_SECURE_NO_WARNINGS
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

#define PI 3.14159265

using namespace std;
using namespace cv;

int main(int argc, char* argv[])//
{
	int frameNumber = 1;
	Mat Original, Original_hsv, mask_for_circle, mask_for_circle_copy,membrane_img, membrane_img_copy;
	Mat threshold_thread, threshold_needle;
	bool is_first_frame = true;
	Point membrane_center,circu_coord,last_circu_coord;
	int membrane_radius;
	chrono::high_resolution_clock::time_point start_time, current_time;
	int64_t time_diff;
	float time_diff_float;
	FILE* frame_rate_f;
	string frame_rate_n, needle_n, thread_n, original_n;
	string file_path = "frameImage/";
	string file_path_needle = "frameImageNeedle/";
	Point pts[1][3];
	const Point* ppt[1];

	// The membrane center
	membrane_center.x = 321;
	membrane_center.y = 226;

	// The membrane radius
	membrane_radius = 210;

	// Check the frame rate
	start_time = chrono::high_resolution_clock::now();
	frame_rate_n = "rate/frame_rate(CPU).txt";
	frame_rate_f = fopen(frame_rate_n.c_str(), "wb");

	if(frame_rate_f==NULL){
		cout<<"error\n";
		exit(1);
	}

	// Compression ratio (for saving png images)
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	// Video directory
	VideoCapture cap("Subject_21_1_Internal_Camera.avi");


	// Check if the video is available or not
	if (cap.isOpened() == false)
	{
		cout << "Cannot open the video file" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	cout<<"about to processing"<<endl;

	// Process video frames one by one
	while (1)
	{
		// Extract a single frame from video
		bool bSuccess = cap.read(Original);

		//
		if (bSuccess == false)
		{
			cout << "Found the end of the video" << endl;
			break;
		}

		// Initialization
		if (is_first_frame == true) {
			mask_for_circle.create(Original.rows, Original.cols, CV_8UC1);
			mask_for_circle.setTo(Scalar(0, 0, 0));
			is_first_frame = false;
		}

		mask_for_circle_copy=mask_for_circle.clone();

		// Remove the peripheral area
		//circle(mask_for_circle_copy, membrane_center, membrane_radius, Scalar(0,0,0), -1);
		
		last_circu_coord.x=membrane_radius*cos(0)+membrane_center.x;
		last_circu_coord.y=membrane_radius*sin(0)+membrane_center.y;
		
		//Doing segmentation at position 5 only
		//i=1 i<13 for a full cycle
		//first segment 5
		for(float i=1;i<13;i++){
			circu_coord.x=membrane_radius*cos(30.0*i*PI/180.0)+membrane_center.x;
			circu_coord.y=membrane_radius*sin(30.0*i*PI/180.0)+membrane_center.y;

			pts[0][0]=circu_coord;
			pts[0][1]=last_circu_coord;
			pts[0][2]=membrane_center;

			ppt[0]={pts[0]};
			int npt[]={3};

			if(i==5){
//			if(i==12){
				fillPoly(mask_for_circle_copy,ppt,npt,1,Scalar(255,255,255),8);
				//fillConvexPoly(mask_for_circle, testPts, Scalar(255,255,255));
			}

			last_circu_coord.x=circu_coord.x;
			last_circu_coord.y=circu_coord.y;
		}

		Original.copyTo(membrane_img, mask_for_circle_copy);
		membrane_img_copy = membrane_img.clone();

		//testing purposes
		if ( frameNumber <= 40) {
			
			// Original frames (PNG)
			//original_n = file_path + to_string(frameNumber) + "frame.png";
			//imwrite(original_n, Original);

			// Needle detection (PNG)
			needle_n = "example/" + to_string(frameNumber) + "FF.png";
            imwrite(needle_n, membrane_img_copy, compression_params);

			// Thread detection (PNG)
			//thread_n = file_path + to_string(frameNumber) + "thread.png";
			//imwrite(thread_n, threshold_thread, compression_params);

		}

		// Convert the image from BGR to HSV
		cvtColor(membrane_img_copy, Original_hsv, COLOR_BGR2HSV, 3);

		// Threshold values for needle
		inRange(Original_hsv, Scalar(0, 0, 100), Scalar(55, 50, 155), threshold_needle);

		// Threshold values for thread
		inRange(Original_hsv, Scalar(90, 22, 150), Scalar(130, 50, 200), threshold_thread);


		// Save video frames whose frame number are between 500 to 1200 
		if ( frameNumber>=500&&frameNumber <= 1200) {
			
			// Original frames (PNG)
			//original_n = file_path + to_string(frameNumber) + "frame.png";
			//imwrite(original_n, Original);

			// Needle detection (PNG)
			needle_n = file_path_needle + to_string(frameNumber) + "needle.png";
			imwrite(needle_n, threshold_needle, compression_params);

			// Thread detection (PNG)
			//thread_n = file_path + to_string(frameNumber) + "thread.png";
			//imwrite(thread_n, threshold_thread, compression_params);

		}


		// Check the frame rate
		current_time = chrono::high_resolution_clock::now();
		time_diff = chrono::duration_cast<chrono::milliseconds> (current_time - start_time).count();
		time_diff_float = float(time_diff) / 1000;
		fprintf(frame_rate_f, "%d, %f\n",frameNumber, time_diff_float);


		// Update the frame number
		frameNumber = frameNumber + 1;

		// Exit the loop after processing frame 1200
		if (frameNumber > 1200) {
			cout << "Programe exits successfully." << endl;
			break;
		}

	}

	fclose(frame_rate_f);
	system("pause");
	return 0;

}