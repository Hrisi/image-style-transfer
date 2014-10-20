#ifndef __color_spaces_h__

#define __color_spaces_h__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;


void conversion(const Mat& original, Mat& result, const Mat& matrix,
                int rows, int cols);


void BGRtoLMS(const Mat& bgrImg, Mat& lmsImg);


void LMStoLab(const Mat& lmsImg, Mat& labImg);


void LabtoLMS(const Mat& labImg, Mat& lmsImg);


void LMStoBGR(const Mat& lmsImg, Mat& bgrImg);


#endif
