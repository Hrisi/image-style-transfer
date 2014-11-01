#ifndef __color_spaces_h__

#define __color_spaces_h__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;


float clipImg(float value, float lower, float upper);


float sRGBCompanding(float value);


float sRGBInverseCompanding(float value);


void conversion(const Mat& original, Mat& result, const Mat& matrix,
                int rows, int cols);


void BGRtoLMS(const Mat& bgrImg, Mat& lmsImg);


void LMStoLab(const Mat& lmsImg, Mat& labImg);


void LabtoLMS(const Mat& labImg, Mat& lmsImg);


void LMStoBGR(const Mat& lmsImg, Mat& bgrImg);


void BGRtoYUV(const Mat& bgrImg, Mat& yuvImg);


bool computeBGRWhitePoint(const Mat& bgrImg, Mat& whitePoint,
                          Mat& previuosWhitePoint);


void XYZtoxy(const Mat& XYZImg, Mat& xyImg);


void xytoXYZ(const Mat& xyImg, float maxLuminance, Mat& XYZImg);


void getRGBtoXYZMatrix(const Mat& whitePoint, Mat& RGBtoXYZMatrix);


void BGRtoXYZWhitePoint(Mat& bgrImg, const Mat& whitePoint, Mat& xyzImg);


void adaptXYZToNewWhitePoint(Mat& XYZWhitePoint, const Mat& BGRWhitePoint);


void CAT(Mat& bgrImg, Mat& inpWhitePoint, const Mat& tarWhitePoint);


#endif
