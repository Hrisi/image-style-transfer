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


void convertBGRtoLMS(Mat& bgrImg, Mat& lmsImg);


void convertLMStoLab(const Mat& lmsImg, Mat& labImg);


void convertLabtoLMS(const Mat& labImg, Mat& lmsImg);


void convertLMStoBGR(const Mat& lmsImg, Mat& bgrImg);


void convertBGRtoYUV(const Mat& bgrImg, Mat& yuvImg);


bool computeBGRGrayWhitePoint(const Mat& bgrImg, Mat& whitePoint,
                              Mat& prevYUVWhitePoint);


void XYZtoxy(const Mat& XYZImg, Mat& xyImg);


void xytoXYZ(const Mat& xyImg, float maxLuminance, Mat& XYZImg);


float product(const Mat& whitePoint, const Mat& invMatrix, int k);


void getRGBtoXYZMatrix(Mat& whitePoint, Mat& RGBtoXYZMatrix);


void convertBGRtoXYZ(Mat& bgrImg, Mat& whitePoint, Mat& xyzImg);


void convertXYZtoBGR(const Mat& xyzImg, Mat& XYZWhitePoint, Mat& bgrImg);


void adaptXYZToNewWhitePoint(Mat& XYZWhitePoint, const Mat& BGRWhitePoint);


void CAT(Mat& bgrImg, Mat& inpWhitePoint, const Mat& tarWhitePoint);


void convertBGRToLab(Mat& bgrImg, Mat& BGRWhitePoint, Mat& labImg);


void convertLabToBGR(const Mat& labImg, const Mat& BGRWhitePoint, Mat& bgrImg);


#endif
