#ifndef __monge_transform_h__

#define __monge_transform_h__

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


void computeHue(const Mat& img, Mat& hue, int chAInd, int chBInd);


void dotProduct(const Mat& matrix1, const Mat& matrix2, Mat& matrixProduct);


void matrixSquareRoot(const Mat& matrix, Mat& squareRoot);


float computeMean(const Mat& imgChannel);


float computeWeightedMean(const Mat& channelChannel);


void computeWeightedMeanVector(const Mat& img, float*& meanVector);


float computeCovarianceElement(const Mat& channelA, const Mat& channelB);


void computeCovariance(const Mat& img, Mat& imgCovMatrix);


void closedFormMatrix(const Mat& inpCovMatrix, const Mat& tarCovMatrix,
                      Mat& matrixT);


void computeMapping(const Mat& inpImg, const Mat& tarImg,
                    float*& inpMean, float*& tarMean,
                    const Mat& matrixT, Mat& newValues);


void applyTransform(const Mat& inpImg, const Mat& tarImg, Mat& newValues,
                    Mat& invCovMat, Mat& means);


#endif
