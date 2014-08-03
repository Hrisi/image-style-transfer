#ifndef __histogram_matching_h__

#define __histogram_matching_h__


#include<iostream>

#include<opencv2/opencv.hpp>
#include<numeric>

using namespace cv;


double* computeCDF(Mat img, int channelInd){
  double* cdf = new double[256];
  double freqCounter[256];
  int size = img.rows * img.cols;

  for (int i = 0; i < 256; i++)
    freqCounter[i] = 0;

  for (int i = 0; i < img.rows; i++)
    for (int j = 0; j < img.cols; j++)
      freqCounter[int(img.at<Vec3b>(i, j)[channelInd])]++;

  // get the probabilities
  for (int i = 0; i < 256; i++)
    freqCounter[i] /= double(size);

  std::partial_sum(freqCounter, freqCounter + 256, cdf);

  return cdf;
};


// matching only for the channelIndth channel;
void performMatching(Mat& inpImg, Mat tarImg, int channelInd){
  double* inpCDF = computeCDF(inpImg, channelInd);
  double* tarCDF = computeCDF(tarImg, channelInd);

  double* matchArr = new double[256];
  int currIndex = 0;

  for (int i = 0; i < 256; i++){
    while (tarCDF[currIndex] < inpCDF[i])
      currIndex++;
    if (tarCDF[currIndex] - inpCDF[i] < inpCDF[i] - tarCDF[currIndex - 1])
      matchArr[i] = currIndex;
    else matchArr[i] = currIndex - 1;
  };

  for (int i = 0; i < inpImg.rows; i++)
    for (int j = 0; j < inpImg.cols; j++)
      inpImg.at<Vec3b>(i, j)[channelInd] =
        matchArr[int(inpImg.at<Vec3b>(i, j)[channelInd])];
};


#endif
