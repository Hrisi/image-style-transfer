#ifndef __img_feature_detection_h__

#define __img_feature_detection_h__

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;


void showHistogram(MatND hist, int bins);


void calcHist1D(Mat& array, MatND& hist, int channels, int bins);


bool isPeak(std::vector<int>& peaks, int currentPeak, MatND hist, int minSize,
            float minPeakDist, int bins);


void findHistPeaks(MatND hist, std::vector<int>& peaks, int histBins,
                   int minSize, float minPeakDist);


#endif
