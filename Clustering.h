#ifndef __clustering_h__

#define __clustering_h__

#include <opencv2/opencv.hpp>

using namespace cv;


void getClusterMap(const Mat& img, Mat& clsMap, Mat& samples, EM& clusters,
                   int clusterNmb, int clsCnt);


double computeProbs(const Mat& sample, const Mat& mean, const Mat& invCovar);


void normalize(double*& probs, int size);


#endif
