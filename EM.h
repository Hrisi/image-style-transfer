#ifndef __em_h__

#define __em_h__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;


struct EMIterationState{
  double normFactor;
  double* normSmoothFactor;
  Mat means;
  Mat covs;
  Mat probs;
  Mat smoothProbs;
};


class EMTai{
  private:
    int clsCnt;
    EMIterationState currentIter;
    EMIterationState previousIter;
    double reestimateProb(const Mat& sample, const Mat& mean, const Mat& cov,
                          int& step);
    void normalizeProbs(int& rowInd, int& step);
    void smooth(const Mat& samples, int& imgColInd);
    float reestimateMean(const Mat& probs, const Mat& samples);
    float reestimateCov(const Mat& probs, const Mat& samples, float& mean);
    float initiateCovs(const Mat& samples, float& mean);
    double bilateralFilter(float& spatialGauss, const Mat& centerI,
                           const Mat& neighI);
  public:
    EMTai(int clsCnt);
    ~EMTai();
    void train(const Mat& samples, int& imgColInd);
    Mat getMeans();
    Mat getCovs();
    float getProbs(int& x, int& y, int& clsCnt, int& imgColInd);
};


#endif
