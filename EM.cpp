#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <EM.h>

using namespace cv;

#define NEIGHBOURHOOD_SIZE 7
#define DELTA 0.1
#define SPATIAL_DEV 7
#define COLOR_DEV 7


EMTai::EMTai(int clsCnt){
  this->clsCnt = clsCnt;
  this->currentIter.normSmoothFactor = new double[this->clsCnt];
  this->previousIter.normSmoothFactor = new double[this->clsCnt];

  this->currentIter.normFactor = 0;
  this->previousIter.normFactor = 0;
}


EMTai::~EMTai(){
  delete [] this->currentIter.normSmoothFactor;
  delete [] this->previousIter.normSmoothFactor;
}


void EMTai::train(const Mat& samples, int& imgColInd){
  this->currentIter.means = Mat(this->clsCnt, samples.cols, CV_32F);
  this->currentIter.covs = Mat(this->clsCnt, samples.cols, CV_32F);
  this->currentIter.probs = Mat(samples.rows, this->clsCnt, CV_64F);
  this->currentIter.smoothProbs = Mat(samples.rows, this->clsCnt, CV_64F);

  this->previousIter.means = Mat(this->clsCnt, samples.cols, CV_32F);
  this->previousIter.covs = Mat(this->clsCnt, samples.cols, CV_32F);
  this->previousIter.probs = Mat(samples.rows, this->clsCnt, CV_64F);
  this->previousIter.smoothProbs = Mat(samples.rows, this->clsCnt, CV_64F);

  Mat labels = Mat(samples.rows, 1, CV_8U);
  //std::cout << samples << std::endl;
  kmeans(samples, this->clsCnt, labels,
         TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3,
         KMEANS_PP_CENTERS,
         this->currentIter.means);
  std::cout << this->currentIter.means << std::endl;

  for (int i = 0; i < this->clsCnt; i++){
    for (int j = 0; j < samples.cols; j++){
      this->currentIter.covs.at<float>(i, j) =
        this->initiateCovs(samples.col(j),
                           this->currentIter.means.at<float>(i, j));
    }
  }
  std::cout << this->currentIter.covs << std::endl;

  int step = 0;
  int converIndicator;

  while(true){
    std::cout << step << " new step" << std::endl;

    for (int i = 0; i < samples.rows; i++){
      this->currentIter.normFactor = 0;
      for (int k = 0; k < this->clsCnt; k++){
        this->currentIter.probs.at<double>(i, k) = this->reestimateProb(
            samples.row(i),
            this->currentIter.means.row(k),
            this->currentIter.covs.row(k), step);
      }
      this->normalizeProbs(i, step);
    }
    std::cout << step << " new step" << std::endl;

    this->smooth(samples, imgColInd);
    std::cout << step << " new step" << std::endl;

    for (int k = 0; k < this->clsCnt; k++){ 
      for (int j = 0; j < samples.cols; j++){
        this->currentIter.means.at<float>(k, j) = this->reestimateMean(
          this->currentIter.smoothProbs.col(k),
          samples.col(j));
        this->currentIter.covs.at<float>(k, j) = this->reestimateCov(
          this->currentIter.smoothProbs.col(k),
          samples.col(j),
          this->currentIter.means.at<float>(k, j));
        std::cout << this->currentIter.means << std::endl;
        std::cout << this->currentIter.covs << std::endl;
      }
    }
    converIndicator = 0;

    if (step > 3){
      std::cout << "break" << std::endl;
      break;
    }
    /*  for (int i = 0; i < samples.rows; i++){
        for (int k = 0; k < this->clsCnt; k++){
          if (abs(this->previousIter.means.at<float>(i, k) -
                  this->currentIter.means.at<float>(i, k)) < DELTA &&
              abs(this->previousIter.covs.at<float>(i, k) -
                  this->currentIter.covs.at<float>(i, k)) < DELTA){
            converIndicator++;
          }
        }
      }
      if (converIndicator == samples.rows * this->clsCnt){
        break;
      }
    }
    this->previousIter = this->currentIter; */
    step++;
  }
}


float EMTai::initiateCovs(const Mat& samples, float& mean){
  float sumMeans = 0.0;
  float cov;

  for (int i = 0; i < samples.rows; i++){
    sumMeans += (samples.at<float>(i, 0) - mean) *
                (samples.at<float>(i, 0) - mean);
  }

  cov = float(sumMeans) / (samples.rows - 1);

  return sqrt(cov);
}


double EMTai::reestimateProb(const Mat& sample, const Mat& mean, const Mat& cov,
                             int& step){
  double prob;
  double powTerm = 0.0;

  for (int i = 0; i < 3; i++){
    if (cov.at<float>(0, i) > 0){
      powTerm += (sample.at<float>(0, i) - mean.at<float>(0, i)) *
                 (sample.at<float>(0, i) - mean.at<float>(0, i)) /
                 (2 * cov.at<float>(0, i) * cov.at<float>(0, i));
    }
  }

  prob = exp(-powTerm);
  this->currentIter.normFactor += prob;

  return prob;
}


void EMTai::normalizeProbs(int& rowInd, int& step){
  for (int i = 0; i < this->clsCnt; i++){
    this->currentIter.probs.at<double>(rowInd, i) /=
      this->currentIter.normFactor;
    if (step){
      this->currentIter.probs.at<double>(rowInd, i) +=
        this->currentIter.smoothProbs.at<double>(rowInd, i);
    }
  }
}


float EMTai::reestimateMean(const Mat& probs, const Mat& samples){
  float mean = 0.0;
  double normFactor = 0;

  for (int i = 0; i < samples.rows; i++){
    mean += probs.at<double>(i, 0) * samples.at<float>(i, 0);
    normFactor += probs.at<double>(i, 0);
  }

  mean /= normFactor;

  return mean;
}


float EMTai::reestimateCov(const Mat& probs, const Mat& samples, float& mean){
  float cov = 0;
  double normFactor = 0;

  for (int i = 0; i < samples.rows; i++){
    cov += probs.at<double>(i, 0) * (samples.at<float>(i, 0) - mean) *
           (samples.at<float>(i, 0) - mean);
    normFactor += probs.at<double>(i, 0);
  }

  cov /= normFactor;
  cov = sqrt(cov);

  return cov;
}


void EMTai::smooth(const Mat& samples, int& imgColInd){
  double bilFilterRes;
  float firstComponent;
  float spatialGauss[NEIGHBOURHOOD_SIZE][NEIGHBOURHOOD_SIZE];
  
  for (int k = 0; k < this->clsCnt; k++){
    this->currentIter.normSmoothFactor[k] = 0;
  }

  // initialize first component of BilFilter
  for (int i = 0; i < NEIGHBOURHOOD_SIZE; i++){
    for (int j = 0; j < NEIGHBOURHOOD_SIZE; j++){
      spatialGauss[i][j] = exp(-(i * i + j * j) / SPATIAL_DEV);
    }
  }

  for (int k = 0; k < this->clsCnt; k++){
    for (int i = 0; i < samples.rows; i++){
      for (int j = -NEIGHBOURHOOD_SIZE / 2; j <= NEIGHBOURHOOD_SIZE / 2; j++){
        if ((i / imgColInd + j) > 0 && (i / imgColInd + j) < samples.rows / imgColInd){
          for (int h = -NEIGHBOURHOOD_SIZE / 2; h <= NEIGHBOURHOOD_SIZE / 2; h++){
            if ((i % imgColInd + h) > 0 &&
                (i % imgColInd + h) < imgColInd){
             if ( i + j * imgColInd + h - 1 >= samples.rows)
               std::cout << i / imgColInd + j << " " << i% imgColInd + h << std::endl;
              firstComponent = 
                spatialGauss[abs(h + NEIGHBOURHOOD_SIZE / 2)]
                            [abs(j + NEIGHBOURHOOD_SIZE / 2)];
              bilFilterRes = this->bilateralFilter(
                firstComponent,
                samples.row(i),
                samples.row(i + j * imgColInd + h));
              this->currentIter.smoothProbs.at<double>(i, k) += bilFilterRes *
                this->currentIter.probs.at<double>(i + j * imgColInd + h, k);
              this->currentIter.normSmoothFactor[k] +=
                this->currentIter.smoothProbs.at<double>(i, k);
            }
          }
        }
      }
    }
  }
  for (int k = 0; k < this->clsCnt; k++){
    for (int i = 0; i < samples.rows; i++){
      this->currentIter.smoothProbs.at<double>(i, k) /=
        this->currentIter.normSmoothFactor[k];
    }
  }
}


double EMTai::bilateralFilter(float& spatialGauss, const Mat& centerI,
                              const Mat& neighI){
  double colorGauss = exp(-((centerI.at<float>(0, 0) - neighI.at<float>(0, 0)) *
                            (centerI.at<float>(0, 0) - neighI.at<float>(0, 0)) +
                            (centerI.at<float>(0, 1) - neighI.at<float>(0, 1)) *
                            (centerI.at<float>(0, 1) - neighI.at<float>(0, 1)) +
                            (centerI.at<float>(0, 2) - neighI.at<float>(0, 2)) *
                            (centerI.at<float>(0, 2) - neighI.at<float>(0, 2))) /
                          COLOR_DEV);

  return spatialGauss * colorGauss;
}


Mat EMTai::getMeans(){
  return this->currentIter.means;
}


Mat EMTai::getCovs(){
  return this->currentIter.covs;
}


float EMTai::getProbs(int& x, int& y, int& clsCnt, int& imgColInd){
  return this->currentIter.probs.at<float>(x * imgColInd + y, clsCnt);
}
