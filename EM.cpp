#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <EM.h>

using namespace cv;

#define NEIGHBOUR_SIZE 7
#define DELTA pow(10, -6)
#define SPATIAL_DEV 1.5
#define COLOR_DEV 2


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


void EMTai::train(const Mat& samples, int& cols){
  this->currentIter.means = Mat(this->clsCnt, samples.cols, CV_32F);
  this->currentIter.covs = Mat(this->clsCnt, samples.cols, CV_32F);
  this->currentIter.probs = Mat(samples.rows, this->clsCnt, CV_64F);
  this->currentIter.smoothProbs = Mat(samples.rows, this->clsCnt, CV_64F);

  this->previousIter.means = Mat(this->clsCnt, samples.cols, CV_32F);
  this->previousIter.covs = Mat(this->clsCnt, samples.cols, CV_32F);
  this->previousIter.probs = Mat(samples.rows, this->clsCnt, CV_64F);
  this->previousIter.smoothProbs = Mat(samples.rows, this->clsCnt, CV_64F);

  Mat labels = Mat(samples.rows, 1, CV_8U);
  kmeans(samples, this->clsCnt, labels,
         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 300, 0.1), 5,
         KMEANS_PP_CENTERS,
         this->currentIter.means);

  for (int i = 0; i < this->clsCnt; i++){
    for (int j = 0; j < samples.cols; j++){
      float cov = this->initiateCovs(samples.col(j),
                                     this->currentIter.means.at<float>(i, j));
      this->currentIter.covs.at<float>(i, j) = cov;
    }
  }

  int step = 0;

  while(true){
    for (int i = 0; i < samples.rows; i++){
      this->currentIter.normFactor = 0;
      for (int k = 0; k < this->clsCnt; k++){
        this->currentIter.probs.at<double>(i, k) = this->reestimateProb(
            samples.row(i),
            this->currentIter.means.row(k),
            this->currentIter.covs.row(k));
      }
      this->normalizeProbs(i, step);
    }
    this->smooth(samples, cols);

    for (int k = 0; k < this->clsCnt; k++){ 
      for (int j = 0; j < samples.cols; j++){
        float mean = this->reestimateMean(this->currentIter.smoothProbs.col(k),
                                          samples.col(j));
        float cov = this->reestimateCov(this->currentIter.smoothProbs.col(k),
                                        samples.col(j),
                                        this->currentIter.means.at<float>(k, j)); 
        this->currentIter.means.at<float>(k, j) = mean;
        this->currentIter.covs.at<float>(k, j) = cov;
      }
    }
    if (isConverged(samples) || step > 20){
      break;
    }
    this->previousIter = this->currentIter;

    step++;
    std::cout << step << std::endl;
  }
}


bool EMTai::isConverged(const Mat& samples){
    int cnt = 0;

    for (int i = 0; i < samples.cols; i++){
      for (int k = 0; k < this->clsCnt; k++){
        float meansDiff = sqrt(pow(this->previousIter.means.at<float>(k, i) -
                                   this->currentIter.means.at<float>(k, i), 2));
        float covsDIff = sqrt(pow(this->previousIter.covs.at<float>(k, i) -
                                  this->currentIter.covs.at<float>(k, i), 2));
        if (meansDiff < DELTA && covsDIff < DELTA){
          cnt++;
        }
      }
    }
    if (cnt == samples.cols * this->clsCnt){
      return true;
    }

    return false;
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


double EMTai::reestimateProb(const Mat& sample, const Mat& mean, const Mat& cov){
  double prob;
  double powTerm = 0.0;

  for (int i = 0; i < 3; i++){
    float nominator = pow(sample.at<float>(0, i) - mean.at<float>(0, i), 2);
    float denominator = 2 * pow(cov.at<float>(0, i), 2);
    powTerm += nominator / denominator;
  }

  prob = exp(-powTerm);
  this->currentIter.normFactor += prob;

  return prob;
}


void EMTai::normalizeProbs(int& rowInd, int& step){
  for (int i = 0; i < this->clsCnt; i++){
    this->currentIter.probs.at<double>(rowInd, i) /= this->currentIter.normFactor;
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
    cov += probs.at<double>(i, 0) * pow(samples.at<float>(i, 0) - mean, 2);
    normFactor += probs.at<double>(i, 0);
  }

  cov /= normFactor;
  cov = sqrt(cov);

  return cov;
}


void EMTai::smooth(const Mat& samples, int& cols){
  double bilFilterRes;
  float firstComponent;
  float spatialGauss[NEIGHBOUR_SIZE][NEIGHBOUR_SIZE];
  
  for (int k = 0; k < this->clsCnt; k++){
    this->currentIter.normSmoothFactor[k] = 0;
  }

  // initialize first component of BilFilter
  for (int i = 0; i < NEIGHBOUR_SIZE; i++){
    for (int j = 0; j < NEIGHBOUR_SIZE; j++){
      spatialGauss[i][j] = exp(-(i * i + j * j) / SPATIAL_DEV);
    }
  }

  for (int k = 0; k < this->clsCnt; k++){
    for (int i = 0; i < samples.rows; i++){
      for (int j = -NEIGHBOUR_SIZE / 2; j <= NEIGHBOUR_SIZE / 2; j++){
        int neigbourRow = i / cols + j;
        int samplesCols = samples.rows / cols;
        if (neigbourRow > 0 && neigbourRow < samplesCols){
          for (int h = -NEIGHBOUR_SIZE / 2; h <= NEIGHBOUR_SIZE / 2; h++){
            int neighCol = i % cols + h;
            if (neighCol > 0 && neighCol < cols){
              int neighSamplesInd = i + j * cols + h;
              firstComponent = spatialGauss[abs(h + NEIGHBOUR_SIZE / 2)]
                                           [abs(j + NEIGHBOUR_SIZE / 2)];
              bilFilterRes = this->bilateralFilter(firstComponent,
                                                   samples.row(i),
                                                   samples.row(neighSamplesInd));
              float prob = this->currentIter.probs.at<double>(neighSamplesInd, k); 
              this->currentIter.smoothProbs.at<double>(i, k) += 
                bilFilterRes * prob;
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


float EMTai::bilateralFilter(float& spatialGauss, const Mat& centerI,
                             const Mat& neighI){
  float distL = pow(centerI.at<float>(0, 0) - neighI.at<float>(0, 0), 2);
  float distA = pow(centerI.at<float>(0, 1) - neighI.at<float>(0, 1), 2);
  float distB = pow(centerI.at<float>(0, 2) - neighI.at<float>(0, 2), 2);
  float colorGauss = exp(-(distL + distA + distB) / COLOR_DEV);

  return spatialGauss * colorGauss;
}


void EMTai::getMeans(Mat& means){
  for (int i = 0; i < this->clsCnt; i++){
    for (int j = 0; j < 3; j++){
      means.at<float>(i, j) = this->currentIter.covs.at<float>(i, j);
    }
  }
}


void EMTai::getCovs(Mat& covs){
  for (int i = 0; i < this->clsCnt; i++){
    for (int j = 0; j < 3; j++){
      covs.at<float>(i, j) = this->currentIter.covs.at<float>(i, j);
    }
  }
}


float EMTai::getProbs(int& x, int& y, int& clsCnt, int& cols){
  return this->currentIter.probs.at<double>(x * cols + y, clsCnt);
}
