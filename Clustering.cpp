#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <Clustering.h>

using namespace cv;

#define PROB_THRESHOLD 0.1


void getClusterMap(const Mat& img, Mat& map, Mat& samples, EM& clusters,
                   int clusterNmb, int clsCnt){
  int rowNmb = 0;
  Vec2d probComp;

  for (int i = 0; i < img.rows; i++){
    for (int j = 0; j < img.cols; j++){
      Mat sampleProbs = Mat(1, clsCnt, CV_64FC1);
      probComp = clusters.predict(samples.row(rowNmb), sampleProbs);

      if (sampleProbs.at<double>(0, clusterNmb) > PROB_THRESHOLD){
        map.at<Vec3f>(i, j)[0] = float(img.at<Vec3b>(i, j)[0]);
        map.at<Vec3f>(i, j)[1] = float(img.at<Vec3b>(i, j)[1]);
        map.at<Vec3f>(i, j)[2] = float(img.at<Vec3b>(i, j)[2]);
      }
      rowNmb++;
    }
  }
}


double computeProbs(const Mat& sample, const Mat& mean, const Mat& invCovar){
  double prob, distance;
  //std::cout << sample << " " << mean << " " << std::endl;

  distance = Mahalanobis(sample, mean, invCovar);

  //std::cout << distance << std::endl;
  prob = double(1) / exp(distance);
  //std::cout << prob << std::endl;

  return prob;
}


void normalize(double*& probs, int size){
  double sum = 0;
  for (int i = 0; i < size; i++){
  sum += probs[i];
  }

  for (int i = 0; i < size; i++){
    probs[i] = probs[i] / sum;
  }
}
