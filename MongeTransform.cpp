#include <MongeTransform.h>
#include <cmath>

#define PI 3.14159265
#define VALUES_RANGE 256


void computeHue(const Mat& img, Mat& hue, int chAInd, int chBInd){
  for (int i = 0; i < img.rows; i++){
    for (int j = 0; j < img.cols; j++){
      hue.at<float>(i, j) = float(180) / PI * atan2(
        int(img.at<Vec3b>(i, j)[chBInd]) - 128,
        int(img.at<Vec3b>(i, j)[chAInd]) - 128);

      hue.at<float>(i, j) = (hue.at<float>(i, j) < 0) ?
                            (hue.at<float>(i, j) + 360) : hue.at<float>(i, j);
      hue.at<float>(i, j) = (hue.at<float>(i, j) > 360) ?
                            (360 - hue.at<float>(i, j)) : hue.at<float>(i, j);
    }
  }
}

// for now, implementation only for 2x2 matrices
void dotProduct(const Mat& matrix1, const Mat& matrix2, Mat& matrixProduct){
  matrixProduct.at<float>(0, 0) = matrix1.at<float>(0, 0) *
                                  matrix2.at<float>(0, 0) +
                                  matrix1.at<float>(0, 1) *
                                  matrix2.at<float>(1, 0);

  matrixProduct.at<float>(0, 1) = matrix1.at<float>(0, 0) *
                                  matrix2.at<float>(0, 1) +
                                  matrix1.at<float>(0, 1) *
                                  matrix2.at<float>(1, 1);

  matrixProduct.at<float>(1, 0) = matrix1.at<float>(1, 0) *
                                  matrix2.at<float>(0, 0) +
                                  matrix1.at<float>(1, 1) *
                                  matrix2.at<float>(1, 0);

  matrixProduct.at<float>(1, 1) = matrix1.at<float>(1, 0) *
                                  matrix2.at<float>(0, 1) +
                                  matrix1.at<float>(1, 1) *
                                  matrix2.at<float>(1, 1);
}


void matrixSquareRoot(const Mat& matrix, Mat& squareRoot){
  Mat tmpMatrix = Mat(2, 2, CV_32F);

  Mat diagMatrix = Mat::zeros(2, 2, CV_32F);
  Mat eigValues = Mat(1, 2, CV_32F);
  Mat eigVectors = Mat(2, 2, CV_32F);

  eigen(matrix, eigValues, eigVectors);

  for (int i = 0; i < 2; i++){
    diagMatrix.at<float>(i, i) = sqrt(eigValues.at<float>(0, i));
  }

  dotProduct(eigVectors.t(), diagMatrix, tmpMatrix);
  dotProduct(tmpMatrix, eigVectors, squareRoot);
}


float computeMean(const Mat& imgChannel){
  float mean = 0.0;
  int size = imgChannel.rows * imgChannel.cols;

  for (int i = 0; i < imgChannel.rows; i++){
    for (int j = 0; j < imgChannel.cols; j++){
      if (!std::isnan(imgChannel.at<float>(i, j))){
        mean += imgChannel.at<float>(i, j);
      }
    }
  }

  mean = float(mean) / size;

  return mean;
}


float computeWeightedMean(const Mat& imgChannel){
  float weightedMean = 0.0;
  float sumWeights = 0.0;
  float sumMeans = 0.0;
  int size = imgChannel.rows * imgChannel.cols;
  float* weights = new float[VALUES_RANGE];
  int* freqCounter = new int[VALUES_RANGE];

  for (int i = 0; i < VALUES_RANGE; i++){
    freqCounter[i] = 0;
  }

  for (int i = 0; i < imgChannel.rows; i++){
    for (int j = 0; j < imgChannel.cols; j++){
      if (!std::isnan(imgChannel.at<float>(i, j))){
        freqCounter[int(imgChannel.at<uchar>(i, j))] ++;
      }
    }
  }

  for (int i = 0; i < VALUES_RANGE; i++){
      weights[i] = float(freqCounter[i]) / size;
  }

  for (int i = 0; i < imgChannel.rows; i++){
    for (int j = 0; j < imgChannel.cols; j++){
      if (!std::isnan(imgChannel.at<float>(i, j))){
        sumMeans += (weights[int(imgChannel.at<uchar>(i, j))] *
          imgChannel.at<float>(i, j));
        sumWeights += weights[int(imgChannel.at<uchar>(i, j))];
      }
    }
  }

  weightedMean = float(sumMeans) / sumWeights;

  return weightedMean;
}


void computeWeightedMeanVector(const Mat& img, float*& meanVector){
  vector<Mat> channels(3);
  split(img, channels);

  meanVector[0] = computeWeightedMean(channels[1]);
  meanVector[1] = computeWeightedMean(channels[2]);
}


float computeCovarianceElement(const Mat& channelA, const Mat& channelB){
  float meanA = computeMean(channelA);
  float meanB = computeMean(channelB);
  float sumMeans = 0.0;
  float covar;

  for (int i = 0; i < channelA.rows; i++){
    for (int j = 0; j < channelA.cols; j++){
      if (!std::isnan(channelA.at<float>(i, j))){
         sumMeans += (float(channelA.at<float>(i, j)) - meanA) *
                     (float(channelB.at<float>(i, j)) - meanB);
      }
    }
  }

  covar = float(sumMeans) / (channelA.rows * channelA.cols - 1);

  return covar;
}


void computeCovariance(const Mat& img, Mat& imgCovMatrix){
  vector<Mat> channels(3);
  split(img, channels);

  imgCovMatrix.at<float>(0, 0) = computeCovarianceElement(channels[1],
                                                          channels[1]);
  imgCovMatrix.at<float>(1, 1) = computeCovarianceElement(channels[2],
                                                          channels[2]);
  imgCovMatrix.at<float>(0, 1) = computeCovarianceElement(channels[2],
                                                          channels[1]);
  imgCovMatrix.at<float>(1, 0) = imgCovMatrix.at<float>(0, 1);
}


void closedFormMatrix(const Mat& inpCovMatrix, const Mat& tarCovMatrix,
                      Mat& matrixT){
  Mat inpSqRootCovMat = Mat(2, 2, CV_32F);
  matrixSquareRoot(inpCovMatrix, inpSqRootCovMat);

  Mat term1 = Mat(2, 2, CV_32F);
  Mat term2 = Mat(2, 2, CV_32F);
  Mat middleTerm = Mat(2, 2, CV_32F);
  dotProduct(inpSqRootCovMat, tarCovMatrix, term1);
  std::cout << term1 << std::endl;
  dotProduct(term1, inpSqRootCovMat, term2);
  matrixSquareRoot(term2, middleTerm);
  std::cout << middleTerm << std::endl;

  Mat term3 = Mat(2, 2, CV_32F);
  dotProduct(inpSqRootCovMat.inv(), middleTerm, term3);

  dotProduct(term3, inpSqRootCovMat.inv(), matrixT);
}


void computeMapping(const Mat& inpImg, const Mat& tarImg,
                    float*& inpMeans, float*& tarMeans,
                    const Mat& matrixT, Mat& newValues){
  
  for (int i = 0; i < inpImg.rows; i++){
    for (int j = 0; j < inpImg.cols; j++){
      for (int k = 0; k < 2; k++){
        newValues.at<Vec2f>(i, j)[k] =
          (std::isnan(float(inpImg.at<Vec3f>(i, j)[k]))) ? NAN :
           (matrixT.at<float>(k, 0) *
           (float(inpImg.at<Vec3f>(i, j)[k + 1]) - inpMeans[0]) +
           matrixT.at<float>(k, 1) *
           (float(inpImg.at<Vec3f>(i, j)[k + 1]) - inpMeans[1]) + tarMeans[k]);
      }
    }
  }
}


void applyTransform(const Mat& inpImg, const Mat& tarImg, Mat& newValues,
                   Mat& invCovMat, Mat& means){
  Mat mu;
  Mat inpCovMatrix = Mat(2, 2, CV_32F);
  Mat tarCovMatrix = Mat(2, 2, CV_32F);

  computeCovariance(inpImg, inpCovMatrix);
  computeCovariance(tarImg, tarCovMatrix);

  invCovMat = inpCovMatrix.inv();

  float* inpMeans = new float[2];
  float* tarMeans = new float[2];

  computeWeightedMeanVector(inpImg, inpMeans);
  computeWeightedMeanVector(tarImg, tarMeans);

  std::cout << inpMeans[0] << std::endl;

  means.at<float>(0, 0) = inpMeans[0];
  means.at<float>(0, 1) = inpMeans[1];

  Mat matrixT = Mat(2, 2, CV_32F);

  closedFormMatrix(inpCovMatrix, tarCovMatrix, matrixT);
  std::cout << "matrixT" << matrixT << std::endl;

  computeMapping(inpImg, tarImg, inpMeans, tarMeans, matrixT, newValues);
}
