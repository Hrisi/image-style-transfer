#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ColorSpaces.h>
#include <MongeTransform.h>

using namespace cv;


void conversion(const Mat& original, Mat& result, const Mat& matrix,
                int rows, int cols){
  for (int k = 0; k < 3; k++){
    for (int i = 0; i < rows; i++){
      for (int j = 0; j < cols; j++){
        result.at<Vec3f>(i, j)[k] = matrix.at<float>(k, 0) *
                                    float(original.at<Vec3f>(i, j)[0]) +
                                    matrix.at<float>(k, 1) *
                                    float(original.at<Vec3f>(i, j)[1]) +
                                    matrix.at<float>(k, 2) *
                                    float(original.at<Vec3f>(i, j)[2]);
      }
    }
  }
}


void BGRtoLMS(const Mat& bgrImg, Mat& lmsImg){
  Mat matrix = (Mat_<float>(3, 3) << 0.3811, 0.5783, 0.0402,
                                     0.1967, 0.7244, 0.0782,
                                     0.0241, 0.1288, 0.8444);

  for (int k = 0; k < 3; k++){
    for (int i = 0; i < bgrImg.rows; i++){
      for (int j = 0; j < bgrImg.cols; j++){
        lmsImg.at<Vec3f>(i, j)[k] = log10(matrix.at<float>(k, 0) *
                                          float(bgrImg.at<Vec3b>(i, j)[2]) / 255 +
                                          matrix.at<float>(k, 1) *
                                          float(bgrImg.at<Vec3b>(i, j)[1]) / 255+
                                          matrix.at<float>(k, 2) *
                                          float(bgrImg.at<Vec3b>(i, j)[0]) / 255);
      }
    }
  }
}


void LMStoLab(const Mat& lmsImg, Mat& labImg){
  Mat base1 = (Mat_<float>(3, 3) << 0.57735026918, 0, 0,
                                    0, 0.40824829046, 0,
                                    0, 0, 0.70710678118);
  Mat base2 = (Mat_<float>(3, 3) << 1, 1, 1,
                                    1, 1, -2,
                                    1, -1, 0);
  Mat matrix = Mat(3, 3, CV_32F);
  dotProduct(base1, base2, matrix);

  conversion(lmsImg, labImg, matrix, lmsImg.rows, lmsImg.cols);
}


void LabtoLMS(const Mat& labImg, Mat& lmsImg){
  Mat base1 = (Mat_<float>(3, 3) << 1, 1, 1,
                                    1, 1, -1,
                                    1, -2, 0);
  Mat base2 = (Mat_<float>(3, 3) << 0.57735026918, 0, 0,
                                    0, 0.40824829046, 0,
                                    0, 0, 0.7071067811);
  Mat matrix = Mat(3, 3, CV_32F);
  dotProduct(base1, base2, matrix);

  conversion(labImg, lmsImg, matrix, lmsImg.rows, lmsImg.cols);

  for (int k = 0; k < 3; k++){
    for (int i = 0; i < lmsImg.rows; i++){
      for (int j = 0; i < lmsImg.rows; i++){
        lmsImg.at<Vec3b>(i, j)[k] = pow(float(lmsImg.at<Vec3b>(i, j)[k]), 10);
      }
    }
  }
}


void LMStoBGR(const Mat& lmsImg, Mat& bgrImg){
  Mat matrix = (Mat_<float>(3, 3) << 4.4679, -3.5873, 0.1193,
                                    -1.2186, 2.3809, -0.1624,
                                    0.0497, -0.2439, 1.2045);

  for (int k = 0; k < 3; k++){
    for (int i = 0; i < bgrImg.rows; i++){
      for (int j = 0; j < bgrImg.cols; j++){
        bgrImg.at<Vec3b>(i, j)[2 - k] = (matrix.at<float>(k, 0) *
                                        float(lmsImg.at<Vec3f>(i, j)[0]) +
                                        matrix.at<float>(k, 1) *
                                        float(lmsImg.at<Vec3f>(i, j)[1]) +
                                        matrix.at<float>(k, 2) *
                                        float(lmsImg.at<Vec3f>(i, j)[2])) * 255;
      }
    }
  }
}
