#include <iostream>
#include <opencv2/opencv.hpp>

#include <ReinhardTransform.h>
#include <ColorSpaces.h>
#include <EM.h>

#define CLS_NMB 3

using namespace cv;


void mappingFunction(const Mat& inpMean, const Mat& tarMean, int* map){
  float dist;
  for (int i = 0; i < CLS_NMB; i++){
    dist = sqrt(pow(inpMean.at<float>(i, 0) - tarMean.at<float>(0, 0), 2));
    map[i] = 0;
    for (int j = 0; j < 3; j++){
      float newDist = sqrt(pow(inpMean.at<float>(i, 0) -
                               tarMean.at<float>(j, 0), 2));
      if (newDist < dist){
        dist = newDist;
        map[i] = j;
      }
    std::cout << map[i] << std::endl;
    }
  }
}


void normalize(const Mat& img, Mat& imgNorm){
  for (int i = 0; i < img.rows; i++){
    for (int j = 0; j < img.cols; j++){
      for (int k = 0; k < 3; k++){
        imgNorm.at<Vec3f>(i, j).val[k] =
          img.at<Vec3b>(i, j).val[k] / 255.0;
      }
    }
  }
}


void reshape(const Mat& img, Mat& samples){
  int row = 0;
  for (int i = 0; i < img.rows; i++){
    for (int j = 0; j < img.cols; j++){
      for (int k = 0; k < 3; k++){
        samples.at<float>(row, k) = img.at<Vec3f>(i, j).val[k];
      }
      row++;
    }
  }
}


int main(int argc, char** argv){
  // 3 arguments: file name, input image, target image, result image
  if (argc < 4){
    std::cout << "No image data \n";
    return -1;
  }

  Mat inpImg, tarImg, finalImg;

  inpImg = imread(argv[1], 1);
  tarImg = imread(argv[2], 1);

  Mat inpImgNorm = Mat(inpImg.rows, inpImg.cols, CV_32FC3);
  Mat tarImgNorm = Mat(tarImg.rows, tarImg.cols, CV_32FC3);

  normalize(inpImg, inpImgNorm);
  normalize(tarImg, tarImgNorm);

  Mat inpImgLab = Mat(inpImg.rows, inpImg.cols, CV_32FC3);
  Mat tarImgLab = Mat(tarImg.rows, tarImg.cols, CV_32FC3);
  Mat inpImgLms = Mat(inpImg.rows, inpImg.cols, CV_32FC3);
  Mat tarImgLms = Mat(tarImg.rows, tarImg.cols, CV_32FC3);

  convertBGRtoLMS(inpImgNorm, inpImgLms);
  convertLMStoLab(inpImgLms, inpImgLab);
  convertBGRtoLMS(tarImgNorm, tarImgLms);
  convertLMStoLab(tarImgLms, tarImgLab);

  EMTai inpClusters = EMTai(CLS_NMB);
  EMTai tarClusters = EMTai(CLS_NMB);

  Mat inpSamples = Mat(inpImgLab.rows * inpImgLab.cols, 3, CV_32F);
  Mat tarSamples = Mat(tarImgLab.rows * tarImgLab.cols, 3, CV_32F);

  reshape(inpImgLab, inpSamples);
  reshape(tarImgLab, tarSamples);

  tarClusters.train(tarSamples, tarImgLab.cols);
  inpClusters.train(inpSamples, inpImgLab.cols);

  Mat inpMean = Mat(CLS_NMB, 3, CV_32F);
  inpClusters.getMeans(inpMean);
  Mat tarMean = Mat(CLS_NMB, 3, CV_32F);
  tarClusters.getMeans(tarMean);
  Mat inpCov = Mat(CLS_NMB, 3, CV_32F);
  inpClusters.getCovs(inpCov);
  Mat tarCov = Mat(CLS_NMB, 3, CV_32F);
  tarClusters.getCovs(tarCov);

  int* map = new int[CLS_NMB];
  mappingFunction(inpMean, tarMean, map);
  Mat imgAfterTransf = Mat::zeros(inpImg.rows, inpImg.cols, CV_32FC3);

  for (int k = 0; k < 3; k++){
    for (int i = 0; i < inpImgLab.rows; i++){
      for (int j = 0; j < inpImgLab.cols; j++){
        for (int h = 0; h < CLS_NMB; h++){
          float newValue = applyReinhard(
              inpImgLab.at<Vec3f>(i, j).val[k],
              inpMean.at<float>(h, k),
              tarMean.at<float>(map[h], k),
              inpCov.at<float>(h, k),
              tarCov.at<float>(map[h], k));
          float prob = inpClusters.getProbs(i, j, h, inpImgLab.cols);
          imgAfterTransf.at<Vec3f>(i, j).val[k] += prob * newValue;
        }
      }
    }
  }

  convertLabtoLMS(imgAfterTransf, inpImgLms);
  convertLMStoBGR(inpImgLms, inpImgNorm);

  for (int i = 0; i < inpImg.rows; i++){
    for (int j = 0; j < inpImg.cols; j++){
      for (int k = 0; k < 3; k++){
        inpImgNorm.at<Vec3f>(i, j).val[k] *= 255.0;
      }
    }
  }

  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", inpImgNorm);
  imwrite("testTai.jpg", inpImgNorm);

  waitKey(0);

  return 0;
}
