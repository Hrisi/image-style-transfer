#include <iostream>
#include <opencv2/opencv.hpp>

#include <ReinhardTransform.h>
#include <ColorSpaces.h>
#include <EM.h>

#define CLS_NMB 3

using namespace cv;


void mappingFunction(const Mat& inpMean, const Mat& tarMean, int* mappingSeq){
  float distance;

  for (int i = 0; i < CLS_NMB; i++){
    distance = abs(inpMean.at<float>(0, i) - tarMean.at<float>(0, 0));
    mappingSeq[i] = 0;
    for (int j = 0; j < CLS_NMB; j++){
      if (abs(inpMean.at<float>(0, i) - tarMean.at<float>(0, j)) < distance){
        distance = abs(inpMean.at<float>(0, i) - tarMean.at<float>(0, j));
        mappingSeq[i] = j;
      }
    std::cout << mappingSeq[i] << std::endl;
    }
  }
  mappingSeq[0] = 0;
  mappingSeq[1] = 1;
  mappingSeq[2] = 2;
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

  for (int i = 0; i < inpImg.rows; i++){
    for (int j = 0; j < inpImg.cols; j++){
      for (int k = 0; k < 3; k++){
        inpImgNorm.at<Vec3f>(i, j).val[k] =
          inpImg.at<Vec3b>(i, j).val[k] / 255.0;
      }
    }
  }

  for (int i = 0; i < tarImg.rows; i++){
    for (int j = 0; j < tarImg.cols; j++){
      for (int k = 0; k < 3; k++){
        tarImgNorm.at<Vec3f>(i, j).val[k] =
          tarImg.at<Vec3b>(i, j).val[k] / 255.0;
      }
    }
  }

  Mat inpImgLab = Mat(inpImg.rows, inpImg.cols, CV_32FC3);
  Mat tarImgLab = Mat(tarImg.rows, tarImg.cols, CV_32FC3);
  Mat inpLmsImg = Mat(inpImg.rows, inpImg.cols, CV_32FC3);
  Mat tarLmsImg = Mat(tarImg.rows, tarImg.cols, CV_32FC3);

  Mat newValues = Mat::zeros(inpImg.rows, inpImg.cols, CV_32FC3);

  convertBGRtoLMS(inpImgNorm, inpLmsImg);
  std::cout << "after bgr" << std::endl;
  convertLMStoLab(inpLmsImg, inpImgLab);
  std::cout << "after lms" << std::endl;
  convertBGRtoLMS(tarImgNorm, tarLmsImg);
  std::cout << "after lab" << std::endl;
  convertLMStoLab(tarLmsImg, tarImgLab);

  EMTai inpClusters = EMTai(CLS_NMB);
  EMTai tarClusters = EMTai(CLS_NMB);
  std::cout << "after em" << std::endl;

  Mat inpSamples = Mat(inpImgLab.rows * inpImgLab.cols, 3, CV_32F);
  Mat tarSamples = Mat(tarImgLab.rows * tarImgLab.cols, 3, CV_32F);
  std::cout << "after em" << std::endl;

  int rowInd = 0;
  for (int i = 0; i < inpImgLab.rows; i++){
    for (int j = 0; j < inpImgLab.cols; j++){
      for (int k = 0; k < 3; k++){
        inpSamples.at<float>(rowInd, k) = inpImgLab.at<Vec3f>(i, j).val[k];
      }
      rowInd++;
    }
  }
  std::cout << "here" << std::endl;

  rowInd = 0;
  for (int i = 0; i < tarImgLab.rows; i++){
    for (int j = 0; j < tarImgLab.cols; j++){
      for (int k = 0; k < 3; k++){
        tarSamples.at<float>(rowInd, k) = tarImgLab.at<Vec3f>(i, j).val[k];
        //std::cout << tarImgLab.at<Vec3f>(i, j).val[k] << std::endl;
      }
      rowInd++;
    }
  }
  std::cout << tarImgLab.at<Vec3f>(20, 54).val[1] << std::endl;

  tarClusters.train(tarSamples, tarImgLab.cols);
  inpClusters.train(inpSamples, inpImgLab.cols);
  std::cout << "end of train" << std::endl;
  
  std::cout << "after train" << std::endl;

  Mat inpMean = Mat(CLS_NMB, 3, CV_32F);
  inpClusters.getMeans(inpMean);
  Mat tarMean = Mat(CLS_NMB, 3, CV_32F);
  tarClusters.getMeans(tarMean);
  Mat inpCov = Mat(CLS_NMB, 3, CV_32F);
  inpClusters.getCovs(inpCov);
  Mat tarCov = Mat(CLS_NMB, 3, CV_32F);
  tarClusters.getCovs(tarCov);
  float tranfRes;
  std::cout << tarMean << " tarMeans" << std::endl;
  std::cout << inpMean << " inpMeans" << std::endl;
  std::cout << inpCov << " inpCov" << std::endl;
  std::cout << tarCov << " tarCov" << std::endl;

  int* mappingSeq = new int[CLS_NMB];
  mappingFunction(inpMean, tarMean, mappingSeq);
  std::cout << mappingSeq[0] << mappingSeq[1] << mappingSeq[2] << std::endl;

  for (int k = 0; k < 3; k++){
    for (int i = 0; i < inpImgLab.rows; i++){
      for (int j = 0; j < inpImgLab.cols; j++){
        for (int h = 0; h < CLS_NMB; h++){
          //inpImgLab.at<Vec3f>(i, j).val[k] =
          //  clipImg(inpImgLab.at<Vec3f>(i, j).val[k], 0, 1);
          tranfRes = applyReinhard(inpImgLab.at<Vec3f>(i, j).val[k],
                                   inpMean.at<float>(h, k),
                                   tarMean.at<float>(mappingSeq[h], k),
                                   inpCov.at<float>(h, k),
                                   tarCov.at<float>(mappingSeq[h], k));

          newValues.at<Vec3f>(i, j).val[k] += 
            inpClusters.getProbs(i, j, h, inpImgLab.cols) * tranfRes;
          //std::cout << newValues.at<Vec3f>(i, j).val[k] << std::endl;
        }
      }
    }
  }

  convertLabtoLMS(newValues, inpLmsImg);
  convertLMStoBGR(inpLmsImg, inpImgNorm);

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
