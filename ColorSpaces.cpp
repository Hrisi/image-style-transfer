#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ColorSpaces.h>
#include <MongeTransform.h>

using namespace cv;

#define GREY_THRESHOLD 0.3


float clipImg(float value, float lower, float upper){
  return std::max(lower, std::min(value, upper));
}


float sRGBCompanding(float value){
  if (value <= 0.0031308){
    return value * 12.92;
  }
  else{
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
  }
}


float sRGBInverseCompanding(float value){
  if (value <= 0.04045){
    return value / 12.92;
  }
  else{
    return pow((value + 0.055)/ 1.055, 2.4);
  }
}


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


void BGRtoYUV(const Mat& bgrImg, Mat& yuvImg){
  Mat matrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114,
                                     -0.229, -0.587, 0.886,
                                     0.701, -0.587, -0.114);

  for (int k = 0; k < 3; k++){
    for (int i = 0; i < bgrImg.rows; i++){
      for (int j = 0; j < bgrImg.cols; j++){
        yuvImg.at<Vec3f>(i, j).val[k] = (matrix.at<float>(k, 0) *
                                         bgrImg.at<Vec3f>(i, j).val[2] +
                                         matrix.at<float>(k, 1) *
                                         bgrImg.at<Vec3f>(i, j).val[1] +
                                         matrix.at<float>(k, 2) *
                                         bgrImg.at<Vec3f>(i, j).val[0]);
      }
    }
  }
  //std::cout << yuvImg << std::endl;
}


bool computeBGRWhitePoint(const Mat& bgrImg, Mat& whitePoint,
                          Mat& previousWhitePoint){
  float sum[3] = {0, 0, 0};
  int cnt = 0;
  Mat matrix = (Mat_<float>(3, 3) << 0.299, 0.587, 0.114,
                                     -0.229, -0.587, 0.886,
                                     0.701, -0.587, -0.114);
  Mat invMatrix = matrix.inv();
  Mat tmpWhitePoint = Mat(1, 1, CV_32FC3);

  Mat bgrImgNonNorm = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);
  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      for(int k = 0; k < 3; k++){
        bgrImgNonNorm.at<Vec3f>(i, j).val[k] = 
          bgrImg.at<Vec3f>(i, j).val[k] * 255.0;
      }
    }
  }

  Mat yuvImg = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);

  BGRtoYUV(bgrImgNonNorm, yuvImg);

  float maxLum = yuvImg.at<Vec3f>(0, 0).val[0];
  for (int i = 0; i < yuvImg.rows; i++){
    for (int j = 0; j < yuvImg.cols; j++){
      if (maxLum < yuvImg.at<Vec3f>(i, j).val[0]){
        maxLum = yuvImg.at<Vec3f>(i, j).val[0];
      }
      float ratio = (abs(yuvImg.at<Vec3f>(i, j).val[1]) +
        abs(yuvImg.at<Vec3f>(i, j).val[2])) / yuvImg.at<Vec3f>(i, j).val[0];
      if (ratio < GREY_THRESHOLD){
        sum[0] += yuvImg.at<Vec3f>(i, j).val[0];
        sum[1] += yuvImg.at<Vec3f>(i, j).val[1];
        sum[2] += yuvImg.at<Vec3f>(i, j).val[2];
        cnt++;
      }
    }
  }

  tmpWhitePoint.at<Vec3f>(0, 0).val[0] = 100;//sum[0] / cnt;
  tmpWhitePoint.at<Vec3f>(0, 0).val[1] = sum[1] / cnt;
  tmpWhitePoint.at<Vec3f>(0, 0).val[2] = sum[2] / cnt;

  // termination criteria
  float norm = sqrt(
    (previousWhitePoint.at<Vec3f>(0, 0).val[1] - tmpWhitePoint.at<Vec3f>(0, 0).val[1]) *
    (previousWhitePoint.at<Vec3f>(0, 0).val[1] - tmpWhitePoint.at<Vec3f>(0, 0).val[1]) +
    (previousWhitePoint.at<Vec3f>(0, 0).val[2] - tmpWhitePoint.at<Vec3f>(0, 0).val[2]) *
    (previousWhitePoint.at<Vec3f>(0, 0).val[2] - tmpWhitePoint.at<Vec3f>(0, 0).val[2]));
  if ((!std::isnan(previousWhitePoint.at<Vec3f>(0, 0).val[1]) &&
      norm < pow(10, -6)) ||
      std::max(abs(tmpWhitePoint.at<Vec3f>(0, 0).val[1]),
               abs(tmpWhitePoint.at<Vec3f>(0, 0).val[2])) < 0.001){
    return true;
  }

  for (int k = 0; k < 3; k++){
    previousWhitePoint.at<Vec3f>(0, 0).val[k] = tmpWhitePoint.at<Vec3f>(0, 0).val[k];
  }
  
  for (int i = 0; i < 3; i++){
    whitePoint.at<Vec3f>(0, 0).val[2 - i] = (invMatrix.at<float>(i, 0) *
                                             tmpWhitePoint.at<Vec3f>(0, 0).val[0] +
                                             invMatrix.at<float>(i, 1) *
                                             tmpWhitePoint.at<Vec3f>(0, 0).val[1] +
                                             invMatrix.at<float>(i, 2) *
                                             tmpWhitePoint.at<Vec3f>(0, 0).val[2]) / 255.0;
    whitePoint.at<Vec3f>(0, 0).val[2 - i] =
      clipImg(whitePoint.at<Vec3f>(0, 0).val[2 - i], 0, 1);
  }

  return false;
}


void XYZtoxy(const Mat& XYZImg, Mat& xyImg){
  float sum =
    XYZImg.at<Vec3f>(0, 0).val[0] +
    XYZImg.at<Vec3f>(0, 0).val[1] +
    XYZImg.at<Vec3f>(0, 0).val[2];

  xyImg.at<Vec2f>(0, 0).val[0] = XYZImg.at<Vec3f>(0, 0).val[0] / sum;
  xyImg.at<Vec2f>(0, 0).val[1] = XYZImg.at<Vec3f>(0, 0).val[1] / sum;
}


void xytoXYZ(const Mat& xyImg, float maxLuminance, Mat& XYZImg){
  XYZImg.at<Vec3f>(0, 0).val[0] = 
    xyImg.at<Vec2f>(0, 0).val[0] * maxLuminance / xyImg.at<Vec2f>(0, 0).val[1];
  XYZImg.at<Vec3f>(0, 0).val[1] = maxLuminance;
  XYZImg.at<Vec3f>(0, 0).val[2] = 
    (1 - xyImg.at<Vec2f>(0, 0).val[0] - xyImg.at<Vec2f>(0, 0).val[1]) *
    maxLuminance / xyImg.at<Vec2f>(0, 0).val[1];
}


void getRGBtoXYZMatrix(const Mat& whitePoint, Mat& RGBtoXYZMatrix){
  // using the sRGB chromaticity coordinates
  float xR = 0.64;
  float yR = 0.33;
  float xG = 0.3;
  float yG = 0.6;
  float xB = 0.15;
  float yB = 0.06;

  float XR, YR, ZR, XB, YB, ZB, XG, YG, ZG;

  XR = xR / yR;
  YR = 1;
  ZR = (1 - xR - yR) / yR;

  XG = xG / yG;
  YG = 1;
  ZG = (1 - xG - yG) / yG;

  XB = xB / yB;
  YB = 1;
  ZB = (1 - xB - yB) / yB;

  Mat matrix = (Mat_<float>(3, 3) << XR, XG, XB,
                                     YR, YG, YB,
                                     ZR, ZG, ZB);
  Mat invMatrix = matrix.inv();
  
  float sR, sG, sB;
  sR = invMatrix.at<float>(0, 0) * whitePoint.at<Vec3f>(0, 0).val[0] +
       invMatrix.at<float>(0, 1) * whitePoint.at<Vec3f>(0, 0).val[1] +
       invMatrix.at<float>(0, 2) * whitePoint.at<Vec3f>(0, 0).val[2];

  sG = invMatrix.at<float>(1, 0) * whitePoint.at<Vec3f>(0, 0).val[0] +
       invMatrix.at<float>(1, 1) * whitePoint.at<Vec3f>(0, 0).val[1] +
       invMatrix.at<float>(1, 2) * whitePoint.at<Vec3f>(0, 0).val[2];

  sB = invMatrix.at<float>(2, 0) * whitePoint.at<Vec3f>(0, 0).val[0] +
       invMatrix.at<float>(2, 1) * whitePoint.at<Vec3f>(0, 0).val[1] +
       invMatrix.at<float>(2, 2) * whitePoint.at<Vec3f>(0, 0).val[2];

  RGBtoXYZMatrix = (Mat_<float>(3, 3) << sR * XR, sG * XG, sB * XB,
                                         sR * YR, sG * YG, sB * YB,
                                         sR * ZR, sG * ZG, sB * ZB);
}


void BGRtoXYZWhitePoint(Mat& bgrImg, const Mat& whitePoint, Mat& xyzImg){
  Mat M = Mat(3, 3, CV_32F);
  getRGBtoXYZMatrix(whitePoint, M);

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      for (int k = 0; k < 3; k++){
        bgrImg.at<Vec3f>(i, j).val[k] =
          sRGBInverseCompanding(bgrImg.at<Vec3f>(i, j).val[k]);
      }
      for (int k = 0; k < 3; k++){
        xyzImg.at<Vec3f>(i, j).val[k] = 
          M.at<float>(k, 0) * bgrImg.at<Vec3f>(i, j).val[2] +
          M.at<float>(k, 1) * bgrImg.at<Vec3f>(i, j).val[1] +
          M.at<float>(k, 2) * bgrImg.at<Vec3f>(i, j).val[0];
      }
    }
  }
}


void adaptXYZToNewWhitePoint(Mat& XYZWhitePoint, const Mat& BGRWhitePoint){
  Mat sRGBChromCoord = (Mat_<float>(3, 3) << 0.6400, 0.3000, 0.1500,
                                             0.3300, 0.6000, 0.0600, 	
                                             0.2126, 0.7152, 0.0722);
  Mat invsRGBChromCoord = sRGBChromCoord.inv();

  Mat XYZD65 = (Mat_<float>(1, 3) << 0.9642, 1.0000, 0.8249);
  Mat tmpWhitePoint = Mat(BGRWhitePoint.rows, BGRWhitePoint.cols, CV_32FC3);
  for (int k = 0; k < 3; k++){
    tmpWhitePoint.at<Vec3f>(0, 0).val[k] = 
      invsRGBChromCoord.at<float>(k, 0) * XYZD65.at<float>(0, 0) +
      invsRGBChromCoord.at<float>(k, 1) * XYZD65.at<float>(0, 1) +
      invsRGBChromCoord.at<float>(k, 2) * XYZD65.at<float>(0, 2);
  }

  Mat T = (Mat_<float>(3, 3) << tmpWhitePoint.at<Vec3f>(0, 0).val[0], 0, 0,
                                0, tmpWhitePoint.at<Vec3f>(0, 0).val[1], 0,
                                0, 0, tmpWhitePoint.at<Vec3f>(0, 0).val[2]);

  Mat tmpMat = Mat(3, 3, CV_32F);
  dotProduct(sRGBChromCoord, T, tmpMat);
  Mat tmpBGRWhitePoint = Mat(XYZWhitePoint.rows, XYZWhitePoint.cols, CV_32FC3);

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  for (int i = 0; i < 3; i++){
    tmpBGRWhitePoint.at<Vec3f>(0, 0).val[i] =
      /*sRGBInverseCompanding(*/BGRWhitePoint.at<Vec3f>(0, 0).val[i];//);
  }
  std::cout << "BGR white" << BGRWhitePoint << std::endl;
  for (int k = 0; k < 3; k++){
    XYZWhitePoint.at<Vec3f>(0, 0).val[k] = 
      tmpBGRWhitePoint.at<float>(0, 2) * tmpMat.at<float>(k, 0) +
      tmpBGRWhitePoint.at<float>(0, 1) * tmpMat.at<float>(k, 1) +
      tmpBGRWhitePoint.at<float>(0, 0) * tmpMat.at<float>(k, 2);
  }
}


void CAT(Mat& bgrImg, Mat& inpWhitePoint, const Mat& tarWhitePoint){
  Mat inpXYZWhitePoint = Mat(inpWhitePoint.rows, inpWhitePoint.cols, CV_32FC3);
  Mat tmpWhitePoint = Mat(inpWhitePoint.rows, inpWhitePoint.cols, CV_32FC2);
  adaptXYZToNewWhitePoint(inpXYZWhitePoint, inpWhitePoint);

  Mat tarXYZWhitePoint = Mat(tarWhitePoint.rows, tarWhitePoint.cols, CV_32FC3);
  Mat tarXYWhitePoint = Mat(tarWhitePoint.rows, tarWhitePoint.cols, CV_32FC2);
  adaptXYZToNewWhitePoint(tarXYZWhitePoint, tarWhitePoint);

  Mat xyzImg = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);
  Mat resXYZImg = Mat(bgrImg.rows, bgrImg.cols, CV_32FC3);
  Mat XYZtoRGBMatrix = Mat(3, 3, CV_32F);

  BGRtoXYZWhitePoint(bgrImg, inpXYZWhitePoint, xyzImg);

  Mat BSPositive = (Mat_<float>(3, 3) << 0.7328, 0.4296, -0.1624,
                                         -0.7036, 1.6975, 0.0061,
                                         0.0030, -0.0136, 0.9834);

  float roI = BSPositive.at<float>(0, 0) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[0] +
              BSPositive.at<float>(0, 1) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[1] +
              BSPositive.at<float>(0, 2) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[2];
  
  float gamaI = BSPositive.at<float>(1, 0) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[0] +
                BSPositive.at<float>(1, 1) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[1] +
                BSPositive.at<float>(1, 2) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[2];

  float betaI = BSPositive.at<float>(2, 0) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[0] +
                BSPositive.at<float>(2, 1) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[1] +
                BSPositive.at<float>(2, 2) * inpXYZWhitePoint.at<Vec3f>(0, 0).val[2];

  float roT = BSPositive.at<float>(0, 0) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[0] +
              BSPositive.at<float>(0, 1) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[1] +
              BSPositive.at<float>(0, 2) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[2];
  
  float gamaT = BSPositive.at<float>(1, 0) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[0] +
                BSPositive.at<float>(1, 1) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[1] +
                BSPositive.at<float>(1, 2) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[2];

  float betaT = BSPositive.at<float>(2, 0) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[0] +
                BSPositive.at<float>(2, 1) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[1] +
                BSPositive.at<float>(2, 2) * tarXYZWhitePoint.at<Vec3f>(0, 0).val[2];

  Mat diagMatrix = (Mat_<float>(3, 3) << roT / roI, 0, 0,
                                         0, gamaT / gamaI, 0,
                                         0, 0, betaT / betaI);

  Mat tmpMatrix = Mat(3, 3, CV_32F);
  dotProduct(BSPositive.inv(), diagMatrix, tmpMatrix);
  Mat M = Mat(3, 3, CV_32F);
  dotProduct(tmpMatrix, BSPositive, M);

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      for (int k = 0; k < 3; k++){
        resXYZImg.at<Vec3f>(i, j).val[k] = 
          M.at<float>(k, 0) * xyzImg.at<Vec3f>(i, j).val[0] +
          M.at<float>(k, 1) * xyzImg.at<Vec3f>(i, j).val[1] +
          M.at<float>(k, 2) * xyzImg.at<Vec3f>(i, j).val[2];
        resXYZImg.at<Vec3f>(i, j).val[k] =
          clipImg(resXYZImg.at<Vec3f>(i, j).val[k], 0, 1);
      }
    }
  }

  getRGBtoXYZMatrix(inpXYZWhitePoint, XYZtoRGBMatrix);
  XYZtoRGBMatrix = XYZtoRGBMatrix.inv();

  for (int i = 0; i < bgrImg.rows; i++){
    for (int j = 0; j < bgrImg.cols; j++){
      for (int k = 0; k < 3; k++){
        bgrImg.at<Vec3f>(i, j).val[2 - k] = 
          XYZtoRGBMatrix.at<float>(k, 0) * resXYZImg.at<Vec3f>(i, j).val[0] +
          XYZtoRGBMatrix.at<float>(k, 1) * resXYZImg.at<Vec3f>(i, j).val[1] +
          XYZtoRGBMatrix.at<float>(k, 2) * resXYZImg.at<Vec3f>(i, j).val[2];
        bgrImg.at<Vec3f>(i, j).val[2 - k] =
          sRGBCompanding(bgrImg.at<Vec3f>(i, j).val[2 - k]);
        bgrImg.at<Vec3f>(i, j).val[2 - k] =
          clipImg(bgrImg.at<Vec3f>(i, j).val[2 - k], 0, 1);
      }
    }
  }
}
